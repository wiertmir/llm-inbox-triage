"""Calendar creation uses the first structured triage, never another LLM call."""

import asyncio
import ast
import inspect
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402


def event_arguments(**overrides):
    return {
        "title": "Planning", "start": "2026-10-05", "end": "2026-10-06",
        "description": "Discuss plans", **overrides,
    }


def make_result(*, proposals=True, **overrides):
    data = {
        "id": "***stdin***",
        "category": triage.Category.other, "priority": 3, "summary": "Planning on October 5.",
        "extracted": triage.Extracted(dates=[date(2026, 10, 5)]),
        "proposed_events": [event_arguments()] if proposals else [],
        **overrides,
    }
    return triage.TriageResult.model_validate(data)


@pytest.fixture
def calendar(monkeypatch):
    writes = []

    def write(backend, title, start, end, description):
        writes.append((backend, title, start, end, description))
        return {"id": f"event_{len(writes)}", "htmlLink": "https://calendar.google.com/"}

    def unexpected_llm(*args, **kwargs):
        pytest.fail("Calendar creation must not call any LLM")

    monkeypatch.setattr(triage, "create_calendar_entry", write)
    monkeypatch.setattr(triage, "AsyncOpenAI", unexpected_llm)
    monkeypatch.setattr(triage, "AsyncAnthropic", unexpected_llm)
    return writes


def test_create_events_uses_only_first_triage_result(calendar):
    created = asyncio.run(triage.create_events(make_result()))
    assert calendar == [
        ("google", "Planning", date(2026, 10, 5), date(2026, 10, 6), "Discuss plans")
    ]
    assert created[0].id == "event_1"
    assert created[0].url == "https://calendar.google.com/"


def test_dates_without_proposals_do_not_write_or_authorize(calendar):
    assert asyncio.run(triage.create_events(make_result(proposals=False))) == []
    assert not calendar


def test_legacy_triage_results_default_to_no_proposals(calendar):
    payload = make_result().model_dump(mode="json")
    del payload["proposed_events"]
    result = triage.TriageResult.model_validate(payload)
    assert result.proposed_events == []
    assert asyncio.run(triage.create_events(result)) == []
    assert not calendar


@pytest.mark.parametrize(
    "overrides",
    [
        {"start": "2026-10-07", "end": "2026-10-08"},
        {"end": "2026-10-05"},
        {"title": " "},
        {"start": "2026-10-05T09:00:00", "end": "2026-10-05T10:00:00"},
        {"start": "2026-10-05T09:00:00+09:00", "end": "2026-10-06"},
        {"extra": "not allowed"},
    ],
)
def test_first_triage_validates_proposals(overrides, calendar):
    with pytest.raises(ValidationError):
        make_result(proposed_events=[event_arguments(**overrides)])
    assert not calendar


def test_mutated_batch_is_validated_before_any_writes(calendar):
    result = make_result()
    result.proposed_events.append(triage.CalendarEvent.model_validate(event_arguments(
        start="2026-10-07", end="2026-10-08",
    )))
    with pytest.raises(triage.CalendarToolError, match="no events created"):
        asyncio.run(triage.create_events(result))
    assert not calendar


def test_deadline_without_other_dates_can_create_event(calendar):
    result = make_result(extracted=triage.Extracted(deadlines=[date(2026, 10, 5)]))
    assert len(asyncio.run(triage.create_events(result))) == 1


def test_explicit_timed_event_retains_timezone(calendar):
    result = make_result(proposed_events=[event_arguments(
        start="2026-10-05T09:00:00+09:00", end="2026-10-05T10:00:00+09:00",
    )])
    created = asyncio.run(triage.create_events(result))
    assert created[0].start.isoformat() == "2026-10-05T09:00:00+09:00"
    assert calendar[0][2].isoformat() == "2026-10-05T09:00:00+09:00"


def test_identical_proposals_write_once(calendar):
    result = make_result(proposed_events=[event_arguments(), event_arguments()])
    assert len(asyncio.run(triage.create_events(result))) == 1
    assert len(calendar) == 1


def test_calendar_failure_is_not_retried(calendar, monkeypatch):
    calls = []

    def fail(*args):
        calls.append(args)
        raise triage.CalendarEntryError("Permission denied")

    monkeypatch.setattr(triage, "create_calendar_entry", fail)
    with pytest.raises(triage.CalendarToolError, match="Check your calendar"):
        asyncio.run(triage.create_events(make_result()))
    assert len(calls) == 1


def test_partial_calendar_failure_reports_created_count(calendar, monkeypatch):
    calls = []

    def fail_second(*args):
        calls.append(args)
        if len(calls) == 2:
            raise triage.CalendarEntryError("Permission denied")
        return {"id": "created-1"}

    monkeypatch.setattr(triage, "create_calendar_entry", fail_second)
    result = make_result(proposed_events=[event_arguments(), event_arguments(title="Other")])
    with pytest.raises(triage.CalendarToolError, match="1 confirmed event"):
        asyncio.run(triage.create_events(result))
    assert len(calls) == 2


def test_unsupported_calendar_is_explicit(calendar):
    with pytest.raises(triage.CalendarToolError, match="only Google"):
        asyncio.run(triage.create_events(make_result(), "hotmail"))
    assert not calendar


def test_main_remains_a_small_orchestrator():
    source = inspect.getsource(triage.main)
    function = ast.parse(source).body[0]
    assert function.end_lineno is not None
    assert function.end_lineno - function.lineno + 1 <= 30


def test_cli_arguments_default_to_google(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["triage.py"])
    _, args = triage.parse_arguments()
    assert args.calendar is triage.Calendar.google
    assert not args.create_events


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_runs_one_provider_request_and_creates_from_its_result(
    provider, enabled, calendar, monkeypatch, tmp_path, capsys,
):
    calls = []
    result = make_result()

    async def parse(**kwargs):
        calls.append(kwargs)
        assert kwargs.get("text_format", kwargs.get("output_format")) is triage.TriageAnalysis
        return SimpleNamespace(
            output_parsed=result, parsed_output=result, stop_reason="end_turn",
        )

    class Client:
        def __init__(self, **kwargs):
            self.base_url = "https://example.com/"
            self.responses = SimpleNamespace(parse=parse)
            self.messages = SimpleNamespace(parse=parse)

        async def close(self):
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr(triage, "AsyncOpenAI" if provider == "openai" else "AsyncAnthropic", Client)
    monkeypatch.setattr(triage, "read_input", lambda _: "message")
    out = tmp_path / "output.json"
    argv = ["triage.py", "--provider", provider, "--json", "--out", str(out)]
    if enabled:
        argv.append("--create-events")
    monkeypatch.setattr(sys, "argv", argv)

    assert triage.main() == 0
    assert len(calls) == 1
    assert len(calendar) == int(enabled)
    printed = json.loads(capsys.readouterr().out)
    assert printed["proposed_events"] == result.model_dump(mode="json")["proposed_events"]
    if enabled:
        assert calendar[0][0] == "google"
        assert printed["calendar_events"][0]["id"] == "event_1"
    else:
        assert "calendar_events" not in printed
    assert json.loads(out.read_text(encoding="utf-8")) == printed


def test_cli_calendar_failure_exits_cleanly(calendar, monkeypatch, capsys):
    async def fake_triage(text):
        return make_result()

    async def fail(result, backend):
        raise triage.CalendarToolError("Google authorization failed")

    monkeypatch.setattr(triage, "triage_openai", fake_triage)
    monkeypatch.setattr(triage, "create_events", fail)
    monkeypatch.setattr(triage, "read_input", lambda _: "message")
    monkeypatch.setattr(sys, "argv", ["triage.py", "--json", "--create-events"])
    assert triage.main() == 1
    captured = capsys.readouterr()
    assert "Google authorization failed" in captured.err
    assert not captured.out


def test_cli_output_failure_warns_about_already_created_events(
    calendar, monkeypatch, tmp_path, capsys,
):
    async def fake_triage(text):
        return make_result()

    def fail_write(*args):
        raise OSError("Disk full")

    monkeypatch.setattr(triage, "triage_openai", fake_triage)
    monkeypatch.setattr(triage, "write_output", fail_write)
    monkeypatch.setattr(triage, "read_input", lambda _: "message")
    monkeypatch.setattr(sys, "argv", [
        "triage.py", "--json", "--create-events", "--out", str(tmp_path / "output.json"),
    ])
    assert triage.main() == 1
    captured = capsys.readouterr()
    assert "Disk full" in captured.err
    assert "1 calendar event(s) were already created" in captured.err
    assert not captured.out
