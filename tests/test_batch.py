"""Offline regression tests for batch processing and application-owned IDs."""

import asyncio
import io
import json
import re
import sys
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_none

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402


def analysis():
    return triage.TriageAnalysis(
        category=triage.Category.question, priority=2, summary="A question.",
        extracted=triage.Extracted(),
    )


@pytest.fixture
def batch(monkeypatch, tmp_path):
    folder = tmp_path / "messages"
    folder.mkdir()
    monkeypatch.setattr(triage, "OUT_DIR", tmp_path / "out")

    async def fake_triage(text):
        return analysis()

    monkeypatch.setattr(triage, "triage_openai", fake_triage)
    monkeypatch.setattr(triage, "triage_anthropic", fake_triage)
    return folder


def run_main(monkeypatch, folder, *options):
    monkeypatch.setattr(sys, "argv", ["triage.py", "--batch", str(folder), *options])
    return triage.main()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_batch_outputs_one_json_and_saves_one_file(batch, monkeypatch, capsys, provider):
    (batch / "b.txt").write_text("second", encoding="utf-8")
    (batch / "a.txt").write_text("first", encoding="utf-8")
    (batch / "c.TXT").write_text("third", encoding="utf-8")
    (batch / "skip.json").write_text("{}", encoding="utf-8")
    (batch / "nested").mkdir()
    (batch / "nested" / "ignored.txt").write_text("nested", encoding="utf-8")

    assert run_main(monkeypatch, batch, "--json", "--provider", provider) == 0

    out, err = capsys.readouterr()
    report = json.loads(out)
    assert set(report) == {"results", "errors"}
    assert report["errors"] == []
    assert [item["id"] for item in report["results"]] == ["a.txt", "b.txt", "c.TXT"]
    for item in report["results"]:
        triage.TriageResult.model_validate(item)
    saved = batch.parent / "out" / "messages.out.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == report
    assert list(saved.parent.glob("*.json")) == [saved]
    assert all(name in err for name in ["a.txt", "b.txt", "c.TXT"])
    assert err.count("Done (question)") == 3
    assert "Batch summary" in err and "question" in err


def test_batch_default_output_is_saved_without_json(batch, monkeypatch, capsys):
    (batch / "a.txt").write_text("first", encoding="utf-8")
    (batch / "b.txt").write_text("second", encoding="utf-8")

    def unexpected_render(*args, **kwargs):
        raise AssertionError("Batch mode must not render individual triage cards")

    monkeypatch.setattr(triage, "render", unexpected_render)

    assert run_main(monkeypatch, batch) == 0

    saved = batch.parent / "out" / "messages.out.json"
    report = json.loads(saved.read_text(encoding="utf-8"))
    assert [item["id"] for item in report["results"]] == ["a.txt", "b.txt"]
    out, err = capsys.readouterr()
    assert out == ""
    assert "Batch summary" in err
    assert re.search(r"question[^\n]*\b2\b", err)
    assert "Inbox Triage" not in err


@pytest.mark.parametrize("json_mode", [False, True])
def test_batch_custom_output_path(batch, monkeypatch, capsys, json_mode):
    (batch / "a.txt").write_text("first", encoding="utf-8")
    target = batch.parent / "custom" / "aggregate.json"
    options = ["--out", str(target)]
    if json_mode:
        options.append("--json")

    assert run_main(monkeypatch, batch, *options) == 0

    assert json.loads(target.read_text(encoding="utf-8"))["errors"] == []
    assert not (batch.parent / "out").exists()


def test_batch_continues_after_input_and_api_failures(batch, monkeypatch, capsys):
    messages = {
        "a.txt": "valid",
        "b.txt": "",
        "c.txt": "refusal",
        "d.txt": "api error",
        "e.txt": "invalid json",
    }
    for name, text in messages.items():
        (batch / name).write_text(text, encoding="utf-8")
    (batch / "f.txt").write_bytes(b"\xff\xfe")
    calls = []

    async def fake_triage(text):
        calls.append(text)
        if text == "refusal":
            raise triage.TriageRefusal("Model refused")
        if text == "api error":
            raise triage.CliError("AI service unavailable")
        if text == "invalid json":
            raise json.JSONDecodeError("Malformed JSON", "", 0)
        return analysis()

    monkeypatch.setattr(triage, "triage_openai", fake_triage)

    assert run_main(monkeypatch, batch, "--json") == 1

    out, err = capsys.readouterr()
    report = json.loads(out)
    assert [item["id"] for item in report["results"]] == ["a.txt"]
    assert [item["id"] for item in report["errors"]] == ["b.txt", "c.txt", "d.txt", "e.txt", "f.txt"]
    assert all(item["error"] for item in report["errors"])
    assert sorted(calls) == sorted(["valid", "refusal", "api error", "invalid json"])
    assert "empty" in report["errors"][0]["error"]
    assert "Model refused" in report["errors"][1]["error"]
    assert "malformed output" in report["errors"][3]["error"]
    assert "Cannot read f.txt" in report["errors"][4]["error"]
    assert err.count("Failed") >= 5
    assert "Traceback" not in err
    saved = batch.parent / "out" / "messages.out.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == report


def test_batch_unreadable_file_does_not_abort_others(batch, monkeypatch, capsys):
    (batch / "a.txt").write_text("first", encoding="utf-8")
    (batch / "b.txt").write_text("second", encoding="utf-8")
    original_read = triage.read_input

    def read(path):
        if Path(path).name == "a.txt":
            raise PermissionError("Access denied")
        return original_read(path)

    monkeypatch.setattr(triage, "read_input", read)

    assert run_main(monkeypatch, batch, "--json") == 1
    report = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in report["results"]] == ["b.txt"]
    assert report["errors"] == [{"id": "a.txt", "error": "Cannot read a.txt: Access denied"}]


def test_all_failed_batch_still_outputs_json(batch, monkeypatch, capsys):
    (batch / "empty.txt").write_text("", encoding="utf-8")

    assert run_main(monkeypatch, batch, "--json") == 1
    report = json.loads(capsys.readouterr().out)
    assert report["results"] == []
    assert report["errors"][0]["id"] == "empty.txt"


@pytest.mark.parametrize("mode", ["missing", "file", "empty"])
def test_invalid_batch_directory_fails_without_provider(batch, monkeypatch, capsys, mode):
    def unexpected(text):
        raise AssertionError("No AI request expected")

    monkeypatch.setattr(triage, "triage_openai", unexpected)
    target = batch
    if mode == "missing":
        target = batch / "missing"
    elif mode == "file":
        target = batch / "input.txt"
        target.write_text("not a directory", encoding="utf-8")

    assert run_main(monkeypatch, target, "--json") == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err and "Traceback" not in err
    assert not (batch.parent / "out").exists()


@pytest.mark.parametrize("jobs", [1, 2, 3])
def test_batch_bounds_concurrency_and_preserves_filename_order(batch, monkeypatch, capsys, jobs):
    for i in range(7):
        (batch / f"{i}.txt").write_text(str(i), encoding="utf-8")
    active = 0
    maximum = 0
    completed = []

    async def fake_triage(text):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01 if text == "0" else 0)
        active -= 1
        completed.append(text)
        return analysis()

    monkeypatch.setattr(triage, "triage_openai", fake_triage)

    assert run_main(monkeypatch, batch, "--json", "--jobs", str(jobs)) == 0
    report = json.loads(capsys.readouterr().out)
    assert maximum == jobs
    assert [item["id"] for item in report["results"]] == [f"{i}.txt" for i in range(7)]
    if jobs > 1:
        assert completed[0] != "0"


def test_request_throttle_spaces_starts_including_retries_and_closes_clients(monkeypatch):
    class TransientError(Exception):
        pass

    clock = [100.0]
    starts = []
    closed = []

    async def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr(triage.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(triage.asyncio, "sleep", sleep)
    monkeypatch.setattr(
        triage, "_retrying",
        lambda _: AsyncRetrying(
            retry=retry_if_exception_type(TransientError),
            stop=stop_after_attempt(2), wait=wait_none(), reraise=True,
        ),
    )

    async def run():
        token = triage._request_throttle.set(triage.RequestThrottle(1.0))
        try:
            async def request(name):
                attempts = 0

                async def call():
                    nonlocal attempts
                    starts.append(clock[0])
                    attempts += 1
                    if name == "retry" and attempts == 1:
                        raise TransientError
                    return name

                async def close():
                    closed.append(name)

                return await triage._with_retry("test", call, close)

            return await asyncio.gather(request("retry"), request("other"), request("third"))
        finally:
            triage._request_throttle.reset(token)

    assert asyncio.run(run()) == ["retry", "other", "third"]
    assert len(starts) == 4
    assert all(b - a >= 1.0 for a, b in zip(starts, starts[1:]))
    assert sorted(closed) == ["other", "retry", "third"]
    assert triage._request_throttle.get() is None


def test_retry_client_closes_on_failure():
    closed = []

    async def call():
        raise triage.TriageRefusal("refused")

    async def close():
        closed.append(True)

    with pytest.raises(triage.TriageRefusal):
        asyncio.run(triage._with_retry("test", call, close))
    assert closed == [True]


@pytest.mark.parametrize("option,value", [
    ("--jobs", "0"), ("--jobs", "-1"), ("--jobs", "1.5"),
    ("--request-interval", "0"), ("--request-interval", "-1"),
    ("--request-interval", "nan"), ("--request-interval", "inf"),
    ("--request-interval", "invalid"),
])
def test_invalid_batch_limits_are_usage_errors(batch, monkeypatch, capsys, option, value):
    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, batch, option, value)
    assert excinfo.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [("--jobs", "2"), ("--request-interval", "0.5")])
def test_batch_limits_require_batch(monkeypatch, capsys, option, value):
    monkeypatch.setattr(sys, "argv", ["triage.py", option, value])
    with pytest.raises(SystemExit) as excinfo:
        triage.parse_arguments()
    assert excinfo.value.code == 2
    assert "require --batch" in capsys.readouterr().err


def test_default_batch_limits(batch, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["triage.py", "--batch", str(batch)])
    _, args = triage.parse_arguments()
    assert args.jobs == 3
    assert args.request_interval == 1.0


@pytest.mark.parametrize("source", [None, "message.txt"])
def test_cli_assigns_required_source_id(batch, monkeypatch, capsys, source):
    argv = ["triage.py", "--json"]
    if source:
        path = batch / source
        path.write_text("message", encoding="utf-8")
        argv.append(str(path))
    else:
        monkeypatch.setattr(sys, "stdin", io.StringIO("message"))
    monkeypatch.setattr(sys, "argv", argv)

    assert triage.main() == 0
    result = triage.TriageResult.model_validate_json(capsys.readouterr().out)
    assert result.id == (source if source else "***stdin***")
    assert triage.TriageResult.model_fields["id"].is_required()
    assert "id" not in triage.TriageAnalysis.model_fields
    with pytest.raises(ValidationError):
        triage.TriageResult.model_validate(analysis().model_dump())


def test_batch_serializes_calendar_creation_and_includes_events(batch, monkeypatch, capsys):
    for name in ["a.txt", "b.txt", "c.txt"]:
        (batch / name).write_text("message", encoding="utf-8")
    active = 0
    maximum = 0
    created_ids = []

    async def create(result, calendar):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        created_ids.append(result.id)
        active -= 1
        return []

    monkeypatch.setattr(triage, "create_events", create)

    assert run_main(monkeypatch, batch, "--json", "--create-events", "--calendar", "me") == 0
    report = json.loads(capsys.readouterr().out)
    assert maximum == 1
    assert sorted(created_ids) == ["a.txt", "b.txt", "c.txt"]
    assert all(item["calendar_events"] == [] for item in report["results"])


def test_batch_calendar_failure_is_a_file_error(batch, monkeypatch, capsys):
    (batch / "a.txt").write_text("message", encoding="utf-8")

    async def fail(result, calendar):
        raise triage.CalendarToolError("Authorization failed; check your calendar before rerunning")

    monkeypatch.setattr(triage, "create_events", fail)

    assert run_main(monkeypatch, batch, "--json", "--create-events") == 1
    report = json.loads(capsys.readouterr().out)
    assert report["results"] == []
    assert "Authorization failed" in report["errors"][0]["error"]


def test_batch_output_failure_keeps_stdout_json_and_fails(batch, monkeypatch, capsys):
    (batch / "a.txt").write_text("message", encoding="utf-8")

    def fail(*args, **kwargs):
        raise PermissionError("Access denied")

    monkeypatch.setattr(triage, "write_output", fail)

    assert run_main(monkeypatch, batch, "--json") == 1
    out, err = capsys.readouterr()
    assert json.loads(out)["results"][0]["id"] == "a.txt"
    assert "Cannot write" in err
    assert "Traceback" not in err


def test_batch_ctrl_c_exits_cleanly(batch, monkeypatch, capsys):
    (batch / "a.txt").write_text("message", encoding="utf-8")

    async def interrupted(text):
        raise KeyboardInterrupt

    monkeypatch.setattr(triage, "triage_openai", interrupted)

    assert run_main(monkeypatch, batch, "--json") == 130
    out, err = capsys.readouterr()
    assert out == ""
    assert "Interrupted" in err
    assert "Traceback" not in err


def test_batch_workers_use_shared_request_throttle(batch, monkeypatch, capsys):
    for name in ["a.txt", "b.txt", "c.txt"]:
        (batch / name).write_text(name, encoding="utf-8")
    clock = [100.0]
    starts = []
    throttles = []

    async def sleep(delay):
        clock[0] += delay

    async def fake_triage(text):
        throttles.append(triage._request_throttle.get())

        async def call():
            starts.append(clock[0])
            return analysis()

        return await triage._with_retry("test", call)

    monkeypatch.setattr(triage.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(triage.asyncio, "sleep", sleep)
    monkeypatch.setattr(triage, "triage_openai", fake_triage)

    assert run_main(monkeypatch, batch, "--json", "--request-interval", "2.5") == 0
    assert len(json.loads(capsys.readouterr().out)["results"]) == 3
    assert starts == [100.0, 102.5, 105.0]
    assert all(throttle is throttles[0] for throttle in throttles)
    assert throttles[0] is not None
    assert triage._request_throttle.get() is None


def test_batch_summary_counts_messages_with_signals_per_category_and_overall():
    def result(id, category, extracted, reply=None):
        return triage.TriageResult(
            id=id, category=category, priority=2, summary="Summary.",
            extracted=extracted, suggested_reply=reply,
        )

    report = triage.BatchReport(
        results=[
            triage.BatchItem(result(
                "a.txt", triage.Category.invoice,
                triage.Extracted(
                    dates=[date(2026, 11, 1), date(2026, 11, 2)],
                    deadlines=[date(2026, 11, 2)],
                    amounts=[
                        triage.Money(amount=100, currency="USD"),
                        triage.Money(amount=200, currency="EUR"),
                    ],
                ),
                "I will arrange payment.",
            )),
            triage.BatchItem(result(
                "b.txt", triage.Category.invoice,
                triage.Extracted(deadlines=[date(2026, 11, 3)]), " \n\t ",
            )),
            triage.BatchItem(result(
                "c.txt", triage.Category.question,
                triage.Extracted(amounts=[triage.Money(amount=0, currency="JPY")]),
                "Here are the instructions.",
            )),
            triage.BatchItem(result("d.txt", triage.Category.ignore, triage.Extracted())),
            triage.BatchItem(result("e.txt", triage.Category.question, triage.Extracted(), "")),
        ],
        errors=[triage.BatchError("failed.txt", "Cannot read")],
    )
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)

    triage.print_batch_summary(report, console)

    rendered = output.getvalue()
    assert all(heading in rendered for heading in ["Count", "Dates", "Amounts", "Reply needed"])
    assert re.search(r"invoice\W+2\W+2\W+1\W+1", rendered)
    assert re.search(r"question\W+2\W+0\W+1\W+1", rendered)
    assert re.search(r"ignore\W+1\W+0\W+0\W+0", rendered)
    assert re.search(r"Total\W+5\W+2\W+2\W+2", rendered)
    assert re.search(r"Failed\W+1\W+-\W+-\W+-", rendered)
    assert "5 successful | 1 failed | 6 files" in rendered
    assert "not individual values" in rendered


def test_all_failed_batch_summary_shows_zero_signals():
    report = triage.BatchReport([], [triage.BatchError("failed.txt", "Cannot read")])
    output = io.StringIO()

    triage.print_batch_summary(report, Console(file=output, width=100, color_system=None))

    rendered = output.getvalue()
    assert re.search(r"Total\W+0\W+0\W+0\W+0", rendered)
    assert "0 successful | 1 failed | 1 file" in rendered


def test_batch_summary_supports_legacy_windows_encoding():
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="cp1252")
    report = triage.BatchReport([], [triage.BatchError("failed.txt", "Cannot read")])

    triage.print_batch_summary(
        report, Console(file=output, width=100, color_system=None, force_terminal=False),
    )
    output.flush()

    rendered = buffer.getvalue().decode("cp1252")
    assert "Batch summary" in rendered
    assert "invoice" in rendered
    assert "Reply needed" in rendered
    assert "0 successful | 1 failed | 1 file" in rendered
