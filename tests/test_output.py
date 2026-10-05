"""Tests for issue #4 — MVP: Pretty terminal output + JSON export.

Acceptance criteria from the issue:
  - Use `rich` for a readable table/panel (category, priority, summary, reply).
  - Keep `--json` flag for raw JSON only.
  - Optionally write to an out/ file.

Run with:  pytest -v
"""

import io
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402


def make_result(**overrides) -> triage.TriageResult:
    data: dict[str, Any] = dict(
        id="sample.txt",
        category=triage.Category.invoice,
        priority=3,
        summary="Invoice #4471 for 128,000 JPY is due on August 31, 2026.",
        suggested_reply="Thanks Sarah, we will pay invoice #4471 before August 31.",
        extracted=triage.Extracted(
            dates=[date(2026, 8, 31)],
            amounts=[triage.Money(amount=128000.0, currency="JPY")],
            names=["Sarah Tanaka", "ACME Cloud Billing"],
            deadlines=[date(2026, 8, 31)],
        ),
    )
    data.update(overrides)
    return triage.TriageResult(**data)


# ---------------------------------------------------------------------------
# 1. render() -> the card shows the key information
# ---------------------------------------------------------------------------

def render_to_text(result: triage.TriageResult) -> str:
    # Wide enough that the summary and reply are not wrapped mid-sentence.
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, color_system=None)
    triage.render(result, console=console)
    return buffer.getvalue()


def test_render_shows_category_priority_summary_and_reply():
    out = render_to_text(make_result())

    assert "INVOICE" in out
    assert "3/5" in out
    assert "normal" in out
    assert "Invoice #4471 for 128,000 JPY is due on August 31, 2026." in out
    assert "Suggested reply" in out
    assert "Thanks Sarah, we will pay invoice #4471 before August 31." in out


def test_render_shows_extracted_data():
    out = render_to_text(make_result())

    assert "Deadline" in out
    assert "Mon 31 Aug 2026" in out
    assert "128,000.00 JPY" in out
    assert "Sarah Tanaka" in out
    assert "ACME Cloud Billing" in out


def test_render_without_reply_or_extracted_data():
    out = render_to_text(
        make_result(
            category=triage.Category.spam,
            priority=1,
            summary="Unsolicited crypto offer.",
            suggested_reply=None,
            extracted=triage.Extracted(),
        )
    )

    assert "SPAM" in out
    assert "1/5" in out
    assert "Unsolicited crypto offer." in out
    assert "No reply needed." in out
    assert "Suggested reply" not in out
    assert "Deadline" not in out


def test_render_shows_calendar_proposals_without_claiming_creation():
    result = make_result(
        proposed_events=[triage.CalendarEvent(
            title="Payment deadline",
            start=date(2026, 8, 31),
            end=date(2026, 9, 1),
            description="Pay invoice #4471",
        )],
    )
    out = render_to_text(result)
    assert "Proposed calendar events" in out
    assert "Payment deadline" in out
    assert "2026-08-31" in out
    assert "2026-09-01" in out
    assert "Created" not in out


# ---------------------------------------------------------------------------
# 2. write_output() -> JSON file on disk
# ---------------------------------------------------------------------------

def test_write_output_saves_json_that_round_trips(tmp_path):
    result = make_result()
    path = tmp_path / "result.json"

    triage.write_output(result, path)

    saved = path.read_text(encoding="utf-8")
    assert json.loads(saved)["category"] == "invoice"
    assert triage.TriageResult.model_validate_json(saved) == result


def test_write_output_creates_missing_folders(tmp_path):
    path = tmp_path / "nested" / "deeper" / "result.json"

    triage.write_output(make_result(), path)

    assert path.is_file()


def test_write_output_overwrites_existing_file(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("old content that is not JSON", encoding="utf-8")

    triage.write_output(make_result(priority=5), path)

    assert json.loads(path.read_text(encoding="utf-8"))["priority"] == 5


def test_write_output_keeps_non_ascii_text(tmp_path):
    path = tmp_path / "result.json"
    result = make_result(summary="Faktura na 500 zł — 締切 8月31日")

    triage.write_output(result, path)

    assert triage.TriageResult.model_validate_json(path.read_text(encoding="utf-8")) == result


def test_default_out_path_uses_input_name():
    assert triage.default_out_path("mail/sample.txt") == triage.OUT_DIR / "sample.out.json"


def test_default_out_path_for_stdin_is_timestamped():
    path = triage.default_out_path(None)

    assert path.parent == triage.OUT_DIR
    assert path.name.startswith("stdin_")
    assert path.name.endswith(".out.json")


# ---------------------------------------------------------------------------
# 3. --json / --out through main()
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_openai(monkeypatch):
    result = make_result()

    async def fake_triage(text):
        return result

    monkeypatch.setattr(triage, "triage_openai", fake_triage)
    return result


def run_main(monkeypatch, *args) -> int:
    monkeypatch.setattr(sys, "argv", ["triage.py", *args])
    return triage.main()


def test_main_out_with_path_writes_file_and_prints_json(tmp_path, monkeypatch, capsys, fake_openai):
    out = tmp_path / "res.json"

    assert run_main(monkeypatch, str(ROOT / "sample.txt"), "--json", "-o", str(out)) == 0

    assert triage.TriageResult.model_validate_json(out.read_text(encoding="utf-8")) == fake_openai
    assert triage.TriageResult.model_validate_json(capsys.readouterr().out) == fake_openai


def test_main_out_without_path_uses_default(tmp_path, monkeypatch, capsys, fake_openai):
    monkeypatch.setattr(triage, "OUT_DIR", tmp_path / "out")

    assert run_main(monkeypatch, str(ROOT / "sample.txt"), "--json", "--out") == 0

    saved = tmp_path / "out" / "sample.out.json"
    assert triage.TriageResult.model_validate_json(saved.read_text(encoding="utf-8")) == fake_openai


def test_main_out_requires_json(tmp_path, monkeypatch, fake_openai):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, str(ROOT / "sample.txt"), "-o", str(tmp_path / "res.json"))

    assert not (tmp_path / "res.json").exists()


def test_main_out_rejects_non_json_path(tmp_path, monkeypatch, fake_openai):
    target = tmp_path / "sample.txt"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--json", "--out", str(target))

    assert target.read_text(encoding="utf-8") == "original"
