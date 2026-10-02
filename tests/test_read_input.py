"""Tests for issue #1 — MVP: Read input from file or stdin.

Acceptance criteria from the issue:
  - If a path arg is given, read that file.
  - If no path, read from stdin (so `cat mail.txt | python triage.py` works).
  - Handle missing file gracefully with a clear error.

Run with:  pytest -v
"""

import io
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402


class ExplodingStdin(io.StringIO):
    """A stdin that fails the test if anything tries to read it."""

    def read(self, *args, **kwargs):
        raise AssertionError("read_input() read stdin even though a path was given")

    readline = readlines = read

    def __iter__(self):
        raise AssertionError("read_input() read stdin even though a path was given")


# ---------------------------------------------------------------------------
# 1. Path given -> read that file
# ---------------------------------------------------------------------------

def test_reads_file_when_path_given(tmp_path):
    f = tmp_path / "mail.txt"
    f.write_text("Subject: hello\n\nPlease call me back.\n", encoding="utf-8")

    text = triage.read_input(str(f))

    assert "Subject: hello" in text
    assert "Please call me back." in text


def test_reads_whole_multiline_file(tmp_path):
    body = "\n".join(f"line {i}" for i in range(1, 201))
    f = tmp_path / "long.txt"
    f.write_text(body + "\n", encoding="utf-8")

    text = triage.read_input(str(f))

    assert text.strip() == body


def test_reads_file_as_utf8(tmp_path):
    # Non-ASCII text must survive regardless of the system's locale encoding.
    body = "Zażółć gęślą jaźń — 128 000 ¥ — 請求書 — café"
    f = tmp_path / "utf8.txt"
    f.write_bytes(body.encode("utf-8"))

    text = triage.read_input(str(f))

    assert text.strip() == body


def test_reads_bundled_sample(monkeypatch):
    text = triage.read_input(str(ROOT / "sample.txt"))

    assert "Invoice #4471" in text
    assert "Sarah Tanaka" in text


def test_path_given_does_not_touch_stdin(tmp_path, monkeypatch):
    f = tmp_path / "mail.txt"
    f.write_text("from the file", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", ExplodingStdin())

    assert triage.read_input(str(f)).strip() == "from the file"


def test_returns_str(tmp_path):
    f = tmp_path / "mail.txt"
    f.write_text("hi", encoding="utf-8")

    assert isinstance(triage.read_input(str(f)), str)


# ---------------------------------------------------------------------------
# 2. No path -> read stdin
# ---------------------------------------------------------------------------

def test_reads_stdin_when_path_is_none(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("piped message\nsecond line\n"))

    text = triage.read_input(None)

    assert "piped message" in text
    assert "second line" in text


def test_reads_all_of_stdin_not_just_first_line(monkeypatch):
    body = "\n".join(f"line {i}" for i in range(1, 51))
    monkeypatch.setattr(sys, "stdin", io.StringIO(body + "\n"))

    assert triage.read_input(None).strip() == body


def test_reads_non_ascii_from_stdin(monkeypatch):
    body = "Dzień dobry, faktura na 500 zł — 締切 8月31日"
    monkeypatch.setattr(sys, "stdin", io.StringIO(body))

    assert triage.read_input(None).strip() == body


# ---------------------------------------------------------------------------
# 3. Missing file -> clear error
#
# The issue leaves the mechanism open, so these tests accept either:
#   - raising FileNotFoundError / OSError, or
#   - exiting via SystemExit (e.g. sys.exit("error: ...") or parser.error()).
# Either way, the message must name the file the user asked for.
# ---------------------------------------------------------------------------

def test_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.txt"

    with pytest.raises((OSError, SystemExit)) as excinfo:
        triage.read_input(str(missing))

    assert "does_not_exist.txt" in str(excinfo.value)


def test_missing_file_does_not_fall_back_to_stdin(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", ExplodingStdin())

    with pytest.raises((OSError, SystemExit)):
        triage.read_input(str(tmp_path / "nope.txt"))


# ---------------------------------------------------------------------------
# 4. End-to-end through the CLI
# ---------------------------------------------------------------------------

def run_cli(*args, stdin=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "triage.py"), *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        timeout=30,
    )


def test_cli_missing_file_exits_cleanly(tmp_path):
    """`python triage.py missing.txt` -> non-zero exit, readable message, no traceback."""
    missing = tmp_path / "missing_mail.txt"

    proc = run_cli(str(missing))

    assert proc.returncode != 0
    assert "missing_mail.txt" in proc.stderr
    assert "Traceback" not in proc.stderr, (
        "Missing file should produce a friendly error, not a stack trace:\n" + proc.stderr
    )


def _fake_result():
    return triage.TriageResult(
        category=triage.Category.question,
        priority=2,
        summary="stub",
        suggested_reply=None,
        extracted=triage.Extracted(),
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_main_passes_stdin_text_to_provider(monkeypatch, capsys, provider):
    """`cat mail.txt | python triage.py` -> the piped text reaches the provider."""
    seen = {}

    async def fake_triage(text):
        seen["text"] = text
        return _fake_result()

    monkeypatch.setattr(triage, f"triage_{provider}", fake_triage)
    monkeypatch.setattr(sys, "argv", ["triage.py", "--provider", provider, "--json"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello from a pipe"))

    assert triage.main() == 0
    assert seen["text"].strip() == "hello from a pipe"


def _no_provider_call(text):
    raise AssertionError("the provider must not be called")


@pytest.mark.parametrize("stdin", ["", "   \n\t\n"])
def test_cli_empty_input_is_not_sent(monkeypatch, capsys, stdin):
    """An empty message would only waste an API call; stop with an error instead."""
    monkeypatch.setattr(triage, "triage_openai", _no_provider_call)
    monkeypatch.setattr(sys, "argv", ["triage.py", "--json"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))

    code = triage.main()

    out, err = capsys.readouterr()
    assert code not in (0, None)
    assert out == ""
    assert "empty" in err.lower()


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(triage, "triage_openai", _no_provider_call)
    monkeypatch.setattr(sys, "argv", ["triage.py", *argv, "--json"])
    try:
        return triage.main()
    except SystemExit as exc:
        return exc.code


def test_cli_directory_as_input_exits_cleanly(monkeypatch, capsys, tmp_path):
    code = _run_cli(monkeypatch, [str(tmp_path)])

    err = capsys.readouterr().err
    assert code not in (0, None)
    assert "Cannot read" in err and "Traceback" not in err


def test_cli_non_utf8_input_exits_cleanly(monkeypatch, capsys, tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes("Zażółć gęślą jaźń".encode("cp1250"))

    code = _run_cli(monkeypatch, [str(path)])

    err = capsys.readouterr().err
    assert code not in (0, None)
    assert "UTF-8" in err and "Traceback" not in err


def test_cli_ctrl_c_exits_cleanly(monkeypatch, capsys):
    async def interrupted(text):
        raise KeyboardInterrupt

    monkeypatch.setattr(triage, "triage_openai", interrupted)
    # Without --json, so the warning reaches stderr (with --json only errors do).
    monkeypatch.setattr(sys, "argv", ["triage.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello"))

    assert triage.main() == 130
    assert "Interrupted" in capsys.readouterr().err


def test_main_passes_file_text_to_provider(monkeypatch, capsys):
    seen = {}

    async def fake_triage(text):
        seen["text"] = text
        return _fake_result()

    monkeypatch.setattr(triage, "triage_openai", fake_triage)
    monkeypatch.setattr(sys, "argv", ["triage.py", str(ROOT / "sample.txt"), "--json"])
    monkeypatch.setattr(sys, "stdin", ExplodingStdin())

    assert triage.main() == 0
    assert "Invoice #4471" in seen["text"]
