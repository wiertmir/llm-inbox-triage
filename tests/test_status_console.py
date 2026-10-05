"""Regression tests for live stderr redraw when Windows stdout is piped."""

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import rich.console
from rich.progress import Progress

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402


class TerminalBuffer(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        return 2


@pytest.fixture
def windows_pipe(monkeypatch):
    def activate():
        stdout = io.StringIO()
        stderr = TerminalBuffer()
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)
        monkeypatch.setattr(triage, "IS_WINDOWS", True)
        monkeypatch.setattr(rich.console, "detect_legacy_windows", lambda: True)
        return stdout, stderr

    return activate


def test_stderr_redraw_uses_ansi_when_stdout_is_piped(windows_pipe, monkeypatch):
    stdout, stderr = windows_pipe()
    enable = Mock(return_value=True)
    monkeypatch.setattr(triage, "enable_stderr_vt", enable)

    console = triage.status_console(quiet=False)
    with Progress(console=console, auto_refresh=False) as progress:
        for i in range(15):
            progress.add_task(f"{i:02}.txt", start=False)
        progress.refresh()
        progress.refresh()

    enable.assert_called_once()
    assert console.is_terminal
    assert console.legacy_windows is False
    assert re.search(r"\x1b\[\d+A", stderr.getvalue())
    assert stdout.getvalue() == ""


def test_unsupported_stderr_redraw_warns_and_prints_only_final_rows(windows_pipe, monkeypatch):
    stdout, stderr = windows_pipe()
    monkeypatch.setattr(triage, "enable_stderr_vt", lambda: False)

    console = triage.status_console(quiet=False)
    with Progress(console=console, auto_refresh=False) as progress:
        for i in range(15):
            progress.add_task(f"{i:02}.txt", start=False)
        progress.refresh()
        progress.refresh()
        assert "00.txt" not in stderr.getvalue()

    assert not console.is_terminal
    assert console.legacy_windows is False
    assert "WARNING" in stderr.getvalue()
    assert "showing final status rows only" in stderr.getvalue()
    assert stderr.getvalue().count("00.txt") == 1
    assert "\x1b[" not in stderr.getvalue()
    assert stdout.getvalue() == ""


@pytest.mark.parametrize("mode", ["quiet", "stdout-terminal", "stderr-file", "non-windows"])
def test_other_console_modes_preserve_existing_detection(windows_pipe, monkeypatch, mode):
    windows_pipe()
    def unexpected():
        raise AssertionError("No Windows stderr mode change expected")

    monkeypatch.setattr(triage, "enable_stderr_vt", unexpected)
    if mode == "stdout-terminal":
        monkeypatch.setattr(sys, "stdout", TerminalBuffer())
    elif mode == "stderr-file":
        monkeypatch.setattr(sys, "stderr", io.StringIO())
    elif mode == "non-windows":
        monkeypatch.setattr(triage, "IS_WINDOWS", False)

    console = triage.status_console(quiet=mode == "quiet")

    assert console.quiet == (mode == "quiet")
    assert console.legacy_windows is True


@pytest.mark.skipif(os.name != "nt", reason="Windows console API")
@pytest.mark.parametrize("get_success,mode,set_success,expected", [
    (True, 7, True, True),
    (True, 3, True, True),
    (True, 3, False, False),
    (False, 0, True, False),
])
def test_stderr_vt_uses_selected_handle_and_preserves_mode_bits(
    monkeypatch, get_success, mode, set_success, expected,
):
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = Mock()

    def get_mode(handle, pointer):
        assert handle.value == 123
        ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD)).contents.value = mode
        return get_success

    kernel32.GetConsoleMode.side_effect = get_mode
    kernel32.SetConsoleMode.return_value = set_success
    get_handle = Mock(return_value=123)
    monkeypatch.setattr(msvcrt, "get_osfhandle", get_handle)
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel32))
    monkeypatch.setattr(sys, "stderr", TerminalBuffer())

    assert triage.enable_stderr_vt() is expected
    get_handle.assert_called_once_with(2)
    if get_success and not mode & 4:
        assert kernel32.SetConsoleMode.call_args.args[0].value == 123
        assert kernel32.SetConsoleMode.call_args.args[1] == mode | 4
    else:
        kernel32.SetConsoleMode.assert_not_called()


@pytest.mark.skipif(os.name != "nt", reason="Windows console API")
def test_invalid_stderr_handle_does_not_crash(monkeypatch):
    import msvcrt

    monkeypatch.setattr(msvcrt, "get_osfhandle", Mock(side_effect=OSError("Invalid handle")))
    monkeypatch.setattr(sys, "stderr", TerminalBuffer())
    assert triage.enable_stderr_vt() is False


def test_real_subprocess_redraws_stderr_without_polluting_json_stdout():
    code = r'''
import io
import json
import sys
from rich.progress import Progress
import triage

class TerminalProxy(io.TextIOBase):
    @property
    def encoding(self):
        return self.target.encoding
    def __init__(self, target):
        self.target = target
    def isatty(self):
        return True
    def fileno(self):
        return self.target.fileno()
    def write(self, text):
        return self.target.write(text)
    def flush(self):
        self.target.flush()

sys.stderr = TerminalProxy(sys.stderr)
triage.IS_WINDOWS = True
triage.enable_stderr_vt = lambda: True
console = triage.status_console(quiet=False)
with Progress(console=console, auto_refresh=False) as progress:
    for i in range(15):
        progress.add_task(f"{i:02}.txt", start=False)
    progress.refresh()
    progress.refresh()
print(json.dumps({"legacy_windows": console.legacy_windows, "initial_rows": 15}))
'''
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"legacy_windows": False, "initial_rows": 15}
    assert re.search(r"\x1b\[\d+A", completed.stderr)
    assert "00.txt" in completed.stderr
