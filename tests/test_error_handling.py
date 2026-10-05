"""Tests for issue #5 — MVP: Error handling + retry on transient API errors.

Acceptance criteria from the issue:
  - Catch transient errors (rate limit, timeout) and retry with backoff.
  - Fail with a clear message on bad/missing API key.
  - Never crash on malformed model output - handle JSON parse errors.

Written test-first: they describe the behaviour, not one particular design.
Everything goes through main(), so the retry can live in triage_openai /
triage_anthropic, in a shared helper, or around the `triage(text)` call in main().
The tests only pin down what the user sees:

  - transient error, then success   -> exit 0 and the normal JSON on stdout
  - transient error every time      -> gives up after a few tries, exit != 0
  - between attempts                -> time.sleep() with a growing delay
  - permanent error (401, 400, ...) -> no retry, exit != 0
  - bad / missing key               -> exit != 0, stderr names the env var
  - malformed model output          -> exit != 0, short message, no traceback

The provider clients are async fakes whose request coroutine raises the real SDK
exception classes (openai.RateLimitError, anthropic.APITimeoutError, ...), so
`except openai.RateLimitError` and `isinstance(exc, openai.APIStatusError)`
checks work as they would against the real API. time.sleep and asyncio.sleep
are replaced, so backoff costs no wall-clock time (tenacity's AsyncRetrying uses asyncio.sleep).

Note: the SDK clients already retry twice on their own (max_retries=2). The fakes
bypass that, so these tests only see your retry loop. With a real client you
probably want AsyncOpenAI(max_retries=0) / AsyncAnthropic(max_retries=0), otherwise each
of your attempts can turn into three HTTP requests.

Run with:  pytest -v tests/test_error_handling.py
"""

import asyncio
import asyncio
import io
import json
import sys
import threading
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

if TYPE_CHECKING:  # one name for the type checker; the installed SDKs use httpx2
    import httpx2 as httpx
else:
    try:  # the SDKs build their responses on httpx2, older releases on httpx
        import httpx2 as httpx
    except ImportError:
        import httpx

# A sensible retry policy fits inside these limits; loosen them if you choose otherwise.
MAX_ATTEMPTS = 10
MAX_TOTAL_BACKOFF_SECONDS = 120

PROVIDERS = {
    "openai": SimpleNamespace(
        module=openai,
        key_env="OPENAI_API_KEY",
        url="https://api.openai.com/v1/responses",
    ),
    "anthropic": SimpleNamespace(
        module=anthropic,
        key_env="ANTHROPIC_API_KEY",
        url="https://api.anthropic.com/v1/messages",
    ),
}


# ---------------------------------------------------------------------------
# Errors, built from the real SDK exception classes
# ---------------------------------------------------------------------------

def _request(provider):
    return httpx.Request("POST", PROVIDERS[provider].url)


def status_error(provider, cls_name, status, message, headers=None):
    module = PROVIDERS[provider].module
    response = httpx.Response(
        status, request=_request(provider), headers=headers, json={"error": {"message": message}}
    )
    return getattr(module, cls_name)(message, response=response, body={"error": {"message": message}})


def rate_limit(provider):
    return status_error(provider, "RateLimitError", 429, "Rate limit reached, please try again later")


def timeout(provider):
    return PROVIDERS[provider].module.APITimeoutError(request=_request(provider))


def connection_error(provider):
    return PROVIDERS[provider].module.APIConnectionError(request=_request(provider))


def server_error(provider):
    return status_error(provider, "InternalServerError", 500, "The server had an error")


def overloaded(provider):
    # Anthropic answers 529 "overloaded" under load; OpenAI uses 503 for the same thing.
    if provider == "anthropic":
        return status_error(provider, "OverloadedError", 529, "Overloaded")
    return status_error(provider, "InternalServerError", 503, "The engine is currently overloaded")


def bad_key(provider):
    return status_error(provider, "AuthenticationError", 401, "Incorrect API key provided")


def permission_denied(provider):
    return status_error(provider, "PermissionDeniedError", 403, "Your key cannot use this model")


def bad_request(provider):
    return status_error(provider, "BadRequestError", 400, "Invalid request")


def unknown_model(provider):
    return status_error(provider, "NotFoundError", 404, "The model does not exist")


TRANSIENT = [
    pytest.param(rate_limit, id="rate-limit-429"),
    pytest.param(timeout, id="timeout"),
    pytest.param(connection_error, id="connection-error"),
    pytest.param(server_error, id="server-error-500"),
    pytest.param(overloaded, id="overloaded"),
]

PERMANENT = [
    pytest.param(bad_key, id="bad-key-401"),
    pytest.param(permission_denied, id="permission-denied-403"),
    pytest.param(bad_request, id="bad-request-400"),
    pytest.param(unknown_model, id="unknown-model-404"),
]


def validation_error(provider):
    """What responses.parse / messages.parse raise when the JSON does not fit TriageResult."""
    try:
        triage.TriageResult.model_validate({"category": "super-urgent", "priority": 9})
    except Exception as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def json_decode_error(provider):
    """Truncated or non-JSON text from the model."""
    try:
        json.loads('{"category": "invoice", "priority": 3, "summ')
    except json.JSONDecodeError as exc:
        return exc
    raise AssertionError("expected a JSONDecodeError")


MALFORMED = [
    pytest.param(validation_error, id="schema-mismatch"),
    pytest.param(json_decode_error, id="invalid-json"),
]


# ---------------------------------------------------------------------------
# Fake clients
# ---------------------------------------------------------------------------

def make_result() -> triage.TriageResult:
    return triage.TriageResult(
        id="***stdin***",
        category=triage.Category.invoice,
        priority=3,
        summary="Invoice #4471 for 128,000 JPY is due on August 31, 2026.",
        suggested_reply="Thanks, we will pay invoice #4471 before August 31.",
        extracted=triage.Extracted(
            dates=[date(2026, 8, 31)],
            amounts=[triage.Money(amount=128000.0, currency="JPY")],
            names=["Sarah Tanaka"],
            deadlines=[date(2026, 8, 31)],
        ),
    )


class Script:
    """The outcome of each API call in order: an exception to raise, or OK.

    The last entry repeats, so [rate_limit] means "rate limited forever".
    """

    OK = object()

    def __init__(self):
        self.outcomes = [self.OK]
        self.calls = 0

    def next(self):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return make_result()


def _openai_reply(result):
    return SimpleNamespace(
        output_parsed=result,
        output_text=result.model_dump_json(),
        output=[SimpleNamespace(type="message", content=[])],
    )


def _anthropic_reply(result):
    block = SimpleNamespace(type="text", text=result.model_dump_json(), parsed_output=result)
    return SimpleNamespace(
        content=[block],
        stop_reason="end_turn",
        parsed_output=result,
        usage=SimpleNamespace(input_tokens=10, output_tokens=10),
    )


def fake_openai_class(script):
    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.base_url = "https://api.openai.com/v1/"
            self.responses = SimpleNamespace(parse=self._call, create=self._call)

        async def _call(self, **kwargs):
            return _openai_reply(script.next())

        async def close(self):
            pass

        def with_options(self, **kwargs):
            return self

    return FakeOpenAI


def fake_anthropic_class(script):
    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.base_url = "https://api.anthropic.com"
            self.messages = SimpleNamespace(parse=self._call, create=self._call)

        async def _call(self, **kwargs):
            return _anthropic_reply(script.next())

        async def close(self):
            pass

        def with_options(self, **kwargs):
            return self

    return FakeAnthropic


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this file may reach the real API, even by mistake."""

    def refuse(*args, **kwargs):
        raise RuntimeError("test tried to send a real HTTP request")

    monkeypatch.setattr(httpx.Client, "send", refuse)


@pytest.fixture
def sleeps(monkeypatch):
    """Replace time.sleep and asyncio.sleep: record the backoff delays instead of waiting.

    tenacity's AsyncRetrying waits with asyncio.sleep, the sync Retrying with time.sleep.
    """
    delays: list[float] = []
    real_sleep = time.sleep

    async def fake_async_sleep(seconds, result=None):
        delays.append(seconds)
        return result

    def fake_sleep(seconds):
        if threading.current_thread() is threading.main_thread():
            delays.append(seconds)
        else:  # e.g. a spinner thread
            real_sleep(min(seconds, 0.01))

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(asyncio, "sleep", fake_async_sleep)
    return delays


@pytest.fixture(params=list(PROVIDERS))
def provider(request):
    return request.param


@pytest.fixture
def api(provider, monkeypatch):
    """Patch the provider's client; set api.outcomes to script the replies."""
    script = Script()
    info = PROVIDERS[provider]
    fake = fake_openai_class(script) if provider == "openai" else fake_anthropic_class(script)
    cls_name = "AsyncOpenAI" if provider == "openai" else "AsyncAnthropic"

    monkeypatch.setenv(info.key_env, "test-key-not-a-real-key")
    monkeypatch.setattr(info.module, cls_name, fake)
    monkeypatch.setattr(triage, cls_name, fake, raising=False)
    return script


def run_main(monkeypatch, capsys, provider, stdin="Please pay invoice #4471 by Friday."):
    monkeypatch.setattr(sys, "argv", ["triage.py", "--provider", provider, "--json"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    try:
        code = triage.main()
    except SystemExit as exc:
        code = exc.code
    except Exception as exc:
        pytest.fail(f"main() crashed with {type(exc).__name__}: {exc}")
    out, err = capsys.readouterr()
    return code, out, err


def assert_clean_failure(code, out, err):
    assert code not in (0, None), "the run failed, so the exit code must be non-zero"
    assert out.strip() == "", f"no JSON should reach stdout when the run failed, got: {out!r}"
    assert err.strip(), "stderr should say what went wrong"
    assert "Traceback" not in err, f"show a short message, not a traceback:\n{err}"


# ---------------------------------------------------------------------------
# 1. Transient errors are retried with backoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", TRANSIENT)
def test_transient_error_then_success_is_retried(api, provider, sleeps, monkeypatch, capsys, error):
    api.outcomes = [error(provider), Script.OK]

    code, out, err = run_main(monkeypatch, capsys, provider)

    assert api.calls == 2, f"expected one failed call and one retry, got {api.calls} calls"
    assert code == 0, f"the retry succeeded, so the run should too; stderr:\n{err}"
    assert triage.TriageResult.model_validate(json.loads(out)) == make_result()


def test_recovers_after_several_transient_errors(api, provider, sleeps, monkeypatch, capsys):
    api.outcomes = [rate_limit(provider), timeout(provider), Script.OK]

    code, out, _ = run_main(monkeypatch, capsys, provider)

    assert api.calls == 3
    assert code == 0
    assert json.loads(out)["category"] == "invoice"


def test_waits_before_each_retry(api, provider, sleeps, monkeypatch, capsys):
    api.outcomes = [rate_limit(provider), rate_limit(provider), Script.OK]

    run_main(monkeypatch, capsys, provider)

    assert len(sleeps) >= 2, f"expected a sleep before each of the 2 retries, got {sleeps}"
    assert all(delay > 0 for delay in sleeps)


def test_backoff_grows_between_attempts(api, provider, sleeps, monkeypatch, capsys):
    """Exponential backoff: later waits are longer than the first one (jitter on top is fine)."""
    api.outcomes = [rate_limit(provider)] * 3 + [Script.OK]

    run_main(monkeypatch, capsys, provider)

    assert len(sleeps) >= 3, f"expected 3 backoff sleeps, got {sleeps}"
    assert sleeps[-1] > sleeps[0], f"backoff should grow, got delays {sleeps}"


def test_no_sleep_when_first_call_succeeds(api, provider, sleeps, monkeypatch, capsys):
    code, _, _ = run_main(monkeypatch, capsys, provider)

    assert code == 0
    assert api.calls == 1
    assert sleeps == []


@pytest.mark.parametrize("error", TRANSIENT)
def test_gives_up_when_error_persists(api, provider, sleeps, monkeypatch, capsys, error):
    api.outcomes = [error(provider)]  # fails forever

    code, out, err = run_main(monkeypatch, capsys, provider)

    assert 1 < api.calls <= MAX_ATTEMPTS, f"expected a few retries then giving up, got {api.calls} calls"
    assert sum(sleeps) <= MAX_TOTAL_BACKOFF_SECONDS, f"total backoff too long: {sum(sleeps):.0f}s"
    assert_clean_failure(code, out, err)


def test_retry_after_header_is_honoured(api, provider, sleeps, monkeypatch, capsys):
    """The server knows best how long to back off: wait at least that long."""
    api.outcomes = [
        status_error(provider, "RateLimitError", 429, "Slow down", headers={"retry-after": "7"}),
        Script.OK,
    ]

    code, _, _ = run_main(monkeypatch, capsys, provider)

    assert code == 0
    assert sleeps and sleeps[0] >= 7, f"Retry-After: 7 should mean a wait of 7s+, got {sleeps}"


def test_retry_after_ms_header_is_honoured(api, provider, sleeps, monkeypatch, capsys):
    api.outcomes = [
        status_error(provider, "RateLimitError", 429, "Slow down", headers={"retry-after-ms": "4500"}),
        Script.OK,
    ]

    run_main(monkeypatch, capsys, provider)

    assert sleeps and sleeps[0] >= 4.5, f"retry-after-ms: 4500 should mean 4.5s+, got {sleeps}"


def test_huge_retry_after_is_capped(api, provider, sleeps, monkeypatch, capsys):
    api.outcomes = [
        status_error(provider, "RateLimitError", 429, "Slow down", headers={"retry-after": "3600"}),
        Script.OK,
    ]

    run_main(monkeypatch, capsys, provider)

    assert sleeps and sleeps[0] <= 60, f"an hour-long Retry-After should be capped, got {sleeps}"


def test_retry_warning_is_logged_with_wait_time(api, provider, sleeps, monkeypatch, capsys):
    """Without --json, warnings reach stderr: the user sees why it is taking longer."""
    api.outcomes = [rate_limit(provider), Script.OK]
    monkeypatch.setattr(sys, "argv", ["triage.py", "--provider", provider])
    monkeypatch.setattr(sys, "stdin", io.StringIO("Please pay invoice #4471 by Friday."))

    assert triage.main() == 0

    err = capsys.readouterr().err
    assert "WARNING" in err and "retrying in" in err, f"expected a retry warning; stderr:\n{err}"


# ---------------------------------------------------------------------------
# 2. Permanent errors fail fast: retrying a bad key or a bad request won't help
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", PERMANENT)
def test_permanent_error_is_not_retried(api, provider, sleeps, monkeypatch, capsys, error):
    api.outcomes = [error(provider), Script.OK]

    code, out, err = run_main(monkeypatch, capsys, provider)

    assert api.calls == 1, f"{type(api.outcomes[0]).__name__} must not be retried, got {api.calls} calls"
    assert sleeps == []
    assert_clean_failure(code, out, err)


# ---------------------------------------------------------------------------
# 3. Bad or missing API key -> a clear message that names the env var
# ---------------------------------------------------------------------------

def test_bad_key_message_names_the_env_var(api, provider, sleeps, monkeypatch, capsys):
    api.outcomes = [bad_key(provider)]

    code, out, err = run_main(monkeypatch, capsys, provider)

    assert_clean_failure(code, out, err)
    assert PROVIDERS[provider].key_env in err, f"tell the user which key to fix; stderr:\n{err}"


def test_missing_key_message_names_the_env_var(provider, sleeps, monkeypatch, capsys, tmp_path):
    """Uses the real SDK client: AsyncOpenAI() raises OpenAIError when the key is missing,
    AsyncAnthropic raises TypeError on the first request. Neither touches the network."""
    info = PROVIDERS[provider]
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    # Once load_dotenv() is wired in, it must not pull a real key from .env here.
    try:
        import dotenv
    except ImportError:
        pass
    else:
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(triage, "load_dotenv", lambda *a, **k: False, raising=False)
    monkeypatch.chdir(tmp_path)

    code, out, err = run_main(monkeypatch, capsys, provider)

    assert_clean_failure(code, out, err)
    assert info.key_env in err, f"tell the user which env var to set; stderr:\n{err}"
    assert sleeps == [], "a missing key will not appear on retry, so don't back off"


# ---------------------------------------------------------------------------
# 4. Malformed model output never crashes the program
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", MALFORMED)
def test_malformed_output_fails_cleanly(api, provider, sleeps, monkeypatch, capsys, error):
    """Retrying malformed output once or twice is allowed, but it must end cleanly."""
    api.outcomes = [error(provider)]  # malformed every time

    code, out, err = run_main(monkeypatch, capsys, provider)

    assert api.calls <= MAX_ATTEMPTS
    assert_clean_failure(code, out, err)
