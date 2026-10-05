"""Tests for issue #2 — MVP: OpenAI provider with Structured Outputs (Pydantic).

Acceptance criteria from the issue:
  - Use the OpenAI Python SDK with the Responses API.
  - The output shape is the Pydantic model `TriageAnalysis`.
  - Call `await client.responses.parse(model=..., input=[system, user], text_format=TriageAnalysis)`.
  - Return `response.output_parsed` (already validated, no `json.loads`).
  - Handle `output_parsed is None` (a safety refusal) gracefully.

The async OpenAI client is replaced with a fake, so no network or API key is needed.
The fake is patched in as both `openai.AsyncOpenAI` and `triage.AsyncOpenAI`, so either
`from openai import AsyncOpenAI` (module or function level) or `openai.AsyncOpenAI()` works.
triage_openai() is a coroutine, so the tests drive it with asyncio.run().

Live test (optional, costs a few tokens):
    OPENAI_LIVE_TEST=1 OPENAI_API_KEY=sk-... pytest -v -k live

Run with:  pytest -v
"""

import asyncio
import inspect
import io
import json
import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402

openai = pytest.importorskip("openai")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def make_result(**overrides) -> triage.TriageAnalysis:
    data: dict[str, Any] = dict(
        category=triage.Category.invoice,
        priority=4,
        summary="Invoice #4471 is overdue and must be paid by Friday.",
        suggested_reply="Thanks, we will pay invoice #4471 by Friday.",
        extracted=triage.Extracted(
            dates=[date(2026, 9, 25)],
            amounts=[triage.Money(amount=1250.0, currency="USD")],
            names=["Sarah Tanaka"],
            deadlines=[date(2026, 9, 25)],
        ),
    )
    data.update(overrides)
    return triage.TriageAnalysis(**data)


def parsed_response(result):
    # output_text is deliberately not JSON: the code must use output_parsed,
    # not re-parse the raw text the old way.
    return SimpleNamespace(
        output_parsed=result,
        output_text="<<not json - use output_parsed>>",
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text="<<not json>>", parsed=result)],
            )
        ],
    )


REFUSAL_TEXT = "I'm sorry, I can't help with that request."


def refusal_response():
    return SimpleNamespace(
        output_parsed=None,
        output_text="",
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="refusal", refusal=REFUSAL_TEXT)],
            )
        ],
    )


def empty_response():
    """output_parsed is None and there is no refusal block (e.g. incomplete output)."""
    return SimpleNamespace(
        output_parsed=None,
        output_text="",
        output=[],
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )


class _Responses:
    def __init__(self, owner):
        self._owner = owner

    async def parse(self, **kwargs):
        self._owner.parse_calls.append(kwargs)
        return self._owner.response

    async def create(self, **kwargs):
        raise AssertionError("Use responses.parse(text_format=...), not responses.create()")


class _Chat:
    @property
    def completions(self):
        raise AssertionError("Use the Responses API, not chat.completions (old pattern)")


class FakeOpenAI:
    """Stand-in for openai.AsyncOpenAI that records calls to responses.parse."""

    instances: list["FakeOpenAI"] = []
    response = None

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.base_url = "https://api.openai.com/v1/"
        self.parse_calls: list[dict] = []
        self.response = type(self).response
        self.responses = _Responses(self)
        self.chat = _Chat()
        type(self).instances.append(self)

    async def close(self):
        pass

    @classmethod
    def all_parse_calls(cls):
        return [call for inst in cls.instances for call in inst.parse_calls]


@pytest.fixture
def fake_openai(monkeypatch):
    """Patch the OpenAI client; set `fake_openai.response` to control the reply."""

    class Fake(FakeOpenAI):
        instances = []
        response = parsed_response(make_result())

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(openai, "AsyncOpenAI", Fake)
    monkeypatch.setattr(triage, "AsyncOpenAI", Fake, raising=False)
    return Fake


def single_parse_call(fake):
    calls = fake.all_parse_calls()
    assert len(calls) == 1, f"expected exactly one responses.parse() call, got {len(calls)}"
    return calls[0]


# ---------------------------------------------------------------------------
# 1. The schema: TriageAnalysis is a Pydantic model usable with Structured Outputs
# ---------------------------------------------------------------------------

def test_triage_result_is_pydantic_model():
    assert issubclass(triage.TriageAnalysis, BaseModel)


def test_schema_has_expected_fields():
    fields = set(triage.TriageAnalysis.model_fields)
    assert fields == {"category", "priority", "summary", "suggested_reply", "extracted", "proposed_events"}
    assert set(triage.Extracted.model_fields) == {"dates", "amounts", "names", "deadlines"}


def test_schema_converts_to_openai_strict_json_schema():
    """responses.parse() converts the model to a strict schema; it must not choke on it."""
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(triage.TriageAnalysis)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


@pytest.mark.parametrize("priority", [0, 6, -1])
def test_priority_out_of_range_rejected(priority):
    with pytest.raises(ValidationError):
        make_result(priority=priority)


@pytest.mark.parametrize("priority", [1, 3, 5])
def test_priority_in_range_accepted(priority):
    assert make_result(priority=priority).priority == priority


def test_hallucinated_category_rejected():
    with pytest.raises(ValidationError):
        triage.TriageAnalysis.model_validate(
            {**make_result().model_dump(), "category": "super-urgent"}
        )


def test_suggested_reply_may_be_null():
    assert make_result(suggested_reply=None).suggested_reply is None


def test_extracted_defaults_to_empty_lists():
    e = triage.Extracted()
    assert (e.dates, e.amounts, e.names, e.deadlines) == ([], [], [], [])


def test_extracted_coerces_json_values():
    """The API returns JSON: ISO date strings and numbers become date / Money."""
    e = triage.Extracted.model_validate(
        {
            "dates": ["2026-08-31"],
            "amounts": [{"amount": 128000, "currency": "JPY"}, {"amount": "1250.50", "currency": "USD"}],
            "deadlines": ["2026-09-25"],
        }
    )

    assert e.dates == [date(2026, 8, 31)]
    assert e.amounts == [
        triage.Money(amount=128000.0, currency="JPY"),
        triage.Money(amount=1250.5, currency="USD"),
    ]
    assert e.deadlines == [date(2026, 9, 25)]


def test_money_currency_defaults_to_default_ccy():
    assert triage.Money(amount=5000).currency == triage.DEFAULT_CCY
    assert triage.Money.model_validate({"amount": 5000}).currency == triage.DEFAULT_CCY


@pytest.mark.parametrize(
    "field, value",
    [
        ("dates", "Friday"),
        ("dates", "August 31, 2026"),
        ("deadlines", "end of month"),
        ("amounts", "128,000 JPY"),
        ("amounts", 1250.0),
        ("amounts", {"amount": "$1,250.00", "currency": "USD"}),
        ("amounts", {"currency": "USD"}),
    ],
)
def test_extracted_rejects_non_iso_dates_and_bad_amounts(field, value):
    with pytest.raises(ValidationError):
        triage.Extracted.model_validate({field: [value]})


def test_schema_declares_date_and_money_types():
    schema = triage.Extracted.model_json_schema()
    props = schema["properties"]

    assert props["dates"]["items"] == {"type": "string", "format": "date"}
    assert props["deadlines"]["items"] == {"type": "string", "format": "date"}
    money = schema["$defs"]["Money"]["properties"]
    assert money["amount"]["type"] == "number"
    assert money["currency"]["type"] == "string"


def test_money_default_is_kept_out_of_strict_schema():
    """Strict mode requires every field, so a `default` in the API schema is meaningless."""
    from openai.lib._pydantic import to_strict_json_schema

    money = to_strict_json_schema(triage.TriageAnalysis)["$defs"]["Money"]

    assert "default" not in money["properties"]["currency"]
    assert set(money["required"]) == {"amount", "currency"}


# ---------------------------------------------------------------------------
# 2. triage_openai() calls responses.parse correctly
# ---------------------------------------------------------------------------

def test_calls_responses_parse_once(fake_openai):
    asyncio.run(triage.triage_openai("hello"))

    single_parse_call(fake_openai)


def test_passes_triage_result_as_text_format(fake_openai):
    asyncio.run(triage.triage_openai("hello"))

    assert single_parse_call(fake_openai)["text_format"] is triage.TriageAnalysis


def test_selected_model_override_reaches_openai(fake_openai):
    provider = triage.select_provider("openai", model="gpt-custom")

    asyncio.run(provider.triage("hello"))

    assert provider.model == "gpt-custom"
    assert single_parse_call(fake_openai)["model"] == "gpt-custom"


def test_passes_a_model_name(fake_openai):
    asyncio.run(triage.triage_openai("hello"))

    model = single_parse_call(fake_openai).get("model")
    assert isinstance(model, str) and model.strip()


def test_input_is_system_then_user(fake_openai):
    message = "Subject: Invoice #4471\n\nPlease pay by Friday."

    asyncio.run(triage.triage_openai(message))

    messages = single_parse_call(fake_openai)["input"]
    assert isinstance(messages, list) and len(messages) == 2
    system, user = messages
    assert system["role"] in ("system", "developer")
    assert system["content"] == triage.SYSTEM_PROMPT
    assert user["role"] == "user"
    assert user["content"] == message


def test_user_text_is_passed_verbatim(fake_openai):
    message = "Zażółć gęślą jaźń — 請求書 — {\"not\": \"a template\"} — <b>"

    asyncio.run(triage.triage_openai(message))

    user = single_parse_call(fake_openai)["input"][-1]
    assert user["content"] == message


def test_system_prompt_does_not_beg_for_json():
    """The schema enforces the shape; the prompt should only describe the task."""
    prompt = triage.SYSTEM_PROMPT.lower()
    assert "strict json" not in prompt
    assert "json and nothing else" not in prompt


def test_system_prompt_asks_to_resolve_relative_dates():
    prompt = triage.SYSTEM_PROMPT
    assert "YYYY-MM-DD" in prompt
    assert "Friday" in prompt
    assert date.today().isoformat() in prompt, "the model needs today's date to resolve 'Friday'"


def test_system_prompt_names_default_currency():
    assert triage.DEFAULT_CCY in triage.SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 3. triage_openai() returns response.output_parsed
# ---------------------------------------------------------------------------

def test_returns_triage_result(fake_openai):
    result = asyncio.run(triage.triage_openai("hello"))

    assert isinstance(result, triage.TriageAnalysis)


def test_returns_output_parsed_unchanged(fake_openai):
    expected = make_result(category=triage.Category.urgent, priority=5, summary="Server down.")
    fake_openai.response = parsed_response(expected)

    result = asyncio.run(triage.triage_openai("prod is down!"))

    assert result == expected


def test_does_not_reparse_output_text(fake_openai):
    """output_text is garbage in the fake; success proves it was not json.loads()'d."""
    result = asyncio.run(triage.triage_openai("hello"))

    assert result.summary == make_result().summary


def test_source_has_no_json_loads():
    src = inspect.getsource(triage.triage_openai)
    assert "json.loads" not in src
    assert "model_validate_json" not in src


# ---------------------------------------------------------------------------
# 4. Refusal: output_parsed is None -> handled gracefully
#
# The issue leaves the mechanism open. These tests require that triage_openai()
# does NOT return None and does NOT crash with an incidental error
# (AttributeError, TypeError, ...). Raising a deliberate exception whose
# message mentions the refusal is the expected behaviour.
# ---------------------------------------------------------------------------

ACCIDENTAL_ERRORS = (AttributeError, TypeError, KeyError, IndexError, NotImplementedError)


@pytest.mark.parametrize("response_factory", [refusal_response, empty_response])
def test_none_output_parsed_does_not_return_none(fake_openai, response_factory):
    fake_openai.response = response_factory()

    try:
        result = asyncio.run(triage.triage_openai("something the model refuses"))
    except ACCIDENTAL_ERRORS as exc:
        pytest.fail(f"Refusal crashed with an incidental {type(exc).__name__}: {exc}")
    except Exception:
        return  # a deliberate, meaningful error is fine
    assert result is not None, "triage_openai() must not silently return None on refusal"
    assert isinstance(result, triage.TriageAnalysis)


def test_refusal_error_is_explicit(fake_openai):
    fake_openai.response = refusal_response()

    with pytest.raises(Exception) as excinfo:
        asyncio.run(triage.triage_openai("something the model refuses"))

    assert not isinstance(excinfo.value, ACCIDENTAL_ERRORS), (
        f"Refusal should raise a deliberate error, got {type(excinfo.value).__name__}"
    )
    msg = str(excinfo.value)
    assert "refus" in msg.lower() or REFUSAL_TEXT in msg, (
        f"Error should say the model refused; got: {msg!r}"
    )


def test_incomplete_output_names_the_reason(fake_openai):
    fake_openai.response = empty_response()

    with pytest.raises(triage.TriageRefusal, match="incomplete.*max_output_tokens"):
        asyncio.run(triage.triage_openai("a very long message"))


def test_client_has_a_finite_timeout(fake_openai):
    """The SDK default (10 minutes per attempt) is far too long for a CLI."""
    asyncio.run(triage.triage_openai("hello"))

    timeout = fake_openai.instances[0].init_kwargs.get("timeout")
    assert isinstance(timeout, (int, float)) and 0 < timeout <= 120


def test_cli_refusal_exits_cleanly(fake_openai, monkeypatch, capsys):
    """`python triage.py` on a refused message -> non-zero exit, message, no JSON, no traceback."""
    fake_openai.response = refusal_response()
    monkeypatch.setattr(sys, "argv", ["triage.py", "--provider", "openai", "--json"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("something the model refuses"))

    try:
        code = triage.main()
    except SystemExit as exc:
        code = exc.code
    except Exception as exc:
        pytest.fail(f"main() let a {type(exc).__name__} escape on refusal: {exc}")

    out, err = capsys.readouterr()
    assert code not in (0, None), "refusal should produce a non-zero exit code"
    assert out.strip() == "", f"nothing should be printed to stdout on refusal, got: {out!r}"
    assert "refus" in err.lower() or REFUSAL_TEXT in err


# ---------------------------------------------------------------------------
# 5. End-to-end through main() with the fake client
# ---------------------------------------------------------------------------

def run_main(monkeypatch, capsys, argv, stdin=""):
    monkeypatch.setattr(sys, "argv", ["triage.py", *argv])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = triage.main()
    return code, capsys.readouterr()


def test_openai_is_default_provider(fake_openai, monkeypatch, capsys):
    code, _ = run_main(monkeypatch, capsys, ["--json"], stdin="hello")

    assert code == 0
    single_parse_call(fake_openai)


def test_main_json_output_matches_schema(fake_openai, monkeypatch, capsys):
    code, captured = run_main(
        monkeypatch, capsys, ["--provider", "openai", "--json", str(ROOT / "sample.txt")]
    )

    assert code == 0
    data = json.loads(captured.out)
    assert triage.TriageAnalysis.model_validate(data) == make_result()


def test_main_sends_file_contents_to_openai(fake_openai, monkeypatch, capsys):
    run_main(monkeypatch, capsys, ["--provider", "openai", "--json", str(ROOT / "sample.txt")])

    user = single_parse_call(fake_openai)["input"][-1]
    assert "Invoice #4471" in user["content"]


# ---------------------------------------------------------------------------
# 6. Optional live test against the real API
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.getenv("OPENAI_LIVE_TEST") and os.getenv("OPENAI_API_KEY")),
    reason="set OPENAI_LIVE_TEST=1 and OPENAI_API_KEY to call the real API",
)
def test_live_openai_on_sample():
    text = (ROOT / "sample.txt").read_text(encoding="utf-8")

    result = asyncio.run(triage.triage_openai(text))

    assert isinstance(result, triage.TriageAnalysis)
    assert result.category == triage.Category.invoice
    assert 1 <= result.priority <= 5
    assert result.summary.strip()
