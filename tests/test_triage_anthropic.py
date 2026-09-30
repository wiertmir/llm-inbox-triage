"""Tests for issue #3 — MVP: Anthropic provider (multi-provider support).

Acceptance criteria from the issue:
  - Use the anthropic SDK with `system=SYSTEM_PROMPT`.
  - Turn Claude's reply into a validated `TriageResult`.
  - `--provider anthropic` gives results equivalent to openai.

The issue says "parse the returned text as JSON", while triage.py suggests a forced
tool whose input_schema is TriageResult's JSON schema. The fake client answers
whichever way it is called:
  - messages.create(tools=[...])         -> a tool_use block with the result as input
  - messages.create() without tools      -> a text block containing the JSON
  - messages.parse(output_format=...)    -> a message with parsed_output
Tool-specific tests are skipped when the forced-tool approach is not used.

Live test (optional, costs a few tokens):
    ANTHROPIC_LIVE_TEST=1 ANTHROPIC_API_KEY=sk-ant-... pytest -v -k live

Run with:  pytest -v
"""

import copy
import io
import json
import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402

anthropic = pytest.importorskip("anthropic")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def make_result(**overrides) -> triage.TriageResult:
    data: dict[str, Any] = dict(
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


def make_payload(**overrides) -> dict:
    """What Claude sends back: plain JSON types (enum as a string)."""
    return {**make_result().model_dump(mode="json"), **overrides}


REFUSAL_TEXT = "I'm sorry, I can't help with that request."


def message(content, stop_reason, parsed=None):
    return SimpleNamespace(
        id="msg_fake",
        type="message",
        role="assistant",
        model="claude-fake",
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=10),
        parsed_output=parsed,
    )


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(("create", kwargs))
        return self._owner.reply("create", kwargs)

    def parse(self, **kwargs):
        self._owner.calls.append(("parse", kwargs))
        return self._owner.reply("parse", kwargs)


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic that records calls to messages.*."""

    instances: list["FakeAnthropic"] = []
    scenario = "ok"  # "ok" | "refusal" | "empty"
    payload: dict = {}
    preamble = False  # tool mode: emit a text block before the tool_use block

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.base_url = "https://api.anthropic.com"
        self.calls: list[tuple[str, dict]] = []
        self.messages = _Messages(self)
        type(self).instances.append(self)

    @classmethod
    def all_calls(cls):
        return [call for inst in cls.instances for call in inst.calls]

    def reply(self, method, kwargs):
        cls = type(self)
        if cls.scenario == "refusal":
            return message([], "refusal")
        if cls.scenario == "empty":
            return message([], "max_tokens")

        if method == "parse":
            # The real SDK validates too, so invalid output raises ValidationError here.
            parsed = triage.TriageResult.model_validate(cls.payload)
            block = SimpleNamespace(type="text", text=json.dumps(cls.payload), parsed_output=parsed)
            return message([block], "end_turn", parsed=parsed)

        if kwargs.get("tools"):
            tool = list(kwargs["tools"])[0]
            blocks = [
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_fake",
                    name=tool["name"],
                    input=copy.deepcopy(cls.payload),
                )
            ]
            if cls.preamble:
                blocks.insert(0, SimpleNamespace(type="text", text="Let me triage this message."))
            return message(blocks, "tool_use")

        return message([SimpleNamespace(type="text", text=json.dumps(cls.payload))], "end_turn")


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Patch the Anthropic client; set `scenario` / `payload` / `preamble` to control the reply."""

    class Fake(FakeAnthropic):
        instances = []
        scenario = "ok"
        payload = make_payload()
        preamble = False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.setattr(anthropic, "Anthropic", Fake)
    monkeypatch.setattr(triage, "Anthropic", Fake, raising=False)
    return Fake


def single_call(fake):
    calls = fake.all_calls()
    assert len(calls) == 1, f"expected exactly one Anthropic messages call, got {len(calls)}"
    return calls[0]


def as_text(value) -> str:
    """`system` and message `content` may be a string or a list of text blocks."""
    if isinstance(value, str):
        return value
    return "".join(block["text"] for block in value)


def forced_tool_kwargs(fake) -> dict:
    method, kwargs = single_call(fake)
    if method != "create" or not kwargs.get("tools"):
        pytest.skip("implementation does not use the forced-tool approach")
    return kwargs


ACCIDENTAL_ERRORS = (AttributeError, TypeError, KeyError, IndexError, NotImplementedError)


# ---------------------------------------------------------------------------
# 1. The request sent to Claude
# ---------------------------------------------------------------------------

def test_calls_anthropic_once(fake_anthropic):
    triage.triage_anthropic("hello")

    single_call(fake_anthropic)


def test_system_prompt_is_passed_as_system_param(fake_anthropic):
    triage.triage_anthropic("hello")

    _, kwargs = single_call(fake_anthropic)
    assert "system" in kwargs, "Anthropic takes the system prompt as `system=`, not as a message"
    assert as_text(kwargs["system"]) == triage.SYSTEM_PROMPT


def test_messages_is_a_single_user_turn(fake_anthropic):
    message_text = "Subject: Invoice #4471\n\nPlease pay by Friday."

    triage.triage_anthropic(message_text)

    _, kwargs = single_call(fake_anthropic)
    messages = list(kwargs["messages"])
    assert len(messages) == 1, "Anthropic rejects a 'system' role inside messages"
    assert messages[0]["role"] == "user"
    assert as_text(messages[0]["content"]) == message_text


def test_user_text_is_passed_verbatim(fake_anthropic):
    message_text = "Zażółć gęślą jaźń — 請求書 — {\"not\": \"a template\"} — <b>"

    triage.triage_anthropic(message_text)

    _, kwargs = single_call(fake_anthropic)
    assert as_text(list(kwargs["messages"])[-1]["content"]) == message_text


def test_passes_a_claude_model(fake_anthropic):
    triage.triage_anthropic("hello")

    _, kwargs = single_call(fake_anthropic)
    model = kwargs.get("model")
    assert isinstance(model, str) and "claude" in model


def test_passes_max_tokens(fake_anthropic):
    """max_tokens is required by the Messages API."""
    triage.triage_anthropic("hello")

    _, kwargs = single_call(fake_anthropic)
    max_tokens = kwargs.get("max_tokens")
    assert isinstance(max_tokens, int) and max_tokens > 0


# ---------------------------------------------------------------------------
# 2. Forced tool approach (as suggested in triage.py)
# ---------------------------------------------------------------------------

def test_single_tool_built_from_triage_result_schema(fake_anthropic):
    triage.triage_anthropic("hello")

    tools = list(forced_tool_kwargs(fake_anthropic)["tools"])
    assert len(tools) == 1
    schema = tools[0]["input_schema"]
    assert set(schema["properties"]) == set(triage.TriageResult.model_fields)
    assert {"category", "priority", "summary", "extracted"} <= set(schema.get("required", []))


def test_tool_choice_forces_the_tool(fake_anthropic):
    triage.triage_anthropic("hello")

    kwargs = forced_tool_kwargs(fake_anthropic)
    tool_choice = kwargs.get("tool_choice")
    assert tool_choice, "without tool_choice Claude may answer in prose instead of calling the tool"
    assert tool_choice["type"] in ("tool", "any")
    if tool_choice["type"] == "tool":
        assert tool_choice["name"] == list(kwargs["tools"])[0]["name"]


def test_text_before_tool_use_is_ignored(fake_anthropic):
    fake_anthropic.preamble = True

    result = triage.triage_anthropic("hello")

    forced_tool_kwargs(fake_anthropic)
    assert result == make_result()


# ---------------------------------------------------------------------------
# 3. The result: a validated TriageResult, equivalent to OpenAI's
# ---------------------------------------------------------------------------

def test_returns_triage_result(fake_anthropic):
    result = triage.triage_anthropic("hello")

    assert isinstance(result, triage.TriageResult)
    assert isinstance(result.category, triage.Category)
    assert isinstance(result.extracted, triage.Extracted)


def test_returns_what_claude_sent(fake_anthropic):
    expected = make_result(category=triage.Category.urgent, priority=5, summary="Server down.")
    fake_anthropic.payload = expected.model_dump(mode="json")

    result = triage.triage_anthropic("prod is down!")

    assert result == expected


def test_null_suggested_reply_is_kept(fake_anthropic):
    fake_anthropic.payload = make_payload(suggested_reply=None)

    assert triage.triage_anthropic("newsletter").suggested_reply is None


def test_extracted_json_values_become_date_and_money(fake_anthropic):
    result = triage.triage_anthropic("hello")

    assert fake_anthropic.payload["extracted"]["dates"] == ["2026-08-31"]
    assert result.extracted.dates == [date(2026, 8, 31)]
    assert result.extracted.deadlines == [date(2026, 8, 31)]
    assert result.extracted.amounts == [triage.Money(amount=128000.0, currency="JPY")]


def test_amount_without_currency_gets_default_ccy(fake_anthropic):
    fake_anthropic.payload = make_payload(
        extracted={**make_payload()["extracted"], "amounts": [{"amount": 5000}]}
    )

    result = triage.triage_anthropic("hello")

    assert result.extracted.amounts == [triage.Money(amount=5000.0, currency=triage.DEFAULT_CCY)]


def _without_summary():
    payload = make_payload()
    del payload["summary"]
    return payload


def _extracted(**overrides):
    return lambda: make_payload(extracted={**make_payload()["extracted"], **overrides})


@pytest.mark.parametrize(
    "bad_payload",
    [
        pytest.param(lambda: make_payload(priority=9), id="priority-out-of-range"),
        pytest.param(lambda: make_payload(category="super-urgent"), id="unknown-category"),
        pytest.param(_without_summary, id="missing-summary"),
        pytest.param(_extracted(dates=["August 31, 2026"]), id="non-iso-date"),
        pytest.param(_extracted(amounts=["128,000 JPY"]), id="amount-as-string"),
        pytest.param(_extracted(amounts=[{"currency": "JPY"}]), id="amount-missing"),
    ],
)
def test_invalid_output_is_rejected(fake_anthropic, bad_payload):
    """Claude's output must go through Pydantic validation, not be trusted as-is."""
    fake_anthropic.payload = bad_payload()

    with pytest.raises(Exception) as excinfo:
        triage.triage_anthropic("hello")

    assert not isinstance(excinfo.value, ACCIDENTAL_ERRORS), (
        f"Invalid output should fail validation, got {type(excinfo.value).__name__}"
    )


def test_equivalent_to_openai_for_the_same_output(fake_anthropic, monkeypatch):
    payload = fake_anthropic.payload

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.base_url = "https://api.openai.com/v1/"
            self.responses = SimpleNamespace(
                parse=lambda **_: SimpleNamespace(
                    output_parsed=triage.TriageResult.model_validate(payload), output=[]
                )
            )

    openai = pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(triage, "OpenAI", FakeOpenAI, raising=False)

    assert triage.triage_anthropic("hello") == triage.triage_openai("hello")


# ---------------------------------------------------------------------------
# 4. Refusal / no usable output -> handled like OpenAI (TriageRefusal)
# ---------------------------------------------------------------------------

def test_refusal_raises_triage_refusal(fake_anthropic):
    fake_anthropic.scenario = "refusal"

    with pytest.raises(triage.TriageRefusal) as excinfo:
        triage.triage_anthropic("something the model refuses")

    assert "refus" in str(excinfo.value).lower()


def test_empty_reply_is_an_explicit_error(fake_anthropic):
    """No tool_use / text block (e.g. stop_reason="max_tokens") must not crash by accident."""
    fake_anthropic.scenario = "empty"

    try:
        result = triage.triage_anthropic("hello")
    except ACCIDENTAL_ERRORS as exc:
        pytest.fail(f"Empty reply crashed with an incidental {type(exc).__name__}: {exc}")
    except Exception:
        return
    pytest.fail(f"Empty reply should raise, got {result!r}")


# ---------------------------------------------------------------------------
# 5. End-to-end through main() with --provider anthropic
# ---------------------------------------------------------------------------

def run_main(monkeypatch, capsys, argv, stdin=""):
    monkeypatch.setattr(sys, "argv", ["triage.py", *argv])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    try:
        code = triage.main()
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr()


def test_cli_json_output_matches_schema(fake_anthropic, monkeypatch, capsys):
    code, captured = run_main(
        monkeypatch, capsys, ["--provider", "anthropic", "--json", str(ROOT / "sample.txt")]
    )

    assert code == 0
    data = json.loads(captured.out)
    assert triage.TriageResult.model_validate(data) == make_result()


def test_cli_sends_file_contents_to_claude(fake_anthropic, monkeypatch, capsys):
    run_main(monkeypatch, capsys, ["--provider", "anthropic", "--json", str(ROOT / "sample.txt")])

    _, kwargs = single_call(fake_anthropic)
    assert "Invoice #4471" in as_text(list(kwargs["messages"])[-1]["content"])


def test_cli_does_not_call_openai(fake_anthropic, monkeypatch, capsys):
    class NoOpenAI:
        def __init__(self, *args, **kwargs):
            raise AssertionError("--provider anthropic must not create an OpenAI client")

    openai = pytest.importorskip("openai")
    monkeypatch.setattr(openai, "OpenAI", NoOpenAI)
    monkeypatch.setattr(triage, "OpenAI", NoOpenAI, raising=False)

    code, _ = run_main(monkeypatch, capsys, ["--provider", "anthropic", "--json"], stdin="hello")

    assert code == 0


def test_cli_refusal_exits_cleanly(fake_anthropic, monkeypatch, capsys):
    """Refused message -> non-zero exit, message on stderr, no JSON, no traceback."""
    fake_anthropic.scenario = "refusal"

    try:
        code, (out, err) = run_main(
            monkeypatch, capsys, ["--provider", "anthropic", "--json"], stdin="refuse me"
        )
    except Exception as exc:
        pytest.fail(f"main() let a {type(exc).__name__} escape on refusal: {exc}")

    assert code not in (0, None), "refusal should produce a non-zero exit code"
    assert out.strip() == "", f"nothing should be printed to stdout on refusal, got: {out!r}"
    assert "refus" in err.lower()


# ---------------------------------------------------------------------------
# 6. Optional live test against the real API
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_LIVE_TEST") and os.getenv("ANTHROPIC_API_KEY")),
    reason="set ANTHROPIC_LIVE_TEST=1 and ANTHROPIC_API_KEY to call the real API",
)
def test_live_anthropic_on_sample():
    text = (ROOT / "sample.txt").read_text(encoding="utf-8")

    result = triage.triage_anthropic(text)

    assert isinstance(result, triage.TriageResult)
    assert result.category == triage.Category.invoice
    assert 1 <= result.priority <= 5
    assert result.summary.strip()
