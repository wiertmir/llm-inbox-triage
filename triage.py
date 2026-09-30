#!/usr/bin/env python3
"""
LLM Inbox Triage - Project 1  (2026 edition: Structured Outputs)

Takes raw text (email / ticket / message) and returns a STRUCTURED analysis:
category, priority, summary, suggested reply, and extracted data.

>>> KEY 2026 IDEA <<<
We do NOT ask the model for "STRICT JSON and nothing else" and then json.loads()
the raw text (that's the old, fragile pattern). Instead we use STRUCTURED OUTPUTS:
define the shape as a Pydantic model and let the API GUARANTEE the schema.
  - OpenAI:    client.responses.parse(..., text_format=TriageResult) -> .output_parsed
  - Anthropic: tool/JSON schema with strict validation, then validate with Pydantic
See ../../../openai-old-vs-new.md for the old->new cheat sheet.

Usage:
    python triage.py sample.txt
    python triage.py sample.txt --provider anthropic
    cat mail.txt | python triage.py --provider openai
"""

import argparse
import os
import sys
from enum import Enum
from datetime import date

from loguru import logger

from pydantic import BaseModel, Field, ValidationError
from pydantic.config import JsonDict
from typing import Iterable

from openai import OpenAI
from openai.types.responses import ParsedResponse, ResponseInputParam

from anthropic import Anthropic
from anthropic.types import MessageParam

# TODO: load .env (python-dotenv) so API keys are picked up automatically
# from dotenv import load_dotenv
# load_dotenv()


# ---------------------------------------------------------------------------
# 1. THE SCHEMA  (this replaces the old "return STRICT JSON" prompt)
#    Define the shape once, in code. The API will guarantee the model obeys it.
# ---------------------------------------------------------------------------

DEFAULT_CCY = "JPY"


class Category(str, Enum):
    urgent = "urgent"
    invoice = "invoice"
    spam = "spam"
    question = "question"
    newsletter = "newsletter"
    ignore = "ignore"
    other = "other"


def _drop_default(schema: JsonDict) -> None:
    # Strict schemas make every field required, so keep the default out of the API schema.
    schema.pop("default", None)


class Money(BaseModel):
    amount: float = Field(description="Numeric value, e.g. 128000 for '128,000 JPY'")
    currency: str = Field(
        default=DEFAULT_CCY,
        description=f"ISO 4217 code, e.g. USD, EUR; {DEFAULT_CCY} if the message names none",
        json_schema_extra=_drop_default,
    )


class Extracted(BaseModel):
    dates: list[date] = Field(
        default_factory=list,
        description="Dates mentioned, as YYYY-MM-DD; resolve relative dates like 'Friday'",
    )
    amounts: list[Money] = Field(default_factory=list, description="Money amounts found")
    names: list[str] = Field(default_factory=list, description="People / companies mentioned")
    deadlines: list[date] = Field(
        default_factory=list,
        description="Explicit deadlines, as YYYY-MM-DD; resolve relative dates like 'Friday'",
    )


class TriageResult(BaseModel):
    category: Category
    priority: int = Field(ge=1, le=5, description="1 = trivial, 5 = drop-everything")
    summary: str = Field(description="One sentence.")
    suggested_reply: str | None = Field(
        default=None, description="Short draft reply, or null if none is appropriate."
    )
    extracted: Extracted

class TriageRefusal(Exception):
    """The model declined or produced no structured output."""


# Note how short the system prompt is now: we no longer beg for JSON formatting.
# The schema does that job. We only describe the *task* and the *judgement*.
SYSTEM_PROMPT = f"""You are an inbox triage assistant. Analyse the message and fill in
the structured fields. Be conservative with priority; reserve 5 for genuinely
time-critical items. If a field has no data, use an empty list or null.

Write every date as YYYY-MM-DD. Convert relative or partial dates such as "Friday",
"tomorrow" or "end of the month" to the actual calendar date, counting from the
message's sent date if it has one, otherwise from today ({date.today().isoformat()}).
Leave out dates you cannot pin down.

For amounts, give the number and its ISO 4217 currency code; use {DEFAULT_CCY} if the
message does not name a currency."""

OPENAI_MODEL = "gpt-5.6"
ANTHROPIC_MODEL = "claude-sonnet-5-5"


# ---------------------------------------------------------------------------
# 2. IO
# ---------------------------------------------------------------------------

def read_input(path: str | None) -> str:
    """Read the message text from a file path, or stdin if no path given."""
    if path is None:
        source = "stdin"
        if sys.stdin.isatty():
            # No path and nothing piped in: usually a forgotten argument.
            logger.warning("No file given, reading message from terminal (finish with Ctrl-D)")
        text = sys.stdin.read()
    else:
        source = path
        # colors=True enables <tag> markup; values passed as arguments are
        # inserted as plain text, so a path containing "<" is safe.
        logger.opt(colors=True).debug("Reading message from <cyan>{path}</cyan>", path=path)
        with open(path, encoding="utf-8") as fd:
            text = fd.read()
    
    if not text.strip():
        logger.opt(colors=True).warning("Input from <cyan>{source}</cyan> is empty", source=source)
    return text


# ---------------------------------------------------------------------------
# 3. PROVIDERS  (both return a validated TriageResult, not a raw dict)
# ---------------------------------------------------------------------------

def _refusal_text(response: ParsedResponse[TriageResult]) -> str | None:
    for output in response.output:
        if output.type != "message":
            continue

        for part in output.content:
            if part.type == "refusal":
                return part.refusal

    return None

def triage_openai(text: str) -> TriageResult:
    """Call OpenAI with Structured Outputs and return a validated TriageResult."""
    logger.info("Using OpenAI provider")
    client = OpenAI()
    logger.debug("OpenAI base URL: {url}", url=client.base_url)

    messages: ResponseInputParam = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': text},
    ]

    logger.info("Using AI model: {model}", model=OPENAI_MODEL)
    try:
        response = client.responses.parse(
            model=OPENAI_MODEL,
            input=messages,
            text_format=TriageResult
        )
    except ValidationError as exc:
        logger.error("OpenAI output does not match TriageResult: {error}", error=exc)
        raise

    parsed = response.output_parsed
    if parsed is None:
        reason = _refusal_text(response)
        if reason:
            raise TriageRefusal(f"Model refused: {reason}")
        raise TriageRefusal("Model refused or returned no structured output")
    
    return parsed


def triage_anthropic(text: str) -> TriageResult:
    """Call Anthropic and return a validated TriageResult.
    """
    logger.info("Using Anthropic provider")
    client = Anthropic()
    logger.debug("Anthropic base URL: {url}", url=client.base_url)

    messages: Iterable[MessageParam] = [
        {'role': 'user', 'content': text}
    ]

    logger.info("Using AI model: {model}", model=ANTHROPIC_MODEL)

    response = client.messages.parse(
        model=ANTHROPIC_MODEL,
        system=SYSTEM_PROMPT,
        max_tokens=1024,
        messages=messages,
        output_format=TriageResult
    )

    if response.stop_reason == "refusal":
        reason = "".join(block.text for block in response.content if block.type == "text")
        raise TriageRefusal(f"Model refused: {reason}" if reason else "Model refused the request")

    parsed = response.parsed_output
    if parsed is None:
        raise TriageRefusal(
            f"Model returned no structured output (stop_reason={response.stop_reason})"
        )

    return parsed


# ---------------------------------------------------------------------------
# 4. OUTPUT
# ---------------------------------------------------------------------------

def render(result: TriageResult) -> None:
    """Pretty-print the result to the terminal."""
    # TODO: use `rich` (Panel/Table) for a nice view; fall back to plain text.
    # Tip: result.model_dump() gives you a dict if you want json.dumps for --json.
    print(result.model_dump_json(indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM inbox triage")
    parser.add_argument("path", nargs="?", help="Text file to triage (default: stdin)")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Which LLM provider to use",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON only")
    args = parser.parse_args()

    # Log to the console at LOG_LEVEL (default DEBUG, so everything shows).
    # Set LOG_LEVEL=WARNING for quiet runs. Logs go to stderr, not stdout, so
    # the --json output stays clean when piped into jq or a file.
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
        # Z = UTC offset, e.g. 2026-09-27 20:49:32.213 +07:00
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    try:
        text = read_input(args.path)
    except FileNotFoundError:
        parser.error(f"File not found: {args.path}")        

    try:
        result = triage_openai(text) if args.provider == "openai" else triage_anthropic(text)
    except TriageRefusal as exc:
        logger.error("{}", exc)
        return 1

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        render(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
