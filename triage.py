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

from loguru import logger
from pydantic import BaseModel, Field

# TODO: load .env (python-dotenv) so API keys are picked up automatically
# from dotenv import load_dotenv
# load_dotenv()


# ---------------------------------------------------------------------------
# 1. THE SCHEMA  (this replaces the old "return STRICT JSON" prompt)
#    Define the shape once, in code. The API will guarantee the model obeys it.
# ---------------------------------------------------------------------------

class Category(str, Enum):
    urgent = "urgent"
    invoice = "invoice"
    spam = "spam"
    question = "question"
    newsletter = "newsletter"
    ignore = "ignore"
    other = "other"


class Extracted(BaseModel):
    dates: list[str] = Field(default_factory=list, description="ISO or human dates found")
    amounts: list[str] = Field(default_factory=list, description="Money amounts found")
    names: list[str] = Field(default_factory=list, description="People / companies mentioned")
    deadlines: list[str] = Field(default_factory=list, description="Explicit deadlines")


class TriageResult(BaseModel):
    category: Category
    priority: int = Field(ge=1, le=5, description="1 = trivial, 5 = drop-everything")
    summary: str = Field(description="One sentence.")
    suggested_reply: str | None = Field(
        default=None, description="Short draft reply, or null if none is appropriate."
    )
    extracted: Extracted


# Note how short the system prompt is now: we no longer beg for JSON formatting.
# The schema does that job. We only describe the *task* and the *judgement*.
SYSTEM_PROMPT = """You are an inbox triage assistant. Analyse the message and fill in
the structured fields. Be conservative with priority; reserve 5 for genuinely
time-critical items. If a field has no data, use an empty list or null."""


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

def triage_openai(text: str) -> TriageResult:
    """Call OpenAI with Structured Outputs and return a validated TriageResult."""
    # TODO: from openai import OpenAI; client = OpenAI()
    # TODO: response = client.responses.parse(
    #           model="gpt-5.6",           # or a current model you have access to
    #           input=[
    #               {"role": "system", "content": SYSTEM_PROMPT},
    #               {"role": "user", "content": text},
    #           ],
    #           text_format=TriageResult,  # <-- schema is guaranteed
    #       )
    # TODO: handle response.output_parsed being None (a refusal) gracefully
    # TODO: return response.output_parsed
    raise NotImplementedError("triage_openai")


def triage_anthropic(text: str) -> TriageResult:
    """Call Anthropic and return a validated TriageResult.

    Anthropic has no responses.parse helper, so the idiomatic way is a tool whose
    input_schema is TriageResult's JSON schema (TriageResult.model_json_schema()),
    force the model to call it, then validate the tool input with Pydantic.
    """
    # TODO: import anthropic; client = anthropic.Anthropic()
    # TODO: build a single tool from TriageResult.model_json_schema()
    # TODO: client.messages.create(..., tools=[tool], tool_choice={"type": "tool", "name": ...})
    # TODO: pull the tool_use block's input and do TriageResult.model_validate(input)
    raise NotImplementedError("triage_anthropic")


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

    # TODO: add basic error handling + a retry on transient API errors
    if args.provider == "openai":
        result = triage_openai(text)
    else:
        result = triage_anthropic(text)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        render(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
