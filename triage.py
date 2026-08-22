#!/usr/bin/env python3
"""
LLM Inbox Triage - Project 1

Takes raw text (email / ticket / message) and returns a structured analysis:
category, priority, summary, suggested reply, and extracted data as JSON.

This is a STARTER SKELETON. The TODOs are yours to implement - that's the point.
Learn by filling them in. Don't just paste an answer; understand each piece.

Usage:
    python triage.py sample.txt
    python triage.py sample.txt --provider anthropic
    cat mail.txt | python triage.py --provider openai
"""

import argparse
import json
import os
import sys

# TODO: load .env (python-dotenv) so API keys are picked up automatically
# from dotenv import load_dotenv
# load_dotenv()


SYSTEM_PROMPT = """You are an inbox triage assistant. Given a raw message, you return
a STRICT JSON object and nothing else. The JSON must match exactly this schema:

{
  "category": "urgent | invoice | spam | question | newsletter | ignore | other",
  "priority": 1-5,               // 1 = trivial, 5 = drop-everything
  "summary": "one sentence",
  "suggested_reply": "a short draft reply, or null if none is appropriate",
  "extracted": {
    "dates": [],                 // ISO dates or human dates found
    "amounts": [],               // money amounts found
    "names": [],                 // people / companies mentioned
    "deadlines": []              // explicit deadlines
  }
}

Rules:
- Output ONLY the JSON. No markdown, no code fences, no commentary.
- If a field has no data, use an empty list or null.
- Be conservative with priority; reserve 5 for genuinely time-critical items.
"""


def read_input(path: str | None) -> str:
    """Read the message text from a file path, or stdin if no path given."""
    # TODO: if path is None, read from sys.stdin
    # TODO: otherwise open(path) and return its contents
    raise NotImplementedError("read_input")


def triage_openai(text: str) -> dict:
    """Call the OpenAI API and return the parsed JSON result."""
    # TODO: from openai import OpenAI; client = OpenAI()
    # TODO: use a chat completion with SYSTEM_PROMPT + the message text
    # TODO: request JSON output (response_format / structured output)
    # TODO: parse the response into a dict and return it
    raise NotImplementedError("triage_openai")


def triage_anthropic(text: str) -> dict:
    """Call the Anthropic API and return the parsed JSON result."""
    # TODO: import anthropic; client = anthropic.Anthropic()
    # TODO: use messages.create with system=SYSTEM_PROMPT, user text
    # TODO: parse the returned text as JSON and return a dict
    raise NotImplementedError("triage_anthropic")


def render(result: dict) -> None:
    """Pretty-print the result to the terminal."""
    # TODO: use `rich` for a nice table/panel, or fall back to json.dumps
    print(json.dumps(result, indent=2, ensure_ascii=False))


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

    text = read_input(args.path)

    # TODO: add basic error handling + a retry on transient API errors
    if args.provider == "openai":
        result = triage_openai(text)
    else:
        result = triage_anthropic(text)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        render(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
