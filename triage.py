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
    python triage.py sample.txt --json --out       # JSON to stdout and out/sample.out.json
    python triage.py sample.txt --json -o res.json # JSON to stdout and res.json
"""

import argparse
import os
import sys
import time
from enum import Enum
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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

LOG_DIR = Path(__file__).resolve().parent / "logs"
OUT_DIR = Path("out")

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

# Look of each category: (emoji, colour).
CATEGORY_STYLE: dict[Category, tuple[str, str]] = {
    Category.urgent: ("🚨", "bold red"),
    Category.invoice: ("💸", "bold yellow"),
    Category.spam: ("🗑️", "bold magenta"),
    Category.question: ("❓", "bold cyan"),
    Category.newsletter: ("📰", "bold blue"),
    Category.ignore: ("💤", "bold bright_black"),
    Category.other: ("📌", "bold white"),
}

# Priority 1..5 -> colour and label, from calm green to alarm red.
PRIORITY_STYLE = {
    1: ("green", "trivial"),
    2: ("bright_green", "low"),
    3: ("yellow", "normal"),
    4: ("dark_orange", "high"),
    5: ("bold red", "drop everything"),
}


def _priority_meter(priority: int) -> Text:
    colour, label = PRIORITY_STYLE[priority]
    meter = Text()
    meter.append("█" * priority * 2, style=colour)
    meter.append("░" * (5 - priority) * 2, style="bright_black")
    meter.append(f"  {priority}/5 ", style=f"bold {colour}")
    meter.append(label, style=f"italic {colour}")
    return meter


def _days_away(day: date) -> Text:
    delta = (day - date.today()).days
    if delta < 0:
        return Text(f"{-delta}d ago", style="bright_black")
    if delta == 0:
        return Text("today", style="bold red")
    if delta <= 3:
        return Text(f"in {delta}d", style="bold dark_orange")
    return Text(f"in {delta}d", style="green")


def _extracted_table(extracted: Extracted) -> Table | None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, expand=True, header_style="bold")
    table.add_column("", width=2)
    table.add_column("Kind", style="bold")
    table.add_column("Value", ratio=1)
    table.add_column("When", justify="right")

    deadlines = set(extracted.deadlines)
    for day in sorted(extracted.deadlines):
        table.add_row("⏰", "Deadline", Text(day.strftime("%a %d %b %Y"), style="bold"), _days_away(day))
    for day in sorted(set(extracted.dates) - deadlines):
        table.add_row("📅", "Date", day.strftime("%a %d %b %Y"), _days_away(day))
    for money in extracted.amounts:
        table.add_row("💰", "Amount", Text(f"{money.amount:,.2f} {money.currency}", style="bold green"), "")
    for name in extracted.names:
        table.add_row("👤", "Name", name, "")

    return table if table.row_count else None


def render(result: TriageResult, console: Console | None = None) -> None:
    """Pretty-print the result to the terminal as a triage card."""
    console = console or Console()
    emoji, colour = CATEGORY_STYLE[result.category]

    header = Table.grid(expand=True, padding=(0, 1))
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(
        Text(f"{emoji}  {result.category.value.upper()}", style=colour),
        _priority_meter(result.priority),
    )

    parts: list = [header, Padding(Text(result.summary, style="italic"), (1, 0, 0, 0))]

    table = _extracted_table(result.extracted)
    if table is not None:
        parts.append(Padding(table, (1, 0, 0, 0)))

    if result.suggested_reply:
        parts.append(
            Padding(
                Panel(
                    Text(result.suggested_reply),
                    title="✍️  Suggested reply",
                    title_align="left",
                    border_style="bright_black",
                    box=box.ROUNDED,
                    padding=(0, 1),
                ),
                (1, 0, 0, 0),
            )
        )
    else:
        parts.append(Padding(Text("No reply needed.", style="bright_black"), (1, 0, 0, 0)))

    console.print(
        Panel(
            Group(*parts),
            title="[bold]📨 Inbox Triage[/bold]",
            title_align="left",
            border_style=PRIORITY_STYLE[result.priority][0],
            box=box.HEAVY,
            padding=(1, 2),
        )
    )


def default_out_path(input_path: str | None) -> Path:
    """out/<input name>.out.json, or a timestamped name when reading stdin."""
    if input_path is None:
        stem = f"stdin_{datetime.now():%Y%m%d-%H%M%S}"
    else:
        stem = Path(input_path).stem
    return OUT_DIR / f"{stem}.out.json"


def write_output(result: TriageResult, path: Path) -> None:
    """Save the result as JSON, creating parent folders as needed."""
    log = logger.opt(colors=True)
    path = path.resolve()
    log.info("Writing result to <cyan>{path}</cyan>", path=path)

    if not path.parent.exists():
        log.debug("Creating folder <cyan>{folder}</cyan>", folder=path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        log.debug("Overwriting existing file <cyan>{path}</cyan>", path=path)

    data = result.model_dump_json(indent=2) + "\n"
    path.write_text(data, encoding="utf-8")
    log.info("Wrote {size:,} bytes to <cyan>{path}</cyan>", size=len(data.encode("utf-8")), path=path)


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
    parser.add_argument(
        "-o",
        "--out",
        nargs="?",
        const="",
        metavar="PATH",
        help="With --json, also save the result to PATH (default: out/<input name>.out.json)",
    )
    args = parser.parse_args()
    if args.out is not None and not args.json:
        parser.error("--out requires --json")
    # "--out sample.txt" would take the input file as the output path and
    # overwrite it, so insist on a .json output name.
    if args.out and Path(args.out).suffix.lower() != ".json":
        parser.error(f"--out path must end in .json, got {args.out!r} (put the input file before --out)")

    # Log to logs/triage_YYYY-MM-DD.log at LOG_LEVEL (default DEBUG), starting
    # a new file at midnight. Warnings and errors are also echoed to stderr so
    # problems stay visible; with --json only errors are, so pipelines stay quiet.
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS Z} | {level: <8} | {message}"
    logger.remove()
    logger.add(
        LOG_DIR / "triage_{time:YYYY-MM-DD}.log",
        level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
        # Z = UTC offset, e.g. 2026-09-27 20:49:32.213 +07:00
        format=log_format,
        rotation="00:00",
        encoding="utf-8",
    )
    logger.add(
        sys.stderr,
        level="ERROR" if args.json else "WARNING",
        format="<level>{level: <8}</level> | <level>{message}</level>",
    )

    try:
        text = read_input(args.path)
    except FileNotFoundError:
        parser.error(f"File not found: {args.path}")        

    if args.provider == "openai":
        triage, provider_name, model = triage_openai, "OpenAI", OPENAI_MODEL
    else:
        triage, provider_name, model = triage_anthropic, "Anthropic", ANTHROPIC_MODEL

    # Spinner and timing go to stderr, and are silenced entirely with --json
    # so the output can be part of a pipeline.
    status_console = Console(stderr=True, quiet=args.json)
    status_text = (
        f"[bold]📨 Triaging message[/bold] ({len(text):,} chars) · "
        f"waiting for [cyan]{model}[/cyan] on {provider_name}…"
    )
    started = time.perf_counter()
    try:
        with status_console.status(status_text, spinner="dots"):
            result = triage(text)
    except TriageRefusal as exc:
        logger.error("{}", exc)
        return 1
    elapsed = time.perf_counter() - started
    logger.info("Got response from {model} in {elapsed:.1f}s", model=model, elapsed=elapsed)
    status_console.print(f"[green]✓[/green] Response from [cyan]{model}[/cyan] in {elapsed:.1f}s", highlight=False)

    if args.out is not None:
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = default_out_path(args.path)
            logger.opt(colors=True).debug(
                "No --out path given, using default <cyan>{path}</cyan>", path=out_path
            )
        try:
            write_output(result, out_path)
        except OSError as exc:
            logger.error("Cannot write {path}: {error}", path=out_path, error=exc)
            return 1

    if args.json:
        logger.debug("Printing JSON result to stdout")
        print(result.model_dump_json(indent=2))
    else:
        logger.debug("Rendering result card")
        render(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
