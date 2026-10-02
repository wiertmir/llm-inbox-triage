#!/usr/bin/env python3
"""
LLM Inbox Triage - Project 1  (2026 edition: Structured Outputs)

Takes raw text (email / ticket / message) and returns a STRUCTURED analysis:
category, priority, summary, suggested reply, and extracted data.

>>> KEY 2026 IDEA <<<
We do NOT ask the model for "STRICT JSON and nothing else" and then json.loads()
the raw text (that's the old, fragile pattern). Instead we use STRUCTURED OUTPUTS:
define the shape as a Pydantic model and let the API GUARANTEE the schema.
  - OpenAI:    await client.responses.parse(..., text_format=TriageResult) -> .output_parsed
  - Anthropic: await client.messages.parse(..., output_format=TriageResult) -> .parsed_output
Both providers use the async SDK clients (AsyncOpenAI / AsyncAnthropic).
Transient API errors (rate limit, timeout, overload, 5xx) are retried with
exponential backoff via tenacity (honouring the server's Retry-After hint);
everything else fails fast with a short message. API keys are read from the
environment or from a .env file.
See ../../../openai-old-vs-new.md for the old->new cheat sheet.

Usage:
    python triage.py sample.txt
    python triage.py sample.txt --provider anthropic
    cat mail.txt | python triage.py --provider openai
    python triage.py sample.txt --json --out       # JSON to stdout and out/sample.out.json
    python triage.py sample.txt --json -o res.json # JSON to stdout and res.json
"""

import argparse
import asyncio
import json
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
from typing import Awaitable, Callable, Iterable, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

import openai
from openai import AsyncOpenAI
from openai.types.responses import ParsedResponse, ResponseInputParam

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from dotenv import load_dotenv


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


class MissingApiKey(Exception):
    """The provider's API key environment variable is not set."""


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

OPENAI_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"

# Room for the summary, reply and extracted lists; longer output gets cut off.
ANTHROPIC_MAX_TOKENS = 2048

# Per-request timeout. The SDK default is 10 minutes, which across all retry
# attempts could keep the spinner going for nearly an hour on a stuck request.
REQUEST_TIMEOUT_SECONDS = 60.0

# Retry policy for transient API errors: up to 5 attempts, waiting about
# 1s, 2s, 4s, 8s in between (plus up to 1s of jitter), i.e. ~15-19s in total.
MAX_ATTEMPTS = 5
BACKOFF_MAX_SECONDS = 20
# Same status codes the SDKs themselves retry; any 5xx (incl. 529 overloaded) too.
RETRYABLE_STATUS = {408, 409, 429}
# Longest wait we accept from a server's Retry-After header.
RETRY_AFTER_MAX_SECONDS = 60


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
    
    logger.opt(colors=True).debug(
        "Read {size:,} chars from <cyan>{source}</cyan>", size=len(text), source=source
    )
    return text


# ---------------------------------------------------------------------------
# 3. PROVIDERS  (both return a validated TriageResult, not a raw dict)
# ---------------------------------------------------------------------------

def _is_transient(exc: BaseException) -> bool:
    """Rate limits, timeouts, dropped connections and server-side errors."""
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return True  # includes APITimeoutError
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return exc.status_code in RETRYABLE_STATUS or exc.status_code >= 500
    return False


def _describe_error(exc: BaseException) -> str:
    """Short one-line description of an API error, e.g. 'RateLimitError (429): ...'."""
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return f"{type(exc).__name__} ({exc.status_code}): {exc.message}"
    return f"{type(exc).__name__}: {exc}"


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The server's own wait hint (retry-after-ms / Retry-After headers), if any."""
    if not isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return None
    headers = exc.response.headers
    try:
        if "retry-after-ms" in headers:
            return float(headers["retry-after-ms"]) / 1000
        if "retry-after" in headers:
            return float(headers["retry-after"])
    except ValueError:
        pass  # e.g. the HTTP-date form; our own backoff is good enough then
    return None


_backoff = wait_exponential(multiplier=1, max=BACKOFF_MAX_SECONDS) + wait_random(0, 1)


def _wait(state: RetryCallState) -> float:
    """Exponential backoff, but at least as long as the server asked (capped)."""
    delay = _backoff(state)
    exc = state.outcome.exception() if state.outcome else None
    hint = _retry_after_seconds(exc) if exc else None
    if hint is not None:
        delay = max(delay, min(hint, RETRY_AFTER_MAX_SECONDS))
    return delay


def _retrying(provider: str) -> AsyncRetrying:
    """Retry transient errors with exponential backoff, logging each wait."""

    def log_retry(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        wait = state.next_action.sleep if state.next_action else 0.0
        logger.warning(
            "{provider} request failed (attempt {attempt}/{max}): {error} - retrying in {wait:.1f}s",
            provider=provider,
            attempt=state.attempt_number,
            max=MAX_ATTEMPTS,
            error=_describe_error(exc) if exc else "unknown error",
            wait=wait,
        )

    return AsyncRetrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=_wait,
        before_sleep=log_retry,
        reraise=True,
    )


T = TypeVar("T")


async def _with_retry(provider: str, call: Callable[[], Awaitable[T]]) -> T:
    """Await call(), retrying transient errors; the last error is re-raised."""
    async for attempt in _retrying(provider):
        with attempt:
            return await call()
    # Unreachable: with reraise=True tenacity either returns above or raises.
    raise AssertionError("retry loop ended without a result")


def _require_key(*env_names: str) -> None:
    """Fail fast, before any request, if none of the key variables is set."""
    if not any(os.getenv(name) for name in env_names):
        raise MissingApiKey(
            f"{env_names[0]} is not set - export it in your environment or add it to .env"
        )


def _refusal_text(response: ParsedResponse[TriageResult]) -> str | None:
    for output in response.output:
        if output.type != "message":
            continue

        for part in output.content:
            if part.type == "refusal":
                return part.refusal

    return None

async def triage_openai(text: str) -> TriageResult:
    """Call OpenAI with Structured Outputs and return a validated TriageResult."""
    logger.info("Using OpenAI provider")
    _require_key(OPENAI_KEY_ENV)
    # Retries are ours (see _retrying), so switch off the SDK's built-in ones.
    client = AsyncOpenAI(max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS)
    logger.debug("OpenAI base URL: {url}", url=client.base_url)

    messages: ResponseInputParam = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': text},
    ]

    logger.info("Using AI model: {model}", model=OPENAI_MODEL)
    response = await _with_retry(
        "OpenAI",
        lambda: client.responses.parse(
            model=OPENAI_MODEL,
            input=messages,
            text_format=TriageResult
        ),
    )

    parsed = response.output_parsed
    if parsed is None:
        reason = _refusal_text(response)
        if reason:
            raise TriageRefusal(f"Model refused: {reason}")
        if response.status == "incomplete":
            why = response.incomplete_details.reason if response.incomplete_details else None
            raise TriageRefusal(f"Model output is incomplete ({why or 'unknown reason'})")
        raise TriageRefusal("Model refused or returned no structured output")
    
    return parsed


async def triage_anthropic(text: str) -> TriageResult:
    """Call Anthropic and return a validated TriageResult.
    """
    logger.info("Using Anthropic provider")
    _require_key(ANTHROPIC_KEY_ENV, "ANTHROPIC_AUTH_TOKEN")
    # Retries are ours (see _retrying), so switch off the SDK's built-in ones.
    client = AsyncAnthropic(max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS)
    logger.debug("Anthropic base URL: {url}", url=client.base_url)

    messages: Iterable[MessageParam] = [
        {'role': 'user', 'content': text}
    ]

    logger.info("Using AI model: {model}", model=ANTHROPIC_MODEL)

    response = await _with_retry(
        "Anthropic",
        lambda: client.messages.parse(
            model=ANTHROPIC_MODEL,
            system=SYSTEM_PROMPT,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            messages=messages,
            output_format=TriageResult
        ),
    )

    if response.stop_reason == "refusal":
        reason = "".join(block.text for block in response.content if block.type == "text")
        raise TriageRefusal(f"Model refused: {reason}" if reason else "Model refused the request")

    if response.stop_reason == "max_tokens":
        raise TriageRefusal(f"Model output was cut off at max_tokens={ANTHROPIC_MAX_TOKENS}")

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
    # Pick up API keys from .env; variables already set in the environment win.
    load_dotenv()
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

    source = args.path or "stdin"
    try:
        text = read_input(args.path)
    except FileNotFoundError:
        parser.error(f"File not found: {args.path}")
    except UnicodeDecodeError:
        parser.error(f"Cannot read {source}: not UTF-8 text")
    except OSError as exc:  # a directory, no permission, ...
        # Windows reports a directory as "Permission denied", so name it ourselves.
        reason = "is a directory" if Path(source).is_dir() else exc.strerror or exc
        parser.error(f"Cannot read {source}: {reason}")
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130

    if not text.strip():
        logger.error("Nothing to triage: {source} is empty", source=source)
        return 1

    triage: Callable[[str], Awaitable[TriageResult]]
    if args.provider == "openai":
        triage, provider_name, model = triage_openai, "OpenAI", OPENAI_MODEL
        key_env = OPENAI_KEY_ENV
    else:
        triage, provider_name, model = triage_anthropic, "Anthropic", ANTHROPIC_MODEL
        key_env = ANTHROPIC_KEY_ENV

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
            result = asyncio.run(triage(text))
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    except (TriageRefusal, MissingApiKey) as exc:
        logger.error("{}", exc)
        return 1
    except (openai.AuthenticationError, anthropic.AuthenticationError) as exc:
        logger.error("{provider} rejected the API key - check {env}: {error}",
                     provider=provider_name, env=key_env, error=_describe_error(exc))
        return 1
    except (openai.APIError, anthropic.APIError) as exc:
        if _is_transient(exc):
            logger.error("{provider} still failing after {n} attempts, giving up: {error}",
                         provider=provider_name, n=MAX_ATTEMPTS, error=_describe_error(exc))
        else:
            logger.error("{provider} request failed: {error}",
                         provider=provider_name, error=_describe_error(exc))
        return 1
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.error("{provider} returned malformed output that does not match TriageResult",
                     provider=provider_name)
        logger.debug("Malformed output details: {error}", error=exc)
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
