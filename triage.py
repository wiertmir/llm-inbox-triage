#!/usr/bin/env python3
"""
LLM Inbox Triage - Project 1  (2026 edition: Structured Outputs)

Takes raw text (email / ticket / message) and returns a STRUCTURED analysis:
category, priority, summary, suggested reply, and extracted data.

>>> KEY 2026 IDEA <<<
We do NOT ask the model for "STRICT JSON and nothing else" and then json.loads()
the raw text (that's the old, fragile pattern). Instead we use STRUCTURED OUTPUTS:
define the analysis shape as a Pydantic model and let the API GUARANTEE the schema.
  - OpenAI:    await client.responses.parse(..., text_format=TriageAnalysis) -> .output_parsed
  - Anthropic: await client.messages.parse(..., output_format=TriageAnalysis) -> .parsed_output
The application adds the required input ID to produce a TriageResult.
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
    python triage.py sample.txt --create-events   # Model-selected Google Calendar entries
    python triage.py --batch evals\\samples --json # One aggregate report, also saved to out/
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from contextvars import ContextVar
from enum import Enum
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from pydantic import BaseModel, Field, ValidationError, model_validator
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

from google_calendar_auth import GoogleCalendarAuthError, get_google_calendar_access_token
from me_calendar import MeCalendarError, build_me_event_body, create_me_calendar_entry
from calendar_tools import (
    CALENDAR_SYSTEM_PROMPT,
    Calendar,
    CalendarEvent,
    CalendarToolError,
    CreatedCalendarEvent,
    validate_event_times,
)


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


class TriageAnalysis(BaseModel):
    category: Category
    priority: int = Field(ge=1, le=5, description="1 = trivial, 5 = drop-everything")
    summary: str = Field(description="One sentence.")
    suggested_reply: str | None = Field(
        default=None, description="Short draft reply, or null if none is appropriate."
    )
    extracted: Extracted
    proposed_events: list[CalendarEvent] = Field(
        default_factory=list,
        description="Calendar-ready meetings, appointments, or deadlines; empty if none are appropriate",
    )

    @model_validator(mode="after")
    def validate_proposed_events(self) -> "TriageAnalysis":
        dates = set(self.extracted.dates) | set(self.extracted.deadlines)
        for event in self.proposed_events:
            start_date = event.start.date() if isinstance(event.start, datetime) else event.start
            if start_date not in dates:
                raise ValueError("Proposed event start must belong to extracted dates/deadlines")
        return self


class TriageResult(TriageAnalysis):
    id: str = Field(description="Input filename, including extension, or ***stdin*** for stdin")


class TriageRefusal(Exception):
    """The model declined or produced no structured output."""


class MissingApiKey(Exception):
    """The provider's API key environment variable is not set."""


class CalendarEntryError(Exception):
    """Calendar event creation failed."""

    def __init__(self, message: str, *, event_may_exist: bool = True) -> None:
        super().__init__(message)
        self.event_may_exist = event_may_exist


def create_calendar_entry(
    calendar: Calendar | str,
    title: str,
    start: date | datetime,
    end: date | datetime,
    description: str = "",
    access_token: str | None = None,
) -> dict[str, Any]:
    """Create an event in Google, Me, or Hotmail/Outlook Calendar.

    ``start`` and ``end`` must both be timezone-aware datetimes, or dates for
    a Google all-day event (exclusive end date). Tokens may be passed directly
    or supplied through GOOGLE_CALENDAR_ACCESS_TOKEN /
    HOTMAIL_CALENDAR_ACCESS_TOKEN. Without a Google token, Desktop app OAuth
    credentials from GOOGLE_CALENDAR_CLIENT_SECRETS_FILE are used to sign in
    once, then reuse or refresh tokens from the OS credential store.
    Me uses ME_BASE / ME_CA_FILE and its own PKCE sign-in with OS token caching.
    """
    calendar = Calendar(calendar)
    validate_event_times(start, end)
    if calendar == Calendar.me:
        try:
            return create_me_calendar_entry(title, start, end, description, access_token=access_token)
        except MeCalendarError as exc:
            raise CalendarEntryError(str(exc), event_may_exist=exc.event_may_exist) from exc
    endpoints = {
        Calendar.google: (
            "GOOGLE_CALENDAR_ACCESS_TOKEN",
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
        ),
        Calendar.hotmail: ("HOTMAIL_CALENDAR_ACCESS_TOKEN", "https://graph.microsoft.com/v1.0/me/events"),
    }
    if calendar == Calendar.hotmail and not isinstance(start, datetime):
        raise ValueError("All-day events are currently supported only for Google Calendar")

    token_env, endpoint = endpoints[calendar]
    token = access_token or os.getenv(token_env)
    if not token and calendar == Calendar.google:
        try:
            token = get_google_calendar_access_token()
        except GoogleCalendarAuthError as exc:
            raise CalendarEntryError(str(exc)) from exc
    if not token:
        raise CalendarEntryError(
            f"No access token provided; pass access_token or set {token_env}"
        )

    if isinstance(start, datetime) and isinstance(end, datetime):
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        start_data = {"dateTime": start_utc.isoformat(), "timeZone": "UTC"}
        end_data = {"dateTime": end_utc.isoformat(), "timeZone": "UTC"}
    else:
        start_data = {"date": start.isoformat()}
        end_data = {"date": end.isoformat()}
    if calendar == Calendar.google:
        event: dict[str, Any] = {
            "summary": title,
            "description": description,
            "start": start_data,
            "end": end_data,
        }
    else:
        assert isinstance(start, datetime) and isinstance(end, datetime)
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        event = {
            "subject": title,
            "body": {"contentType": "text", "content": description},
            "start": {"dateTime": start_utc.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": "UTC"},
            "end": {"dateTime": end_utc.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": "UTC"},
        }

    request = Request(
        endpoint,
        data=json.dumps(event).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise CalendarEntryError(
            f"{calendar} Calendar API returned HTTP {exc.code}: {body}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise CalendarEntryError(
            f"{calendar} Calendar request failed; an event may already have been created: {exc}"
        ) from exc

    try:
        created = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalendarEntryError(
            f"{calendar} Calendar API returned invalid JSON; an event may already have been created"
        ) from exc
    if not isinstance(created, dict) or not isinstance(created.get("id"), str) or not created["id"]:
        raise CalendarEntryError(
            f"{calendar} Calendar API returned no event ID; an event may already have been created"
        )
    return created


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
SYSTEM_PROMPT += "\n\n" + CALENDAR_SYSTEM_PROMPT

LOG_DIR = Path(__file__).resolve().parent / "logs"
OUT_DIR = Path("out")

OPENAI_MODEL = "gpt-5.6"
ANTHROPIC_MODEL = "claude-sonnet-5-5"

OPENAI_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"

# Room for the summary, reply, extracted lists and calendar proposals.
ANTHROPIC_MAX_TOKENS = 4096

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
# 3. PROVIDERS  (both return a validated analysis, not a raw dict)
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


class RequestThrottle:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.lock = asyncio.Lock()
        self.next_start = 0.0

    async def wait(self) -> None:
        async with self.lock:
            delay = self.next_start - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = time.monotonic() + self.interval


_request_throttle: ContextVar[RequestThrottle | None] = ContextVar("request_throttle", default=None)


async def _with_retry(
    provider: str, call: Callable[[], Awaitable[T]],
    close: Callable[[], Awaitable[None]] | None = None,
) -> T:
    """Await call(), retrying transient errors; the last error is re-raised."""
    try:
        async for attempt in _retrying(provider):
            with attempt:
                throttle = _request_throttle.get()
                if throttle is not None:
                    await throttle.wait()
                return await call()
        raise AssertionError("retry loop ended without a result")
    finally:
        if close is not None:
            await close()


def _require_key(*env_names: str) -> None:
    """Fail fast, before any request, if none of the key variables is set."""
    if not any(os.getenv(name) for name in env_names):
        raise MissingApiKey(
            f"{env_names[0]} is not set - export it in your environment or add it to .env"
        )


def _refusal_text(response: ParsedResponse[TriageAnalysis]) -> str | None:
    for output in response.output:
        if output.type != "message":
            continue

        for part in output.content:
            if part.type == "refusal":
                return part.refusal

    return None

async def triage_openai(text: str) -> TriageAnalysis:
    """Call OpenAI with Structured Outputs; the application supplies the result ID."""
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
            text_format=TriageAnalysis
        ),
        close=client.close,
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


async def triage_anthropic(text: str) -> TriageAnalysis:
    """Call Anthropic and return a validated TriageAnalysis.
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
            output_format=TriageAnalysis
        ),
        close=client.close,
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


async def create_events(
    result: TriageAnalysis, calendar: Calendar | str = Calendar.google,
) -> list[CreatedCalendarEvent]:
    """Create the first triage's proposals without any additional LLM requests."""
    try:
        calendar = Calendar(calendar)
    except ValueError as exc:
        raise CalendarToolError(f"Unknown calendar backend: {calendar}") from exc
    if calendar not in (Calendar.google, Calendar.me):
        raise CalendarToolError("Automatic event creation supports only Google Calendar or Me Calendar")
    try:
        validated = TriageAnalysis.model_validate(result.model_dump())
    except ValidationError as exc:
        raise CalendarToolError("Invalid calendar proposals in the triage result; no events created") from exc
    if calendar == Calendar.me:
        try:
            for event in validated.proposed_events:
                build_me_event_body(event.title, event.start, event.end, event.description)
        except MeCalendarError as exc:
            raise CalendarToolError(f"Invalid Me Calendar proposal; no events created: {exc}") from exc
    created: list[CreatedCalendarEvent] = []
    seen: set[str] = set()
    for event in validated.proposed_events:
        key = event.model_dump_json()
        if key in seen:
            continue
        try:
            response = await asyncio.to_thread(
                create_calendar_entry, calendar, event.title, event.start, event.end, event.description
            )
            entry = CreatedCalendarEvent(
                **event.model_dump(), id=response["id"], url=response.get("htmlLink")
            )
        except (CalendarEntryError, ValidationError) as exc:
            if isinstance(exc, CalendarEntryError) and not exc.event_may_exist:
                status = "The current event was not created. "
                guidance = (
                    "Previously created events remain; check your calendar before rerunning."
                    if created else "Correct the reported error before retrying."
                )
            else:
                status = "The current event may also have been created. "
                guidance = "Check your calendar before rerunning."
            raise CalendarToolError(
                f"Calendar creation failed after {len(created)} confirmed event(s): {exc}. "
                f"{status}{guidance}"
            ) from exc
        created.append(entry)
        seen.add(key)
        logger.info("Created calendar event {id}: {title}", id=entry.id, title=entry.title)
    return created


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

    parts: list = [
        header, Text(f"Input: {result.id}", style="dim"),
        Padding(Text(result.summary, style="italic"), (1, 0, 0, 0)),
    ]

    table = _extracted_table(result.extracted)
    if table is not None:
        parts.append(Padding(table, (1, 0, 0, 0)))

    if result.proposed_events:
        proposals = Table(title="Proposed calendar events", box=box.SIMPLE, expand=True)
        proposals.add_column("Title")
        proposals.add_column("Start")
        proposals.add_column("End (exclusive)")
        for event in result.proposed_events:
            proposals.add_row(Text(event.title), event.start.isoformat(), event.end.isoformat())
        parts.append(Padding(proposals, (1, 0, 0, 0)))

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


@dataclass
class BatchItem:
    result: TriageResult
    calendar_events: list[CreatedCalendarEvent] | None = None


@dataclass
class BatchError:
    id: str
    error: str


@dataclass
class BatchReport:
    results: list[BatchItem]
    errors: list[BatchError]


def result_data(
    result: TriageResult, calendar_events: list[CreatedCalendarEvent] | None = None,
) -> dict[str, Any]:
    data = result.model_dump(mode="json")
    if calendar_events is not None:
        data["calendar_events"] = [event.model_dump(mode="json") for event in calendar_events]
    return data


def output_json(
    result: TriageResult | BatchReport,
    calendar_events: list[CreatedCalendarEvent] | None = None,
) -> str:
    if isinstance(result, BatchReport):
        data = {
            "results": [result_data(item.result, item.calendar_events) for item in result.results],
            "errors": [{"id": error.id, "error": error.error} for error in result.errors],
        }
    else:
        data = result_data(result, calendar_events)
    return json.dumps(data, ensure_ascii=False, indent=2)


def write_output(
    result: TriageResult | BatchReport, path: Path,
    calendar_events: list[CreatedCalendarEvent] | None = None,
) -> None:
    """Save the result as JSON, creating parent folders as needed."""
    log = logger.opt(colors=True)
    path = path.resolve()
    log.info("Writing result to <cyan>{path}</cyan>", path=path)

    if not path.parent.exists():
        log.debug("Creating folder <cyan>{folder}</cyan>", folder=path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        log.debug("Overwriting existing file <cyan>{path}</cyan>", path=path)

    data = output_json(result, calendar_events) + "\n"
    path.write_text(data, encoding="utf-8")
    log.info("Wrote {size:,} bytes to <cyan>{path}</cyan>", size=len(data.encode("utf-8")), path=path)


class CliOptions(argparse.Namespace):
    path: str | None
    batch: str | None
    jobs: int
    request_interval: float
    provider: str
    calendar: Calendar
    create_events: bool
    json: bool
    out: str | None


class CliError(Exception):
    """An expected CLI failure that should be reported without a traceback."""


@dataclass(frozen=True)
class Provider:
    triage: Callable[[str], Awaitable[TriageAnalysis]]
    name: str
    model: str
    key_env: str


def positive_jobs(value: str) -> int:
    try:
        jobs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if jobs < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return jobs


def positive_interval(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds") from exc
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return interval


def parse_arguments() -> tuple[argparse.ArgumentParser, CliOptions]:
    parser = argparse.ArgumentParser(description="LLM inbox triage")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("path", nargs="?", help="Text file to triage (default: stdin)")
    input_group.add_argument(
        "--batch",
        metavar="DIR",
        help="Triage every .txt file directly in DIR (cannot be used with path)",
    )
    parser.add_argument(
        "--jobs", type=positive_jobs,
        help="Maximum concurrent batch requests (default: 3; requires --batch)",
    )
    parser.add_argument(
        "--request-interval", type=positive_interval, metavar="SECONDS",
        help="Minimum time between batch AI request starts, including retries (default: 1)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Which LLM provider to use",
    )
    parser.add_argument(
        "--calendar",
        type=Calendar,
        choices=[Calendar.google, Calendar.me],
        default=Calendar.google,
        help="Calendar backend for --create-events (default: google)",
    )
    parser.add_argument(
        "--create-events",
        action="store_true",
        help="Authorize model-selected calendar writes without confirmation (Google or Me)",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON only")
    parser.add_argument(
        "-o",
        "--out",
        nargs="?",
        const="",
        metavar="PATH",
        help="Save JSON to PATH (--json required except in batch; default: out/<input name>.out.json)",
    )
    args = CliOptions()
    parser.parse_args(namespace=args)
    if args.batch is None and (args.jobs is not None or args.request_interval is not None):
        parser.error("--jobs and --request-interval require --batch")
    args.jobs = args.jobs if args.jobs is not None else 3
    args.request_interval = args.request_interval if args.request_interval is not None else 1.0
    if args.out is not None and not args.json and args.batch is None:
        parser.error("--out requires --json")
    # "--out sample.txt" would take the input file as the output path and
    # overwrite it, so insist on a .json output name.
    if args.out and Path(args.out).suffix.lower() != ".json":
        parser.error(f"--out path must end in .json, got {args.out!r} (put the input file before --out)")
    return parser, args


def configure_logging(json_output: bool) -> None:
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
        level="ERROR" if json_output else "WARNING",
        format="<level>{level: <8}</level> | <level>{message}</level>",
    )


def read_cli_input(parser: argparse.ArgumentParser, path: str | None) -> str:
    source = path or "stdin"
    try:
        text = read_input(path)
    except FileNotFoundError:
        parser.error(f"File not found: {path}")
    except UnicodeDecodeError:
        parser.error(f"Cannot read {source}: not UTF-8 text")
    except OSError as exc:  # a directory, no permission, ...
        # Windows reports a directory as "Permission denied", so name it ourselves.
        reason = "is a directory" if Path(source).is_dir() else exc.strerror or exc
        parser.error(f"Cannot read {source}: {reason}")
    if not text.strip():
        raise CliError(f"Nothing to triage: {source} is empty")
    return text


def select_provider(name: str) -> Provider:
    if name == "openai":
        return Provider(triage_openai, "OpenAI", OPENAI_MODEL, OPENAI_KEY_ENV)
    if name == "anthropic":
        return Provider(triage_anthropic, "Anthropic", ANTHROPIC_MODEL, ANTHROPIC_KEY_ENV)
    raise CliError(f"Unknown LLM provider: {name}")


async def analyze_message(text: str, provider: Provider) -> TriageAnalysis:
    try:
        return await provider.triage(text)
    except (openai.AuthenticationError, anthropic.AuthenticationError) as exc:
        raise CliError(
            f"{provider.name} rejected the API key - check {provider.key_env}: {_describe_error(exc)}"
        ) from exc
    except (openai.APIError, anthropic.APIError) as exc:
        if _is_transient(exc):
            message = (
                f"{provider.name} still failing after {MAX_ATTEMPTS} attempts, "
                f"giving up: {_describe_error(exc)}"
            )
        else:
            message = f"{provider.name} request failed: {_describe_error(exc)}"
        raise CliError(message) from exc
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.debug("Malformed output details: {error}", error=exc)
        raise CliError(
            f"{provider.name} returned malformed output that does not match TriageResult"
        ) from exc


def identified_result(analysis: TriageAnalysis, source: str | None) -> TriageResult:
    return TriageResult(
        **analysis.model_dump(exclude={"id"}),
        id=Path(source).name if source is not None else "***stdin***",
    )


def run_triage(text: str, provider: Provider, console: Console) -> TriageAnalysis:
    # Spinner and timing go to stderr, and are silenced entirely with --json
    # so the output can be part of a pipeline.
    status_text = (
        f"[bold]📨 Triaging message[/bold] ({len(text):,} chars) · "
        f"waiting for [cyan]{provider.model}[/cyan] on {provider.name}…"
    )
    started = time.perf_counter()
    with console.status(status_text, spinner="dots"):
        result = asyncio.run(analyze_message(text, provider))
    elapsed = time.perf_counter() - started
    logger.info("Got response from {model} in {elapsed:.1f}s", model=provider.model, elapsed=elapsed)
    console.print(
        f"[green]✓[/green] Response from [cyan]{provider.model}[/cyan] in {elapsed:.1f}s",
        highlight=False,
    )
    return result


def run_calendar_creation(
    result: TriageResult, calendar: Calendar, console: Console,
) -> list[CreatedCalendarEvent]:
    with console.status("Creating calendar events from triage proposals...", spinner="dots"):
        calendar_events = asyncio.run(create_events(result, calendar))
    console.print(f"Created {len(calendar_events)} {calendar.label} event(s).", highlight=False)
    for event in calendar_events:
        console.print(Text(f"{event.title}: {event.start.isoformat()} (ID: {event.id})"))
    return calendar_events


def save_cli_output(
    args: CliOptions, result: TriageResult,
    calendar_events: list[CreatedCalendarEvent] | None,
) -> None:
    if args.out is not None:
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = default_out_path(args.path)
            logger.opt(colors=True).debug(
                "No --out path given, using default <cyan>{path}</cyan>", path=out_path
            )
        try:
            write_output(result, out_path, calendar_events)
        except OSError as exc:
            message = f"Cannot write {out_path}: {exc}"
            if calendar_events:
                message += (
                    f"; {len(calendar_events)} calendar event(s) were already created; "
                    "check your calendar before rerunning"
                )
            raise CliError(message) from exc


def print_cli_output(
    result: TriageResult, json_output: bool,
    calendar_events: list[CreatedCalendarEvent] | None,
) -> None:
    if json_output:
        logger.debug("Printing JSON result to stdout")
        print(output_json(result, calendar_events))
    else:
        logger.debug("Rendering result card")
        render(result)


def batch_files(directory: str) -> list[Path]:
    folder = Path(directory)
    if not folder.is_dir():
        raise CliError(f"Batch directory does not exist or is not a directory: {directory}")
    try:
        files = sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".txt"),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise CliError(f"Cannot list batch directory {directory}: {exc}") from exc
    if not files:
        raise CliError(f"No .txt files found in batch directory: {directory}")
    return files


async def process_batch(
    files: list[Path], args: CliOptions, provider: Provider, progress: Progress,
) -> BatchReport:
    outcomes: dict[Path, BatchItem | BatchError] = {}
    tasks = {path: progress.add_task(path.name, total=1, state="Queued", start=False) for path in files}
    pending = iter(files)
    calendar_lock = asyncio.Lock()

    async def worker() -> None:
        for path in pending:
            task = tasks[path]
            progress.start_task(task)
            progress.update(task, state="Reading")
            try:
                try:
                    text = read_input(str(path))
                except (OSError, UnicodeDecodeError) as exc:
                    raise CliError(f"Cannot read {path.name}: {exc}") from exc
                if not text.strip():
                    raise CliError(f"Nothing to triage: {path.name} is empty")
                progress.update(task, state="Waiting for AI")
                analysis = await analyze_message(text, provider)
                result = identified_result(analysis, str(path))
                events = None
                if args.create_events:
                    progress.update(task, state="Creating calendar events")
                    # OAuth and calendar writes are serialized, not retried.
                    async with calendar_lock:
                        events = await create_events(result, args.calendar)
                outcomes[path] = BatchItem(result, events)
                progress.update(task, completed=1, state=f"Done ({result.category.value})")
            except (CliError, TriageRefusal, MissingApiKey, CalendarToolError, CalendarEntryError) as exc:
                outcomes[path] = BatchError(path.name, str(exc))
                logger.error("{file}: {error}", file=path.name, error=exc)
                progress.update(task, completed=1, state="Failed")
            finally:
                progress.stop_task(task)

    token = _request_throttle.set(RequestThrottle(args.request_interval))
    try:
        await asyncio.gather(*(worker() for _ in range(min(args.jobs, len(files)))))
    finally:
        _request_throttle.reset(token)
    ordered = [outcomes[path] for path in files]
    return BatchReport(
        [outcome for outcome in ordered if isinstance(outcome, BatchItem)],
        [outcome for outcome in ordered if isinstance(outcome, BatchError)],
    )


def batch_signal_counts(results: list[TriageResult]) -> tuple[int, int, int]:
    return (
        sum(bool(result.extracted.dates or result.extracted.deadlines) for result in results),
        sum(bool(result.extracted.amounts) for result in results),
        sum(bool(result.suggested_reply and result.suggested_reply.strip()) for result in results),
    )


def print_batch_summary(report: BatchReport, console: Console) -> None:
    results = [item.result for item in report.results]
    succeeded, failed = len(results), len(report.errors)
    total = succeeded + failed
    table = Table(
        title="[bold cyan]Batch summary[/bold cyan]", title_justify="left",
        box=box.ROUNDED, border_style="cyan", header_style="bold cyan",
        caption=(
            f"{succeeded} successful | {failed} failed | {total} {'file' if total == 1 else 'files'}\n"
            "Signals count successful messages, not individual values."
        ),
        caption_justify="left",
    )
    table.add_column("Category")
    table.add_column("Count", justify="right")
    table.add_column("Dates", justify="right", style="cyan")
    table.add_column("Amounts", justify="right", style="yellow")
    table.add_column("Reply needed", justify="right", style="green")
    for category in Category:
        matching = [result for result in results if result.category == category]
        emoji, colour = CATEGORY_STYLE[category]
        try:
            emoji.encode(console.encoding)
        except UnicodeEncodeError:
            label = category.value
        else:
            label = f"{emoji} {category.value}"
        table.add_row(
            Text(label, style=colour), str(len(matching)),
            *(str(count) for count in batch_signal_counts(matching)),
            style="" if matching else "dim",
        )
    table.add_section()
    table.add_row(
        "Total", str(succeeded), *(str(count) for count in batch_signal_counts(results)),
        style="bold",
    )
    table.add_row("Failed", str(failed), "-", "-", "-", style="bold red" if failed else "dim")
    console.print(table)


def run_batch(args: CliOptions, console: Console) -> int:
    assert args.batch is not None
    files = batch_files(args.batch)
    with Progress(
        SpinnerColumn(finished_text=""), TextColumn("{task.description}", markup=False),
        TextColumn("{task.fields[state]}"), TimeElapsedColumn(), console=console,
    ) as progress:
        report = asyncio.run(process_batch(files, args, select_provider(args.provider), progress))
    print_batch_summary(report, console)
    out_path = Path(args.out) if args.out else OUT_DIR / f"{Path(args.batch).resolve().name}.out.json"
    try:
        write_output(report, out_path)
    except OSError as exc:
        if args.json:
            print(output_json(report))
        guidance = "; check your calendar before rerunning" if args.create_events else ""
        raise CliError(f"Cannot write {out_path}: {exc}{guidance}") from exc
    if args.json:
        print(output_json(report))
    return 1 if report.errors else 0


def run_single(parser: argparse.ArgumentParser, args: CliOptions, console: Console) -> None:
    text = read_cli_input(parser, args.path)
    result = identified_result(run_triage(text, select_provider(args.provider), console), args.path)
    calendar_events = (
        run_calendar_creation(result, args.calendar, console) if args.create_events else None
    )
    save_cli_output(args, result, calendar_events)
    print_cli_output(result, args.json, calendar_events)


def main() -> int:
    parser, args = parse_arguments()
    load_dotenv()
    configure_logging(args.json)
    console = Console(stderr=True, quiet=args.json and args.batch is None)
    try:
        if args.batch is not None:
            return run_batch(args, console)
        run_single(parser, args, console)
    except KeyboardInterrupt:
        (logger.error if args.batch is not None else logger.warning)(
            "Interrupted; check your calendar before rerunning" if args.create_events else "Interrupted"
        )
        return 130
    except (CliError, TriageRefusal, MissingApiKey, CalendarToolError, CalendarEntryError) as exc:
        logger.error("{}", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
