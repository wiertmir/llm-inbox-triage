# 📨 LLM Inbox Triage

A small, practical LLM assistant that takes raw text (an email, a support ticket, a message) and returns a **structured analysis**: category, priority, one-line summary, a suggested reply draft, and extracted data (dates, amounts, names, deadlines) as clean JSON.

Built as **Project 1** on my path to becoming an LLM/GenAI Engineer — a focused way to learn LLM APIs, structured output, prompt engineering, and multi-provider code in one useful tool.

```bash
python triage.py sample.txt
python triage.py sample.txt --provider anthropic
```

### Batch processing (#7)

Use `--batch <dir>` to process every `.txt` file directly in a directory
(case-insensitive extension; subdirectories and other file types are skipped):

```powershell
python triage.py --batch .\messages
python triage.py --batch .\evals\samples --json
python triage.py --batch .\messages --provider anthropic --jobs 2 --request-interval 2
python triage.py --batch .\messages --json --out .\reports\messages.json
```

`--batch` requires a directory argument and cannot be combined with the positional
input `path`, in either order. Without `--batch`, file input works as before;
omitting both `path` and `--batch` reads stdin.

Batch processing defaults to **3 concurrent workers** and **at least 1 second
between AI request starts**, including retries. `--jobs` accepts a positive
integer and `--request-interval` a finite positive number of seconds; both
options require `--batch`. Transient errors retain the existing exponential
backoff and bounded `Retry-After` handling. These are request-rate controls,
not token quotas: tune them for your provider/model's limits. Each SDK client
is closed after its request/retry sequence.

All filenames have their own simultaneous status line (queued, waiting for AI,
done, or failed), followed by a counts-per-category summary on **stderr**.
These remain visible with `--json`; stdout contains only one JSON document.
For redirected/noninteractive stderr, Rich prints the final status rows.
Successful results and errors are ordered by filename, not completion time.

Batch mode always saves **one aggregate JSON**, by default to
`out/<directory-name>.out.json`; existing output is overwritten. `--out PATH`
overrides the destination and may be used without `--json` in batch mode.
`--json` additionally prints the same document to stdout. This intentionally
replaces issue #7's one-JSON-per-input output:

```json
{
  "results": [
    {
      "id": "message.txt",
      "category": "question",
      "priority": 2,
      "summary": "A question about exporting records.",
      "suggested_reply": null,
      "extracted": {"dates": [], "amounts": [], "names": [], "deadlines": []},
      "proposed_events": []
    }
  ],
  "errors": [
    {"id": "unreadable.txt", "error": "Cannot read unreadable.txt: Access denied"}
  ]
}
```

An unreadable/empty message, API failure, or calendar failure is recorded in
`errors` and does not stop other files. Any per-file failure makes the command
exit with code **1**; all-success batches exit **0**. Invalid or empty directories
fail explicitly before any requests. An output-file write failure also exits 1;
with `--json`, the aggregate remains available on stdout. Interruption exits 130
and does not write a partial aggregate.

`--create-events` also works in batch mode. AI analysis is concurrent but calendar
authorization/writes are serialized. Each successful result includes its own
`calendar_events` list when enabled. Calendar writes are not retried; an error
may leave events already created. Check your calendar before rerunning. Deduplication
is per input message, not across messages or separate runs.

### Result IDs

Every exported `TriageResult` now requires `id: str`. For file input this is the
filename, **including its extension**, not a full path. For stdin it is the
literal `***stdin***`. IDs are assigned by the application, never by the model.
The provider functions return a validated `TriageAnalysis`; the CLI adds the
source ID to create the final `TriageResult`. Callers constructing or loading
older `TriageResult` objects must supply the source ID themselves.

`create_calendar_entry(calendar, title, start, end, description="", access_token=None)`
is also available for callers that want to create an event in Google, Me, or
Hotmail/Outlook Calendar. Use the `Calendar` enum (`Calendar.google`, `Calendar.me`,
or `Calendar.hotmail`); existing string backend names remain accepted for
compatibility. Supply timezone-aware `datetime` values (or `date`
values for Google/Me all-day events, with an exclusive end date) and an OAuth
access token, either directly or through `GOOGLE_CALENDAR_ACCESS_TOKEN` or
`HOTMAIL_CALENDAR_ACCESS_TOKEN`. Google can also obtain and refresh tokens
automatically as described below. The CLI calendar backend defaults to Google;
Google and Me are supported by the CLI automatic-creation workflow.

### Model-selected calendar events (#6)

```powershell
python triage.py sample.txt --create-events
python triage.py sample.txt --provider anthropic --create-events --calendar google
python triage.py sample.txt --create-events --json --out
python triage.py sample.txt --create-events --calendar me
```

`--create-events` authorizes calendar writes **without confirmation**. Without
this flag, triage behaves as before and never requests calendar authorization.
`--calendar` defaults to `google`; specifying the backend alone does not enable
writes.

The first structured triage includes a `proposed_events` list. The model selects
actionable meetings, appointments, and deadlines and supplies their `title`,
`start`, `end`, and `description`, or returns an empty list when none are appropriate.
These proposals are returned even without `--create-events`, so you can inspect
them in JSON or the terminal card before opting into writes. Dates without a complete time range and timezone
are represented as all-day events, with an exclusive end date.

With `--create-events`, a shared provider-independent `create_events(result)`
function validates and creates the first triage's proposals directly. Start dates
must belong to the extracted dates/deadlines, and the entire batch is validated
before any writes. The raw message is not analyzed again: there are **no additional
LLM selection, tool-calling, or acknowledgement requests**.

With `--create-events --json`, stdout and `--out` include an additional
`calendar_events` list containing each created event's `id`, `title`, `start`,
`end`, `description`, and `url`. Without `--create-events`, the JSON shape is
unchanged apart from the new `proposed_events` field. No proposals means no
calendar authorization or calendar requests. Older saved triage results without
`proposed_events` default to an empty list; when loading them as `TriageResult`,
supply the newly required source `id` as described above.

Identical proposals are deduplicated within a run, but separate runs
can create duplicates. Calendar writes are not automatically retried because
a lost response may mean the event was already created. If a run fails after
creating events, inspect your calendar before rerunning.

### Google Calendar authorization

1. Enable the Google Calendar API in your Google Cloud project and configure
   the OAuth consent screen. If the app is in testing, add your account as a
   test user.
2. Create a **Desktop app** OAuth client and download its credentials JSON.
   Keep this file outside the repository.
3. Set its path in PowerShell (or in your gitignored `.env`):

   ```powershell
   $env:GOOGLE_CALENDAR_CLIENT_SECRETS_FILE = 'C:\path\to\client_secret.json'
   ```

4. Call the function without `access_token`:

   ```python
   from datetime import datetime, timedelta, timezone
   from dotenv import load_dotenv
   from triage import Calendar, create_calendar_entry

   load_dotenv()  # Needed when calling directly and using .env
   start = datetime.now(timezone.utc) + timedelta(days=1)
   event = create_calendar_entry(
       Calendar.google, "Planning meeting", start, start + timedelta(hours=1)
   )
   print(event["id"])
   ```

The first call opens a browser for sign-in and consent to the
`https://www.googleapis.com/auth/calendar.events` scope. Authorization has a
three-minute timeout. Tokens are stored in Windows Credential Manager, macOS
Keychain, or Linux Secret Service, **not** in a plaintext file. The OS credential
store must be available/unlocked; there is no plaintext fallback.

Later calls reuse a valid token or refresh it without opening the browser.
An explicit token or `GOOGLE_CALENDAR_ACCESS_TOKEN` takes precedence over OAuth;
remove that override to use automatic refresh.

To authorize without creating an event, call
`get_google_calendar_access_token()` from `google_calendar_auth`. To replace
revoked credentials or switch Google accounts, call
`get_google_calendar_access_token(reauthorize=True)`. The cache is per OAuth
client, with one authorized Google account per client. Failed sign-in or refresh
raises a descriptive error instead of silently retrying in a browser.

Google may expire refresh tokens (including after seven days for external
testing-mode apps using calendar scopes), or users may revoke access; those
cases require sign-in again. Hotmail/Outlook automatic OAuth is not implemented:
continue supplying its access token.

### Me Calendar authorization

The Me integration follows the provided desktop example: browser authorization
code flow with PKCE, a loopback `/callback`, and the `/calendar/v1` REST API.
Set the service address and CA certificate path in PowerShell or your `.env`:

```powershell
$env:ME_BASE = 'https://pop-os.local'
$env:ME_CA_FILE = 'C:\path\to\me-calendar-example\caddy-root.crt'
python triage.py sample.txt --create-events --calendar me
```

The certificate stays outside the repository. HTTPS certificate and hostname
verification remain enabled. Without `ME_CA_FILE`, the service must have a
certificate trusted by your system. `ME_CLIENT_ID` defaults to `me-desktop`;
the auth service must register that client with the loopback redirect
`http://127.0.0.1/callback`. Sign-in has a five-minute timeout. The loopback
server handles connections concurrently, so Edge/Chrome preconnections that
send no data cannot block the actual OAuth callback. Idle connections time out
after ten seconds, and the listener shuts down on success, timeout, or failure.

Access tokens are cached in the same native OS credential store as Google
credentials. Me's example documents a 15-minute token lifetime; an `expires_in`
value returned by the service takes precedence. The example provides no refresh
flow, so an expired token requires a new browser sign-in. `ME_ACCESS_TOKEN`
is an optional override and disables automatic sign-in while it is set.
If authorization is revoked before the cached token expires, use
`get_me_calendar_access_token(MeConfig.from_environment(), reauthorize=True)`
from `me_calendar` to replace it. The browser must also trust Me's local CA
certificate for the sign-in page to load without a certificate warning.
For Windows, the updated example includes `TRUST-CERTIFICATE-WINDOWS.md` with
browser trust instructions. Verify the root certificate's thumbprint against
that guide before importing it into your Current User Trusted Root store.
Trusting a root CA grants it authority to sign certificates for any website;
only import the verified root from your own server. The triage script never
installs certificates or changes system trust settings automatically.

By default, the first calendar returned by Me is used, matching the example
(listing calendars can initialize its "Personal" calendar). Set
`ME_CALENDAR_ID` to target a specific calendar and skip that listing.
Payloads follow Me's live `EventInput` schema at `/calendar/openapi.json` and
the companion `EVENT-BODY-RULES.md`. Timed events are converted to UTC
wall-clock values with seconds only (no fractional seconds, offset, or `Z`)
and `tz="UTC"`. All-day events use date strings and an exclusive end date,
and **omit `tz` entirely**. Descriptions are sent to Me. Reminder settings
match the example (`[15]`).

Me-specific limits are checked for the entire proposal batch before any writes:
summaries at most 500 characters, descriptions at most 10,000 characters, and
years 1900 through 2200. Events must still end after they start when serialized
at whole-second precision.

HTTP errors include the service's `ErrorBody.message`. Me documents that HTTP
422 stores nothing: the current event is reported as not created, rather than
possibly duplicated. Earlier successful events in the same batch still exist.
Network failures and invalid responses after a write remain ambiguous; check
your calendar before rerunning those requests.

For direct Python calls:

```python
from triage import Calendar, create_calendar_entry

event = create_calendar_entry(Calendar.me, "Planning meeting", start, end)
```

## ✨ What it does

Given raw text, it returns:

| Field | Description |
|-------|-------------|
| `id` | input filename including extension, or `***stdin***` |
| `category` | e.g. urgent / invoice / spam / question / ignore |
| `priority` | 1 (low) – 5 (critical) |
| `summary` | one-sentence summary |
| `suggested_reply` | a draft response |
| `extracted` | structured data: dates, amounts, names, deadlines |
| `proposed_events` | calendar-ready proposals selected during the same triage |

## 🎯 Why this project

It touches everything required at the LLM fundamentals stage:

- **LLM APIs** — OpenAI (Responses API) and Anthropic
- **Structured Outputs** — a Pydantic schema the API *guarantees*, not "please return JSON" + `json.loads` and a prayer
- **Actions** — optionally create calendar entries from the first triage's structured proposals
- **Prompt engineering** — a short system prompt describing the *task*, while the schema enforces the *shape*
- **Evals** — a mini eval harness that measures categorization accuracy (the 2026 differentiator)
- **Production hygiene** — error handling, retries, refusal handling, cost awareness
- **Multi-provider** — same logic runs on OpenAI *and* Anthropic

It's also the seed for later projects: add RAG (Project 2) so it knows the context of past mail, then turn it into an agent (Project 3).

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/wiertmir/llm-inbox-triage.git
cd llm-inbox-triage

# 2. Install
pip install -r requirements.txt

# 3. Set your API key(s) — never hardcode them; .env is gitignored
cp .env.example .env    # then fill in the keys (loaded automatically)
# ...or export them; variables already set in the environment win over .env
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
python triage.py sample.txt
```

## Evaluation samples (issue #8 preparation)

[evals/samples](evals/samples) contains 20 synthetic messages, separate from
the original `sample.txt`. [evals/expected.json](evals/expected.json) labels
each message with its expected category and all four extracted fields:
`dates`, `amounts`, `names`, and `deadlines`.

The dataset covers every category: three samples each for `urgent`, `invoice`,
`spam`, `question`, `newsletter`, and `ignore`, plus two for `other`. It includes
empty fields, multiple amounts, JPY/USD/EUR amounts, the default JPY currency,
deadlines, and a relative deadline anchored to an explicit sent date. All
messages and entities are fictional; no real inbox data is included.

The manifest has `schema_version: 1` and a `samples` array. Each entry contains
a unique `id`, a `file` path relative to the manifest, and an `expected` object.
For a future runner, compare categories exactly and extracted lists as unordered
sets of values, including empty lists (unexpected extra values are mismatches).
Dates use ISO `YYYY-MM-DD`; money values pair a numeric amount with an ISO currency
code. Names retain their spelling and capitalization. Summaries, priorities,
reply drafts, and calendar proposals are intentionally not labeled or scored.

You can try a single sample with:

```powershell
python triage.py .\evals\samples\04_invoice_jpy.txt --json
```

These are hand-authored expected labels, not recorded model outputs or measured
accuracy. The evaluation runner and accuracy reporting from issue #8 are not
implemented yet. Batch results can be matched to labels using each manifest
entry's input filename, rather than its extensionless fixture `id`. Offline fixture
integrity checks run with `python -m pytest tests\test_evals.py` without API calls.

## 🛠️ Roadmap

See the [Issues](https://github.com/wiertmir/llm-inbox-triage/issues) tab. MVP first, enhancements later.

**MVP:**
- [ ] Read text from file / stdin
- [ ] **Structured Outputs**: Pydantic `TriageAnalysis` via `responses.parse` (OpenAI) — schema guaranteed, no `json.loads`
- [ ] Works on OpenAI and Anthropic (`--provider` flag)
- [ ] Pretty terminal output + JSON export
- [ ] **Mini-eval**: 10–20 sample messages with expected output, measure categorization accuracy
- [ ] README with example + lessons learned + eval results

**Later:**
- [x] Model-selected Google/Me Calendar events from a single triage with `--create-events`
- [x] Batch mode with throttled concurrency, per-file status, and aggregate JSON (#7)
- [ ] Refusal / safety handling surfaced cleanly

## 📝 License

MIT

---

*Part of my 6-month roadmap to LLM/GenAI Engineer. 🪶*
