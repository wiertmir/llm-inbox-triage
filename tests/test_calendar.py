import json
import sys
from datetime import date, datetime, timezone
from email.message import Message
from pathlib import Path
from io import BytesIO
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402


@pytest.mark.parametrize(
    ("calendar", "endpoint"),
    [
        ("google", "https://www.googleapis.com/calendar/v3/calendars/primary/events"),
        ("hotmail", "https://graph.microsoft.com/v1.0/me/events"),
    ],
)
def test_create_calendar_entry_posts_event(calendar, endpoint, monkeypatch):
    captured = {}
    expected_response = {"id": "event-123"}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(expected_response).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        captured["data"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(triage, "urlopen", fake_urlopen)
    start = datetime(2026, 10, 5, 9, tzinfo=timezone.utc)
    end = datetime(2026, 10, 5, 10, tzinfo=timezone.utc)

    result = triage.create_calendar_entry(
        calendar, "Planning", start, end, "Quarterly planning", "test-token"
    )

    assert result == expected_response
    assert captured["url"] == endpoint
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["timeout"] == 30
    if calendar == "google":
        assert captured["data"]["summary"] == "Planning"
        assert captured["data"]["description"] == "Quarterly planning"
        assert captured["data"]["start"]["dateTime"] == "2026-10-05T09:00:00+00:00"
    else:
        assert captured["data"]["subject"] == "Planning"
        assert captured["data"]["body"]["content"] == "Quarterly planning"
        assert captured["data"]["start"] == {
            "dateTime": "2026-10-05T09:00:00",
            "timeZone": "UTC",
        }


def test_create_calendar_entry_reads_provider_token_from_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", "environment-token")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"id":"event-123"}'

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(triage, "urlopen", fake_urlopen)

    triage.create_calendar_entry(
        "google",
        "Planning",
        datetime(2026, 10, 5, 9, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 10, tzinfo=timezone.utc),
    )

    assert captured["authorization"] == "Bearer environment-token"


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (datetime(2026, 10, 5, 9), datetime(2026, 10, 5, 10), "start must be timezone-aware"),
        (
            datetime(2026, 10, 5, 9, tzinfo=timezone.utc),
            datetime(2026, 10, 5, 10),
            "end must be timezone-aware",
        ),
        (
            datetime(2026, 10, 5, 10, tzinfo=timezone.utc),
            datetime(2026, 10, 5, 9, tzinfo=timezone.utc),
            "end must be later than start",
        ),
    ],
)
def test_create_calendar_entry_rejects_invalid_times(start, end, message):
    with pytest.raises(ValueError, match=message):
        triage.create_calendar_entry("google", "Planning", start, end, access_token="token")


def test_create_calendar_entry_requires_access_token(monkeypatch):
    monkeypatch.delenv("GOOGLE_CALENDAR_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CALENDAR_CLIENT_SECRETS_FILE", raising=False)
    with pytest.raises(triage.CalendarEntryError, match="GOOGLE_CALENDAR_CLIENT_SECRETS_FILE"):
        triage.create_calendar_entry(
            "google",
            "Planning",
            datetime(2026, 10, 5, 9, tzinfo=timezone.utc),
            datetime(2026, 10, 5, 10, tzinfo=timezone.utc),
        )


def test_create_calendar_entry_uses_google_oauth(monkeypatch):
    monkeypatch.delenv("GOOGLE_CALENDAR_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(triage, "get_google_calendar_access_token", lambda: "oauth-token")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"id":"oauth-event"}'

    def fake_urlopen(request, timeout):
        assert request.headers["Authorization"] == "Bearer oauth-token"
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr(triage, "urlopen", fake_urlopen)
    result = triage.create_calendar_entry(
        "google", "Planning",
        datetime(2026, 10, 5, 9, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 10, tzinfo=timezone.utc),
    )
    assert result["id"] == "oauth-event"


def test_google_token_override_does_not_start_oauth(monkeypatch):
    def unexpected_oauth():
        pytest.fail("OAuth must not run when an access token is supplied")

    monkeypatch.setattr(triage, "get_google_calendar_access_token", unexpected_oauth)
    test_create_calendar_entry_reads_provider_token_from_environment(monkeypatch)


def test_hotmail_does_not_start_google_oauth(monkeypatch):
    monkeypatch.delenv("HOTMAIL_CALENDAR_ACCESS_TOKEN", raising=False)

    def unexpected_oauth():
        pytest.fail("Hotmail must not use Google OAuth")

    monkeypatch.setattr(triage, "get_google_calendar_access_token", unexpected_oauth)
    with pytest.raises(triage.CalendarEntryError, match="HOTMAIL_CALENDAR_ACCESS_TOKEN"):
        triage.create_calendar_entry(
            "hotmail", "Planning",
            datetime(2026, 10, 5, 9, tzinfo=timezone.utc),
            datetime(2026, 10, 5, 10, tzinfo=timezone.utc),
        )


def test_create_calendar_entry_surfaces_api_error(monkeypatch):
    def fail_urlopen(request, timeout):
        assert timeout == 30
        raise HTTPError(request.full_url, 401, "Unauthorized", Message(), BytesIO(b"invalid token"))

    monkeypatch.setattr(triage, "urlopen", fail_urlopen)

    with pytest.raises(triage.CalendarEntryError, match="HTTP 401"):
        triage.create_calendar_entry(
            "google",
            "Planning",
            datetime(2026, 10, 5, 9, tzinfo=timezone.utc),
            datetime(2026, 10, 5, 10, tzinfo=timezone.utc),
            access_token="invalid-token",
        )


def test_google_all_day_event_uses_exclusive_end_date(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"id":"all-day"}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(triage, "urlopen", fake_urlopen)
    triage.create_calendar_entry(
        "google", "Deadline", date(2026, 10, 5), date(2026, 10, 6), access_token="test",
    )
    assert captured["start"] == {"date": "2026-10-05"}
    assert captured["end"] == {"date": "2026-10-06"}


@pytest.mark.parametrize("body", [b"[]", b'{"message":"No event ID"}', b"invalid JSON"])
def test_google_invalid_response_is_not_success(monkeypatch, body):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return body

    monkeypatch.setattr(triage, "urlopen", lambda request, timeout: FakeResponse())
    with pytest.raises(triage.CalendarEntryError):
        triage.create_calendar_entry(
            "google", "Deadline", date(2026, 10, 5), date(2026, 10, 6), access_token="test",
        )


def test_main_accepts_calendar_selector(monkeypatch, capsys):
    async def fake_triage(text):
        assert text == "message"
        return triage.TriageResult(
            id="***stdin***",
            category=triage.Category.other,
            priority=1,
            summary="Done",
            extracted=triage.Extracted(),
        )

    monkeypatch.setattr(triage, "triage_openai", fake_triage)
    monkeypatch.setattr(sys, "argv", ["triage.py", "--calendar", "google", "--json"])
    monkeypatch.setattr(triage, "read_input", lambda _: "message")

    assert triage.main() == 0
    assert '"summary": "Done"' in capsys.readouterr().out
