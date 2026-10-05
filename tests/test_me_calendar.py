"""Me tests use synthetic tokens, mocked REST calls, and a local OAuth callback."""

import base64
import hashlib
import http.client
import json
import socket
import ssl
import asyncio
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from email.message import Message
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from keyring.errors import PasswordSetError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import me_calendar as me  # noqa: E402
import triage  # noqa: E402


@pytest.fixture(autouse=True)
def clean_me_environment(monkeypatch):
    for name in ("ME_BASE", "ME_CLIENT_ID", "ME_CA_FILE", "ME_CALENDAR_ID", "ME_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config():
    return me.MeConfig("https://me.example", "me-desktop", ssl.create_default_context(), None)


@pytest.fixture
def callback_servers(monkeypatch):
    servers = []
    original = me.CallbackServer

    class TrackingServer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            servers.append(self)

        def serve_forever(self, poll_interval=0.5):
            self.serving_thread = threading.current_thread()
            return super().serve_forever(poll_interval=poll_interval)

    monkeypatch.setattr(me, "CallbackServer", TrackingServer)
    yield servers
    for server in servers:
        assert server.socket.fileno() == -1
        assert not server.serving_thread.is_alive()


@pytest.fixture
def store(monkeypatch):
    keyring = Mock()
    keyring.get_password.return_value = None
    keyring.sign_in = Mock(return_value={"access_token": "synthetic-token"})
    monkeypatch.setattr(me, "get_credential_store", lambda: keyring)
    monkeypatch.setattr(me, "_sign_in", keyring.sign_in)
    return keyring


def test_config_defaults_and_tls_verification():
    config = me.MeConfig.from_environment()
    assert config.base == "https://pop-os.local"
    assert config.client_id == "me-desktop"
    assert config.tls.verify_mode == ssl.CERT_REQUIRED
    assert config.tls.check_hostname is True


def test_config_loads_ca_without_disabling_verification(monkeypatch):
    context = Mock()
    monkeypatch.setattr(me.ssl, "create_default_context", lambda: context)
    monkeypatch.setenv("ME_CA_FILE", r"C:\external\caddy-root.crt")
    me.MeConfig.from_environment()
    context.load_verify_locations.assert_called_once_with(cafile=r"C:\external\caddy-root.crt")


def test_missing_certificate_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("ME_CA_FILE", str(tmp_path / "missing.crt"))
    with pytest.raises(me.MeCalendarError, match="ME_CA_FILE"):
        me.MeConfig.from_environment()


@pytest.mark.parametrize("base", ["http://me.example", "https://user:pass@me.example", "https://me.example?q=1"])
def test_invalid_service_address_is_rejected(monkeypatch, base):
    monkeypatch.setenv("ME_BASE", base)
    with pytest.raises(me.MeCalendarError, match="ME_BASE"):
        me.MeConfig.from_environment()


def test_sign_in_is_cached_in_native_store(config, store):
    assert me.get_me_calendar_access_token(config) == "synthetic-token"
    service, account, saved = store.set_password.call_args.args
    assert service == me.KEYRING_SERVICE
    assert account == "https://me.example|me-desktop"
    data = json.loads(saved)
    assert data["access_token"] == "synthetic-token"
    assert 895 < data["expires_at"] - time.time() <= 900


def test_valid_cached_token_needs_no_browser(config, store):
    store.get_password.return_value = json.dumps({
        "access_token": "cached-token", "expires_at": time.time() + 600,
    })
    assert me.get_me_calendar_access_token(config) == "cached-token"
    store.sign_in.assert_not_called()
    store.set_password.assert_not_called()


def test_expired_token_requires_new_sign_in(config, store):
    store.get_password.return_value = json.dumps({
        "access_token": "expired", "expires_at": time.time() - 1,
    })
    assert me.get_me_calendar_access_token(config) == "synthetic-token"
    store.sign_in.assert_called_once_with(config)


def test_token_response_lifetime_is_respected(config, store, monkeypatch):
    monkeypatch.setattr(me, "_sign_in", Mock(return_value={
        "access_token": "short-lived", "expires_in": 120,
    }))
    assert me.get_me_calendar_access_token(config) == "short-lived"
    assert 115 < json.loads(store.set_password.call_args.args[2])["expires_at"] - time.time() <= 120


def test_reauthorize_skips_cache(config, store):
    assert me.get_me_calendar_access_token(config, reauthorize=True) == "synthetic-token"
    store.get_password.assert_not_called()


def test_token_override_skips_browser_and_store(config, store, monkeypatch):
    monkeypatch.setenv("ME_ACCESS_TOKEN", "override")
    assert me.get_me_calendar_access_token(config) == "override"
    store.sign_in.assert_not_called()
    store.get_password.assert_not_called()


@pytest.mark.parametrize("cached", ["invalid", "[]", "{}", '{"access_token":"","expires_at":1}'])
def test_bad_cache_does_not_silently_sign_in(config, store, cached):
    store.get_password.return_value = cached
    with pytest.raises(me.MeCalendarError, match="Invalid cached"):
        me.get_me_calendar_access_token(config)
    store.sign_in.assert_not_called()


@pytest.mark.parametrize("operation", ["get_password", "set_password"])
def test_store_errors_are_explicit(config, store, operation):
    getattr(store, operation).side_effect = PasswordSetError("locked")
    with pytest.raises(me.MeCalendarError, match="OS credential store"):
        me.get_me_calendar_access_token(config)


@pytest.mark.parametrize("lifetime", [0, -1, True, "900", float("inf")])
def test_invalid_token_lifetime_is_not_cached(config, store, monkeypatch, lifetime):
    monkeypatch.setattr(me, "_sign_in", Mock(return_value={
        "access_token": "synthetic", "expires_in": lifetime,
    }))
    with pytest.raises(me.MeCalendarError, match="invalid token lifetime"):
        me.get_me_calendar_access_token(config)
    store.set_password.assert_not_called()


def test_pkce_callback_checks_state_then_exchanges_code(config, monkeypatch, callback_servers):
    threads, statuses, exchanges = [], [], []

    def open_browser(url):
        params = parse_qs(urlsplit(url).query)
        redirect = urlsplit(params["redirect_uri"][0])

        def callbacks():
            for state in ("incorrect-state", params["state"][0]):
                connection = http.client.HTTPConnection(redirect.hostname, redirect.port, timeout=3)
                try:
                    connection.request("GET", "/callback?" + urlencode({"state": state, "code": "fake-code"}))
                    response = connection.getresponse()
                    statuses.append(response.status)
                    response.read()
                finally:
                    connection.close()

        worker = threading.Thread(target=callbacks)
        threads.append(worker)
        worker.start()
        exchanges.append(params)
        return True

    def exchange(cfg, method, path, *, form):
        assert cfg is config
        assert (method, path) == ("POST", "/oauth/token")
        assert form["code"] == "fake-code"
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(form["code_verifier"].encode()).digest()
        ).rstrip(b"=").decode()
        assert exchanges[0]["code_challenge"] == [challenge]
        assert exchanges[0]["code_challenge_method"] == ["S256"]
        assert form["redirect_uri"] == exchanges[0]["redirect_uri"][0]
        assert form["client_id"] == "me-desktop"
        return {"access_token": "callback-token"}

    monkeypatch.setattr(me.webbrowser, "open", open_browser)
    monkeypatch.setattr(me, "_http_json", exchange)
    assert me._sign_in(config) == {"access_token": "callback-token"}
    for worker in threads:
        worker.join(timeout=3)
        assert not worker.is_alive()
    assert statuses == [400, 200]


def test_no_callback_times_out(config, monkeypatch, callback_servers):
    monkeypatch.setattr(me.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(me, "AUTH_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(me.MeCalendarError, match="timed out"):
        me._sign_in(config)


def test_idle_browser_connection_does_not_block_oauth_callback(config, monkeypatch, callback_servers):
    idle_connections, workers, errors = [], [], []
    monkeypatch.setattr(me, "AUTH_TIMEOUT_SECONDS", 1)

    def open_browser(url):
        params = parse_qs(urlsplit(url).query)
        redirect = urlsplit(params["redirect_uri"][0])
        idle_connections.append(socket.create_connection(
            (redirect.hostname, redirect.port), timeout=1,
        ))

        def callback():
            connection = http.client.HTTPConnection(redirect.hostname, redirect.port, timeout=1)
            try:
                connection.request("GET", "/callback?" + urlencode({
                    "state": params["state"][0], "code": "fake-code",
                }))
                response = connection.getresponse()
                assert response.status == 200
                response.read()
            except (OSError, AssertionError) as exc:
                errors.append(exc)
            finally:
                connection.close()

        worker = threading.Thread(target=callback)
        workers.append(worker)
        worker.start()
        return True

    monkeypatch.setattr(me.webbrowser, "open", open_browser)
    exchange = Mock(return_value={"access_token": "callback-token"})
    monkeypatch.setattr(me, "_http_json", exchange)
    started = time.monotonic()
    try:
        assert me._sign_in(config) == {"access_token": "callback-token"}
    finally:
        for connection in idle_connections:
            connection.close()
        for worker in workers:
            worker.join(timeout=2)
    assert time.monotonic() - started < 1.5
    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    exchange.assert_called_once()


def test_browser_failure_is_explicit(config, monkeypatch, callback_servers):
    monkeypatch.setattr(me.webbrowser, "open", lambda url: False)
    with pytest.raises(me.MeCalendarError, match="Could not open"):
        me._sign_in(config)


def test_duplicate_callbacks_cannot_replace_first_authorization(
    config, monkeypatch, callback_servers,
):
    statuses = []

    def open_browser(url):
        params = parse_qs(urlsplit(url).query)
        redirect = urlsplit(params["redirect_uri"][0])
        for code in ("first-code", "second-code"):
            connection = http.client.HTTPConnection(redirect.hostname, redirect.port, timeout=1)
            try:
                connection.request("GET", "/callback?" + urlencode({
                    "state": params["state"][0], "code": code,
                }))
                response = connection.getresponse()
                statuses.append(response.status)
                response.read()
            finally:
                connection.close()
        return True

    exchange = Mock(return_value={"access_token": "first-token"})
    monkeypatch.setattr(me.webbrowser, "open", open_browser)
    monkeypatch.setattr(me, "_http_json", exchange)
    assert me._sign_in(config) == {"access_token": "first-token"}
    assert statuses == [200, 409]
    exchange.assert_called_once()
    assert exchange.call_args.kwargs["form"]["code"] == "first-code"


def test_denied_sign_in_does_not_exchange_a_code(config, monkeypatch, callback_servers):
    def open_browser(url):
        params = parse_qs(urlsplit(url).query)
        redirect = urlsplit(params["redirect_uri"][0])
        connection = http.client.HTTPConnection(redirect.hostname, redirect.port, timeout=1)
        try:
            connection.request("GET", "/callback?" + urlencode({
                "state": params["state"][0], "error": "access_denied",
            }))
            response = connection.getresponse()
            assert response.status == 200
            response.read()
        finally:
            connection.close()
        return True

    exchange = Mock()
    monkeypatch.setattr(me.webbrowser, "open", open_browser)
    monkeypatch.setattr(me, "_http_json", exchange)
    with pytest.raises(me.MeCalendarError, match="denied"):
        me._sign_in(config)
    exchange.assert_not_called()


def test_auth_timeout_matches_updated_example():
    assert me.AUTH_TIMEOUT_SECONDS == 300
    assert me.CallbackServer.daemon_threads is True


def test_http_uses_verified_tls_bearer_token_and_timeout(config, monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"id":"new-event"}'

    def open_request(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    def opener(*handlers):
        captured["handlers"] = handlers
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(me, "build_opener", opener)
    assert me._http_json(config, "POST", "/calendar/v1/calendars/1/events",
                         token="test", body={"summary": "Meeting"}) == {"id": "new-event"}
    assert captured["request"].headers["Authorization"] == "Bearer test"
    assert json.loads(captured["request"].data) == {"summary": "Meeting"}
    assert captured["handlers"][0]._context is config.tls
    assert isinstance(captured["handlers"][1], me.NoRedirect)
    assert captured["timeout"] == 30


@pytest.mark.parametrize("error", [
    HTTPError("https://me.example", 401, "Unauthorized", Message(), None),
    URLError(ssl.SSLCertVerificationError("untrusted")),
    TimeoutError("timeout"),
])
def test_http_errors_are_explicit_and_not_retried(config, monkeypatch, error):
    mocked = Mock(side_effect=error)
    monkeypatch.setattr(me, "build_opener", lambda *args: SimpleNamespace(open=mocked))
    with pytest.raises(me.MeCalendarError):
        me._http_json(config, "GET", "/calendar/v1/calendars", token="fake")
    assert mocked.call_count == 1


@pytest.mark.parametrize("all_day", [False, True])
def test_me_event_payload_matches_example(config, monkeypatch, all_day):
    calls = []
    monkeypatch.setattr(me.MeConfig, "from_environment", classmethod(lambda cls: config))
    monkeypatch.setattr(me, "get_me_calendar_access_token", lambda cfg: "token")

    def api(cfg, method, path, **kwargs):
        calls.append((method, path, kwargs))
        return [{"id": "personal", "name": "Personal"}] if method == "GET" else {"id": 123}

    monkeypatch.setattr(me, "_http_json", api)
    start = date(2026, 10, 5) if all_day else datetime(
        2026, 10, 5, 11, 30, tzinfo=timezone(timedelta(hours=9))
    )
    end = date(2026, 10, 6) if all_day else start + timedelta(hours=1)
    result = me.create_me_calendar_entry("Planning", start, end, "Discuss plans")
    assert result["id"] == "123"
    assert calls[0][:2] == ("GET", "/calendar/v1/calendars")
    assert calls[1][:2] == ("POST", "/calendar/v1/calendars/personal/events")
    payload = calls[1][2]["body"]
    expected = {
        "summary": "Planning", "description": "Discuss plans", "all_day": all_day,
        "start": "2026-10-05" if all_day else "2026-10-05T02:30:00",
        "end": "2026-10-06" if all_day else "2026-10-05T03:30:00",
        "reminders": [15],
    }
    if not all_day:
        expected["tz"] = "UTC"
    assert payload == expected


def test_all_day_payload_forbids_timezone():
    body = me.build_me_event_body("Deadline", date(2026, 10, 5), date(2026, 10, 6))
    assert body["all_day"] is True
    assert body["start"] == "2026-10-05"
    assert body["end"] == "2026-10-06"
    assert "tz" not in body


def test_timed_payload_uses_whole_second_wall_clock_values():
    start = datetime(2026, 10, 5, 11, 30, 0, 123456, tzinfo=timezone(timedelta(hours=9)))
    body = me.build_me_event_body("Meeting", start, start + timedelta(hours=1), "Notes")
    assert body["start"] == "2026-10-05T02:30:00"
    assert body["end"] == "2026-10-05T03:30:00"
    assert body["tz"] == "UTC"
    assert body["description"] == "Notes"


def test_subsecond_event_must_remain_positive_after_serialization():
    start = datetime(2026, 10, 5, 11, 30, tzinfo=timezone.utc)
    with pytest.raises(me.MeCalendarError, match="whole-second precision"):
        me.build_me_event_body("Meeting", start, start + timedelta(microseconds=1))


def test_payload_accepts_exact_text_limits_and_boundary_years():
    body = me.build_me_event_body(
        "x" * 500, date(1900, 1, 1), date(2200, 12, 31), "y" * 10000,
    )
    assert len(body["summary"]) == 500
    assert len(body["description"]) == 10000


@pytest.mark.parametrize(
    ("title", "start", "end", "description", "message"),
    [
        ("x" * 501, date(2026, 10, 5), date(2026, 10, 6), "", "500 characters"),
        ("Title", date(2026, 10, 5), date(2026, 10, 6), "y" * 10001, "10000 characters"),
        ("Title", date(1899, 12, 31), date(1900, 1, 1), "", "1900 and 2200"),
        ("Title", date(2200, 12, 31), date(2201, 1, 1), "", "1900 and 2200"),
    ],
)
def test_invalid_me_payload_fails_before_authorization(
    monkeypatch, title, start, end, description, message,
):
    auth = Mock(side_effect=AssertionError("Must not authorize invalid events"))
    monkeypatch.setattr(me, "get_me_calendar_access_token", auth)
    with pytest.raises(me.MeCalendarError, match=message) as caught:
        me.create_me_calendar_entry(title, start, end, description)
    assert caught.value.event_may_exist is False
    auth.assert_not_called()


def test_me_limits_are_validated_for_entire_batch_before_writes(monkeypatch):
    write = Mock(side_effect=AssertionError("Must not write an invalid batch"))
    monkeypatch.setattr(triage, "create_calendar_entry", write)
    result = triage.TriageResult(
        id="sample.txt",
        category=triage.Category.other, priority=2, summary="Two deadlines",
        extracted=triage.Extracted(deadlines=[date(2026, 10, 5)]),
        proposed_events=[
            triage.CalendarEvent(
                title=title, start=date(2026, 10, 5), end=date(2026, 10, 6), description="",
            )
            for title in ("Valid deadline", "x" * 501)
        ],
    )
    with pytest.raises(triage.CalendarToolError, match="no events created"):
        asyncio.run(triage.create_events(result, triage.Calendar.me))
    write.assert_not_called()


def test_validation_response_preserves_message_and_confirms_no_event(config, monkeypatch):
    error = HTTPError(
        "https://me.example/calendar/v1/calendars/1/events", 422, "Unprocessable Entity",
        Message(), BytesIO(b'{"code":"validation","message":"tz is forbidden for all-day events"}'),
    )
    opener = Mock(side_effect=error)
    monkeypatch.setattr(me, "build_opener", lambda *args: SimpleNamespace(open=opener))
    with pytest.raises(me.MeCalendarError, match="tz is forbidden for all-day events") as caught:
        me._http_json(config, "POST", "/calendar/v1/calendars/1/events", token="test", body={})
    assert caught.value.event_may_exist is False
    assert opener.call_count == 1


@pytest.mark.parametrize("error", [
    URLError("Connection lost"),
    HTTPError("https://me.example", 500, "Internal error", Message(), BytesIO(b"")),
])
def test_uncertain_write_error_still_warns_about_existing_event(config, monkeypatch, error):
    monkeypatch.setattr(me, "build_opener", lambda *args: SimpleNamespace(open=Mock(side_effect=error)))
    with pytest.raises(me.MeCalendarError) as caught:
        me._http_json(config, "POST", "/calendar/v1/calendars/1/events", token="test", body={})
    assert caught.value.event_may_exist is True


@pytest.mark.parametrize("partial_success", [False, True])
def test_failed_creation_reports_422_as_not_created(monkeypatch, partial_success):
    rejection = me.MeCalendarError("HTTP 422: invalid event", event_may_exist=False)
    outcomes = [{"id": "first-event"}, rejection] if partial_success else [rejection]
    monkeypatch.setattr(triage, "create_me_calendar_entry", Mock(side_effect=outcomes))
    result = triage.TriageResult(
        id="sample.txt",
        category=triage.Category.other, priority=2, summary="Deadlines",
        extracted=triage.Extracted(deadlines=[date(2026, 10, 5)]),
        proposed_events=[
            triage.CalendarEvent(
                title=title, start=date(2026, 10, 5), end=date(2026, 10, 6), description="Notes",
            )
            for title in (("First", "Second") if partial_success else ("First",))
        ],
    )
    with pytest.raises(triage.CalendarToolError) as caught:
        asyncio.run(triage.create_events(result, triage.Calendar.me))
    message = str(caught.value)
    assert "The current event was not created" in message
    assert "may also have been created" not in message
    if partial_success:
        assert "1 confirmed event" in message
        assert "Previously created events remain" in message
    else:
        assert "0 confirmed event" in message


def test_selected_calendar_skips_listing_and_escapes_id(config, monkeypatch):
    config = me.MeConfig(config.base, config.client_id, config.tls, "my/calendar")
    mocked = Mock(return_value={"id": "created"})
    monkeypatch.setattr(me.MeConfig, "from_environment", classmethod(lambda cls: config))
    monkeypatch.setattr(me, "_http_json", mocked)
    me.create_me_calendar_entry("Title", date(2026, 10, 5), date(2026, 10, 6), access_token="test")
    assert mocked.call_count == 1
    assert mocked.call_args.args[2] == "/calendar/v1/calendars/my%2Fcalendar/events"


@pytest.mark.parametrize("answer", [[], {}, [None], [{"name": "no id"}]])
def test_invalid_calendar_listing_does_not_write(config, monkeypatch, answer):
    mocked = Mock(return_value=answer)
    monkeypatch.setattr(me.MeConfig, "from_environment", classmethod(lambda cls: config))
    monkeypatch.setattr(me, "_http_json", mocked)
    with pytest.raises(me.MeCalendarError):
        me.create_me_calendar_entry("Title", date(2026, 10, 5), date(2026, 10, 6), access_token="test")
    assert mocked.call_count == 1


def test_me_dispatch_does_not_use_google_oauth(monkeypatch):
    mocked = Mock(return_value={"id": "me-event"})
    monkeypatch.setattr(triage, "create_me_calendar_entry", mocked)
    monkeypatch.setattr(triage, "get_google_calendar_access_token", Mock(side_effect=AssertionError))
    created = triage.create_calendar_entry(
        triage.Calendar.me, "Title", date(2026, 10, 5), date(2026, 10, 6), "Notes from first triage",
    )
    assert created["id"] == "me-event"
    mocked.assert_called_once_with(
        "Title", date(2026, 10, 5), date(2026, 10, 6), "Notes from first triage", access_token=None,
    )


def test_me_batch_reuses_token_and_exports_event_ids(config, store, monkeypatch):
    calls = []
    monkeypatch.setattr(me.MeConfig, "from_environment", classmethod(lambda cls: config))

    def api(cfg, method, path, **kwargs):
        assert kwargs["token"] == "synthetic-token"
        calls.append((method, path))
        return [{"id": "personal"}] if method == "GET" else {"id": f"event-{len(calls)}"}

    monkeypatch.setattr(me, "_http_json", api)
    store.set_password.side_effect = lambda service, account, saved: setattr(
        store.get_password, "return_value", saved
    )
    result = triage.TriageResult(
        id="sample.txt",
        category=triage.Category.other, priority=2, summary="Two deadlines",
        extracted=triage.Extracted(deadlines=[date(2026, 10, 5)]),
        proposed_events=[
            triage.CalendarEvent(
                title=title, start=date(2026, 10, 5), end=date(2026, 10, 6), description="",
            )
            for title in ("First deadline", "Second deadline")
        ],
    )
    created = asyncio.run(triage.create_events(result, triage.Calendar.me))
    assert len(created) == 2
    store.sign_in.assert_called_once_with(config)
    output = json.loads(triage.output_json(result, created))
    assert [event["id"] for event in output["calendar_events"]] == ["event-2", "event-4"]
    assert all(event["url"] is None for event in output["calendar_events"])


def test_me_error_becomes_standard_calendar_error(monkeypatch):
    monkeypatch.setattr(triage, "create_me_calendar_entry", Mock(side_effect=me.MeCalendarError("TLS failed")))
    with pytest.raises(triage.CalendarEntryError, match="TLS failed"):
        triage.create_calendar_entry(
            triage.Calendar.me, "Title", date(2026, 10, 5), date(2026, 10, 6),
        )


def test_cli_me_is_typed_and_uses_correct_status(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["triage.py", "--calendar", "me", "--create-events"])
    _, args = triage.parse_arguments()
    assert args.calendar is triage.Calendar.me

    async def first_triage(text):
        return triage.TriageResult(
            id="***stdin***",
            category=triage.Category.other, priority=2, summary="Deadline",
            extracted=triage.Extracted(deadlines=[date(2026, 10, 5)]),
            proposed_events=[triage.CalendarEvent(
                title="Deadline", start=date(2026, 10, 5), end=date(2026, 10, 6), description="",
            )],
        )

    def write(backend, *args):
        assert backend is triage.Calendar.me
        return {"id": "me-created"}

    monkeypatch.setattr(triage, "triage_openai", first_triage)
    monkeypatch.setattr(triage, "create_calendar_entry", write)
    monkeypatch.setattr(triage, "read_input", lambda _: "message")
    assert triage.main() == 0
    output = capsys.readouterr()
    assert "Created 1 Me Calendar event(s)." in output.err
    assert "Google Calendar" not in output.err
