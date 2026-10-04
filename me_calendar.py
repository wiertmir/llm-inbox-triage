"""Me calendar REST API and desktop PKCE sign-in, with verified TLS."""

import base64
import hashlib
import json
import math
import os
import secrets
import socket
import ssl
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from keyring.errors import KeyringError
from loguru import logger

from calendar_auth_store import get_credential_store
from calendar_tools import validate_event_times

KEYRING_SERVICE = "llm-inbox-triage.me-calendar"
AUTH_TIMEOUT_SECONDS = 300


class MeCalendarError(Exception):
    """Me sign-in, credential storage, or calendar API request failed."""

    def __init__(self, message: str, *, event_may_exist: bool = False) -> None:
        super().__init__(message)
        self.event_may_exist = event_may_exist


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl) -> None:
        # Never forward OAuth codes or bearer tokens to a redirected endpoint.
        return None


class CallbackServer(ThreadingHTTPServer):
    daemon_threads = True

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address


@dataclass(frozen=True)
class MeConfig:
    base: str
    client_id: str
    tls: ssl.SSLContext
    calendar_id: str | None

    @classmethod
    def from_environment(cls) -> "MeConfig":
        base = os.getenv("ME_BASE", "https://pop-os.local").rstrip("/")
        url = urlsplit(base)
        if (
            url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment
        ):
            raise MeCalendarError("ME_BASE must be an HTTPS URL without credentials, query, or fragment")
        client_id = os.getenv("ME_CLIENT_ID", "me-desktop")
        if not client_id.strip():
            raise MeCalendarError("ME_CLIENT_ID must not be empty")
        ca_file = os.getenv("ME_CA_FILE")
        try:
            tls = ssl.create_default_context()
            if ca_file:
                tls.load_verify_locations(cafile=str(Path(ca_file).expanduser()))
        except (OSError, ssl.SSLError) as exc:
            raise MeCalendarError("Cannot load ME_CA_FILE; check the path to Me's CA certificate") from exc
        return cls(base, client_id, tls, os.getenv("ME_CALENDAR_ID") or None)


def _http_json(
    config: MeConfig, method: str, path: str, *,
    token: str | None = None, form: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if form is not None:
        data = urlencode(form).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(config.base + path, data=data, headers=headers, method=method)
    opener = build_opener(HTTPSHandler(context=config.tls), NoRedirect())
    event_write = method == "POST" and path.endswith("/events")
    try:
        with opener.open(request, timeout=30) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = ""
        try:
            error_body = json.loads(exc.read())
        except (UnicodeDecodeError, json.JSONDecodeError):
            error_body = None
        if isinstance(error_body, dict) and isinstance(error_body.get("message"), str):
            detail = f": {error_body['message']}"
        raise MeCalendarError(
            f"Me {method} {path} failed (HTTP {exc.code}){detail}",
            event_may_exist=event_write and exc.code != 422,
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise MeCalendarError(
            f"Cannot reach Me for {method} {path}; check ME_BASE and ME_CA_FILE. "
            "For event writes, check your calendar before retrying.",
            event_may_exist=event_write,
        ) from exc
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeCalendarError(
            f"Me {method} {path} returned invalid JSON; check your calendar before retrying writes",
            event_may_exist=event_write,
        ) from exc


def _sign_in(config: MeConfig) -> dict[str, Any]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(32)
    received: dict[str, str] = {}
    done = threading.Event()
    received_lock = threading.Lock()

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path != "/callback":
                self.send_error(404)
                return
            query = parse_qs(url.query)
            states = query.get("state", [])
            if len(states) != 1 or not secrets.compare_digest(states[0], state):
                self.send_error(400, "Invalid OAuth state")
                return
            codes = query.get("code", [])
            errors = query.get("error", [])
            if len(codes) == 1 and codes[0] and not errors:
                key, value = "code", codes[0]
            elif len(errors) == 1 and not codes:
                key, value = "error", errors[0]
            else:
                self.send_error(400, "Invalid OAuth response")
                return
            with received_lock:
                if received:
                    self.send_error(409, "OAuth response already received")
                    return
                received[key] = value
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<p>Return to the terminal to see the sign-in result.</p>")
            except OSError:
                logger.warning("Browser disconnected after submitting the Me OAuth response")
            finally:
                done.set()

        def log_message(self, format: str, *args: Any) -> None:
            # Callback query strings contain authorization codes.
            return

    try:
        with CallbackServer(("127.0.0.1", 0), Callback) as server:
            server_thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
            )
            server_thread.start()
            redirect_uri = f"http://127.0.0.1:{server.server_port}/callback"
            authorize = config.base + "/oauth/authorize?" + urlencode({
                "response_type": "code", "client_id": config.client_id,
                "redirect_uri": redirect_uri, "scope": "openid", "state": state,
                "code_challenge": challenge, "code_challenge_method": "S256",
            })
            try:
                logger.info("Opening the browser for Me Calendar authorization")
                if not webbrowser.open(authorize):
                    raise MeCalendarError("Could not open a browser for Me sign-in")
                if not done.wait(timeout=AUTH_TIMEOUT_SECONDS):
                    raise MeCalendarError(
                        "Me sign-in timed out; the browser did not return within five minutes"
                    )
            finally:
                server.shutdown()
                server_thread.join()
    except (OSError, webbrowser.Error) as exc:
        raise MeCalendarError("Could not start Me browser sign-in and its local callback server") from exc
    if "error" in received:
        raise MeCalendarError("Me sign-in was denied; no token was obtained")
    logger.info("Me sign-in completed; obtaining an access token")
    tokens = _http_json(config, "POST", "/oauth/token", form={
        "grant_type": "authorization_code", "client_id": config.client_id,
        "code": received["code"], "redirect_uri": redirect_uri, "code_verifier": verifier,
    })
    if not isinstance(tokens, dict) or not isinstance(tokens.get("access_token"), str) or not tokens["access_token"]:
        raise MeCalendarError("Me sign-in returned no access token")
    return tokens


def get_me_calendar_access_token(config: MeConfig, *, reauthorize: bool = False) -> str:
    """Reuse a cached access token; sign in again after it expires (Me has no refresh flow)."""
    override = os.getenv("ME_ACCESS_TOKEN")
    if override and not reauthorize:
        return override
    account = f"{config.base}|{config.client_id}"
    try:
        store = get_credential_store()
        cached = None if reauthorize else store.get_password(KEYRING_SERVICE, account)
    except (KeyringError, ImportError, RuntimeError) as exc:
        raise MeCalendarError("Cannot access the OS credential store for Me tokens") from exc
    if cached is not None:
        try:
            saved = json.loads(cached)
            token, expires_at = saved["access_token"], saved["expires_at"]
            if (
                not isinstance(token, str) or not token
                or isinstance(expires_at, bool) or not isinstance(expires_at, (int, float))
                or not math.isfinite(expires_at)
            ):
                raise ValueError("Invalid cached token")
        except (ValueError, TypeError, KeyError) as exc:
            raise MeCalendarError(
                "Invalid cached Me credentials; call get_me_calendar_access_token("
                "MeConfig.from_environment(), reauthorize=True)"
            ) from exc
        if expires_at > time.time() + 60:
            return token
        logger.info("Cached Me authorization expired; a fresh browser sign-in is required")
    tokens = _sign_in(config)
    issued_at = time.time()
    # The supplied Me example documents a 15-minute token lifetime.
    expires_in = tokens.get("expires_in", 900)
    if (
        isinstance(expires_in, bool) or not isinstance(expires_in, (int, float))
        or not math.isfinite(expires_in) or expires_in <= 0
    ):
        raise MeCalendarError("Me sign-in returned an invalid token lifetime")
    saved = json.dumps({"access_token": tokens["access_token"], "expires_at": issued_at + expires_in})
    try:
        store.set_password(KEYRING_SERVICE, account, saved)
    except (KeyringError, RuntimeError) as exc:
        raise MeCalendarError("Cannot save Me tokens in the OS credential store") from exc
    return tokens["access_token"]


def _event_id(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise MeCalendarError("Me returned no valid calendar/event ID")


def build_me_event_body(
    title: str, start: date | datetime, end: date | datetime, description: str = "",
) -> dict[str, Any]:
    """Apply Me's EventInput rules before authorization or any calendar writes."""
    validate_event_times(start, end)
    if len(title) > 500:
        raise MeCalendarError("Me event summary must be at most 500 characters")
    if len(description) > 10000:
        raise MeCalendarError("Me event description must be at most 10000 characters")
    all_day = not isinstance(start, datetime)
    body: dict[str, Any] = {
        "summary": title, "description": description, "all_day": all_day, "reminders": [15],
    }
    if isinstance(start, datetime) and isinstance(end, datetime):
        start = start.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)
        end = end.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)
        if end <= start:
            raise MeCalendarError("Me event end must be after start at whole-second precision")
        body["tz"] = "UTC"
    if not (1900 <= start.year <= 2200 and 1900 <= end.year <= 2200):
        raise MeCalendarError("Me event start and end years must be between 1900 and 2200")
    body["start"] = start.isoformat()
    body["end"] = end.isoformat()
    return body


def create_me_calendar_entry(
    title: str, start: date | datetime, end: date | datetime,
    description: str = "", *, access_token: str | None = None,
) -> dict[str, Any]:
    body = build_me_event_body(title, start, end, description)
    config = MeConfig.from_environment()
    token = access_token or get_me_calendar_access_token(config)
    calendar_id = config.calendar_id
    if calendar_id is None:
        calendars = _http_json(config, "GET", "/calendar/v1/calendars", token=token)
        if not isinstance(calendars, list) or not calendars or not isinstance(calendars[0], dict):
            raise MeCalendarError("Me returned no calendars; set ME_CALENDAR_ID or check your account")
        calendar_id = _event_id(calendars[0].get("id"))
    event = _http_json(
        config, "POST", f"/calendar/v1/calendars/{quote(calendar_id, safe='')}/events",
        token=token, body=body,
    )
    if not isinstance(event, dict):
        raise MeCalendarError(
            "Me returned an invalid event; check your calendar before retrying", event_may_exist=True,
        )
    try:
        event_id = _event_id(event.get("id"))
    except MeCalendarError as exc:
        raise MeCalendarError(
            "Me returned no valid event ID; check your calendar before retrying", event_may_exist=True,
        ) from exc
    return {**event, "id": event_id}
