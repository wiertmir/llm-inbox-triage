"""Google Calendar desktop OAuth with an OS-protected credential cache."""

import json
import os
import webbrowser
from pathlib import Path

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow, WSGITimeoutError
from keyring.errors import KeyringError
from loguru import logger
from oauthlib.oauth2 import OAuth2Error
from requests.exceptions import RequestException

from calendar_auth_store import get_credential_store as _credential_store

GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
KEYRING_SERVICE = "llm-inbox-triage.google-calendar"


class GoogleCalendarAuthError(Exception):
    """Google authorization or credential storage failed."""


def get_google_calendar_access_token(
    credentials_path: str | Path | None = None,
    *,
    reauthorize: bool = False,
) -> str:
    """Sign in once, then reuse or refresh tokens from the OS credential store.

    Requires a Desktop app OAuth JSON file, supplied as ``credentials_path`` or
    GOOGLE_CALENDAR_CLIENT_SECRETS_FILE. ``reauthorize=True`` replaces the cached
    authorization after a fresh sign-in (for example, after access is revoked).
    """
    configured_path = credentials_path or os.getenv("GOOGLE_CALENDAR_CLIENT_SECRETS_FILE")
    if not configured_path:
        raise GoogleCalendarAuthError(
            "Set GOOGLE_CALENDAR_CLIENT_SECRETS_FILE to your Desktop app OAuth JSON "
            "file, or pass credentials_path"
        )
    path = Path(configured_path).expanduser()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoogleCalendarAuthError(
            "Cannot read Google OAuth credentials; check the configured JSON file"
        ) from exc
    if not isinstance(config, dict) or not isinstance(config.get("installed"), dict):
        raise GoogleCalendarAuthError("Google OAuth credentials must be for a Desktop app")
    installed = config["installed"]
    required = ("client_id", "client_secret", "auth_uri", "token_uri")
    if not all(isinstance(installed.get(name), str) and installed[name] for name in required):
        raise GoogleCalendarAuthError("Google Desktop app OAuth credentials are incomplete")

    try:
        store = _credential_store()
        cached = None if reauthorize else store.get_password(KEYRING_SERVICE, installed["client_id"])
    except (KeyringError, ImportError, RuntimeError) as exc:
        raise GoogleCalendarAuthError(
            "Cannot access the OS credential store; enable Windows Credential Manager, "
            "macOS Keychain, or Linux Secret Service"
        ) from exc

    credentials: Credentials | None = None
    if cached is not None:
        try:
            credentials = Credentials.from_authorized_user_info(json.loads(cached))
        except (ValueError, TypeError, AttributeError) as exc:
            raise GoogleCalendarAuthError(
                "Cached Google credentials are invalid; call "
                "get_google_calendar_access_token(reauthorize=True) to sign in again"
            ) from exc
        if credentials.client_id != installed["client_id"] or not credentials.has_scopes(
            GOOGLE_CALENDAR_SCOPES
        ):
            raise GoogleCalendarAuthError(
                "Cached Google credentials do not match the client or calendar scope; "
                "call get_google_calendar_access_token(reauthorize=True)"
            )
        if credentials.valid and credentials.token:
            return credentials.token
        if not credentials.refresh_token:
            raise GoogleCalendarAuthError(
                "Cached Google credentials have no refresh token; call "
                "get_google_calendar_access_token(reauthorize=True)"
            )
        logger.info("Refreshing Google Calendar authorization")
        try:
            credentials.refresh(Request())
        except (GoogleAuthError, RequestException) as exc:
            raise GoogleCalendarAuthError(
                "Google token refresh failed; check your connection. If access was "
                "revoked, call get_google_calendar_access_token(reauthorize=True)"
            ) from exc
    else:
        logger.info("Opening the browser for initial Google Calendar authorization")
        try:
            flow = InstalledAppFlow.from_client_config(
                config, GOOGLE_CALENDAR_SCOPES, autogenerate_code_verifier=True
            )
            authorized = flow.run_local_server(
                host="127.0.0.1",
                port=0,
                open_browser=True,
                timeout_seconds=180,
                authorization_prompt_message="",
                access_type="offline",
                prompt="consent",
            )
        except (
            GoogleAuthError, OAuth2Error, RequestException, WSGITimeoutError,
            webbrowser.Error, OSError, ValueError,
        ) as exc:
            raise GoogleCalendarAuthError(
                "Google sign-in failed or timed out; approve calendar access in the "
                "browser and check your Desktop app OAuth configuration"
            ) from exc
        if not isinstance(authorized, Credentials):
            raise GoogleCalendarAuthError("Google sign-in returned unsupported credentials")
        credentials = authorized

    if not credentials.valid or not credentials.token or not credentials.refresh_token:
        raise GoogleCalendarAuthError(
            "Google did not return usable access and refresh tokens; authorize again"
        )
    if credentials.granted_scopes is not None and not set(GOOGLE_CALENDAR_SCOPES).issubset(
        credentials.granted_scopes
    ):
        raise GoogleCalendarAuthError("Google Calendar permission was not granted")
    try:
        store.set_password(KEYRING_SERVICE, installed["client_id"], credentials.to_json())
    except (KeyringError, RuntimeError) as exc:
        raise GoogleCalendarAuthError("Cannot save Google tokens in the OS credential store") from exc
    return credentials.token
