"""OAuth tests use synthetic credentials and never open a browser or keyring."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import WSGITimeoutError
from keyring.errors import PasswordSetError
from oauthlib.oauth2 import AccessDeniedError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import google_calendar_auth as auth  # noqa: E402


@pytest.fixture
def oauth(monkeypatch, tmp_path):
    config = {"installed": {
        "client_id": "test-client",
        "client_secret": "synthetic-secret",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }}
    path = tmp_path / "client_secret.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("GOOGLE_CALENDAR_CLIENT_SECRETS_FILE", str(path))
    store = Mock()
    store.get_password.return_value = None
    monkeypatch.setattr(auth, "_credential_store", lambda: store)
    factory = Mock()
    monkeypatch.setattr(auth.InstalledAppFlow, "from_client_config", factory)
    factory.return_value.run_local_server.return_value = make_credentials()
    return path, store, factory


def make_credentials(
    *, expired: bool = False, scopes: list[str] | None = None,
    refresh_token: str | None = "test-refresh",
) -> Credentials:
    expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
        hours=-1 if expired else 1
    )
    return Credentials(
        token="test-access",
        refresh_token=refresh_token,
        client_id="test-client",
        client_secret="synthetic-secret",
        token_uri="https://oauth2.googleapis.com/token",
        scopes=auth.GOOGLE_CALENDAR_SCOPES if scopes is None else scopes,
        expiry=expiry,
    )


def test_initial_authorization_is_saved_in_os_store(oauth):
    _, store, factory = oauth
    assert auth.get_google_calendar_access_token() == "test-access"
    assert factory.call_args.kwargs["autogenerate_code_verifier"] is True
    kwargs = factory.return_value.run_local_server.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 0
    assert kwargs["access_type"] == "offline"
    assert kwargs["prompt"] == "consent"
    assert kwargs["timeout_seconds"] == 180
    service, client_id, saved = store.set_password.call_args.args
    assert (service, client_id) == (auth.KEYRING_SERVICE, "test-client")
    assert json.loads(saved)["refresh_token"] == "test-refresh"


def test_valid_cached_token_needs_no_browser_or_refresh(oauth, monkeypatch):
    _, store, factory = oauth
    store.get_password.return_value = make_credentials().to_json()
    refresh = Mock()
    monkeypatch.setattr(auth.Credentials, "refresh", refresh)
    assert auth.get_google_calendar_access_token() == "test-access"
    refresh.assert_not_called()
    factory.assert_not_called()
    store.set_password.assert_not_called()


def test_expired_cached_token_is_refreshed_and_saved(oauth, monkeypatch):
    _, store, factory = oauth
    store.get_password.return_value = make_credentials(expired=True).to_json()
    calls = []

    def refresh(credentials, request):
        calls.append(request)
        credentials.token = "refreshed-access"
        credentials.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)

    monkeypatch.setattr(auth.Credentials, "refresh", refresh)
    assert auth.get_google_calendar_access_token() == "refreshed-access"
    assert len(calls) == 1
    factory.assert_not_called()
    assert json.loads(store.set_password.call_args.args[2])["token"] == "refreshed-access"


def test_failed_refresh_does_not_launch_browser(oauth, monkeypatch):
    _, store, factory = oauth
    store.get_password.return_value = make_credentials(expired=True).to_json()
    monkeypatch.setattr(auth.Credentials, "refresh", Mock(side_effect=RefreshError("revoked")))
    with pytest.raises(auth.GoogleCalendarAuthError, match="refresh failed"):
        auth.get_google_calendar_access_token()
    factory.assert_not_called()
    store.set_password.assert_not_called()


def test_reauthorize_replaces_cached_credentials(oauth):
    _, store, factory = oauth
    store.get_password.return_value = "invalid old cache"
    assert auth.get_google_calendar_access_token(reauthorize=True) == "test-access"
    store.get_password.assert_not_called()
    factory.assert_called_once()
    store.set_password.assert_called_once()


@pytest.mark.parametrize("cached", ["invalid json", "[]", "{}"])
def test_invalid_cache_fails_without_browser(oauth, cached):
    _, store, factory = oauth
    store.get_password.return_value = cached
    with pytest.raises(auth.GoogleCalendarAuthError, match="Cached Google credentials are invalid"):
        auth.get_google_calendar_access_token()
    factory.assert_not_called()


def test_wrong_cached_scope_requires_reauthorization(oauth):
    _, store, factory = oauth
    store.get_password.return_value = make_credentials(scopes=["openid"]).to_json()
    with pytest.raises(auth.GoogleCalendarAuthError, match="calendar scope"):
        auth.get_google_calendar_access_token()
    factory.assert_not_called()


def test_missing_configuration_is_explicit(monkeypatch):
    monkeypatch.delenv("GOOGLE_CALENDAR_CLIENT_SECRETS_FILE", raising=False)
    with pytest.raises(auth.GoogleCalendarAuthError, match="GOOGLE_CALENDAR_CLIENT_SECRETS_FILE"):
        auth.get_google_calendar_access_token()


def test_explicit_credentials_path_overrides_environment(oauth, monkeypatch):
    path, _, _ = oauth
    monkeypatch.setenv("GOOGLE_CALENDAR_CLIENT_SECRETS_FILE", "nonexistent.json")
    assert auth.get_google_calendar_access_token(path) == "test-access"


@pytest.mark.parametrize("content", ["invalid json", "[]", '{"web":{}}', '{"installed":{}}'])
def test_invalid_client_configuration_is_rejected(oauth, content):
    path, store, factory = oauth
    path.write_text(content, encoding="utf-8")
    with pytest.raises(auth.GoogleCalendarAuthError):
        auth.get_google_calendar_access_token()
    store.get_password.assert_not_called()
    factory.assert_not_called()


def test_missing_file_is_explicit(oauth):
    path, _, _ = oauth
    path.unlink()
    with pytest.raises(auth.GoogleCalendarAuthError, match="Cannot read"):
        auth.get_google_calendar_access_token()


@pytest.mark.parametrize("error", [AccessDeniedError(), WSGITimeoutError("timeout")])
def test_denied_or_timed_out_consent_is_explicit(oauth, error):
    _, store, factory = oauth
    factory.return_value.run_local_server.side_effect = error
    with pytest.raises(auth.GoogleCalendarAuthError, match="sign-in failed"):
        auth.get_google_calendar_access_token()
    store.set_password.assert_not_called()


def test_unavailable_os_store_does_not_fall_back_to_file(oauth):
    path, store, factory = oauth
    store.get_password.side_effect = PasswordSetError("locked")
    with pytest.raises(auth.GoogleCalendarAuthError, match="OS credential store"):
        auth.get_google_calendar_access_token()
    factory.assert_not_called()
    assert list(path.parent.iterdir()) == [path]


def test_failed_cache_write_is_explicit(oauth):
    _, store, _ = oauth
    store.set_password.side_effect = PasswordSetError("locked")
    with pytest.raises(auth.GoogleCalendarAuthError, match="Cannot save"):
        auth.get_google_calendar_access_token()


def test_missing_refresh_token_is_not_cached(oauth):
    _, store, factory = oauth
    credentials = make_credentials(refresh_token=None)
    factory.return_value.run_local_server.return_value = credentials
    with pytest.raises(auth.GoogleCalendarAuthError, match="access and refresh tokens"):
        auth.get_google_calendar_access_token()
    store.set_password.assert_not_called()


def test_missing_calendar_grant_is_not_cached(oauth):
    _, store, factory = oauth
    original = make_credentials()
    credentials = Credentials(
        token=original.token,
        refresh_token=original.refresh_token,
        client_id=original.client_id,
        client_secret=original.client_secret,
        token_uri=original.token_uri,
        scopes=auth.GOOGLE_CALENDAR_SCOPES,
        granted_scopes=["openid"],
        expiry=original.expiry,
    )
    factory.return_value.run_local_server.return_value = credentials
    with pytest.raises(auth.GoogleCalendarAuthError, match="permission was not granted"):
        auth.get_google_calendar_access_token()
    store.set_password.assert_not_called()
