"""Thread-safe, bounded, in-memory session storage with expiration and deletion."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from phantom.compat import UTC
from phantom.exceptions import CapacityError, SessionExpiredError, SessionNotFoundError
from phantom.schemas import (
    AnalysisResult,
    ConsentSettings,
    PrivacySettings,
    SessionCreateRequest,
    SessionInfo,
    UserPreferences,
)


@dataclass(slots=True)
class SessionState:
    session_id: str
    created_at: datetime
    expires_at: datetime
    consent: ConsentSettings
    preferences: UserPreferences
    privacy: PrivacySettings
    mock_mode: bool
    last_analysis: AnalysisResult | None = None
    derived_features: dict[str, Any] = field(default_factory=dict)

    def public(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.session_id,
            created_at=self.created_at,
            expires_at=self.expires_at,
            consent=self.consent,
            preferences=self.preferences,
            mock_mode=self.mock_mode,
        )

    def clear(self) -> None:
        self.last_analysis = None
        self.derived_features.clear()


class SessionManager:
    """Own session lifecycle; raw audio/images are never accepted for storage."""

    def __init__(self, ttl_seconds: int = 1800, max_sessions: int = 1000) -> None:
        if ttl_seconds < 60:
            raise ValueError("session TTL must be at least 60 seconds")
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def create(self, request: SessionCreateRequest) -> SessionInfo:
        now = datetime.now(UTC)
        with self._lock:
            self._prune_locked(now)
            if len(self._sessions) >= self._max_sessions:
                raise CapacityError(
                    "session capacity reached; delete or wait for an existing session"
                )
            session_id = secrets.token_urlsafe(24)
            privacy = PrivacySettings(
                store_raw_data=request.consent.store_raw_data,
                store_derived_features=False,
                store_optional_attributes=request.consent.store_optional_attributes,
                cloud_upload=False,
                session_ttl_seconds=self._ttl_seconds,
            )
            state = SessionState(
                session_id=session_id,
                created_at=now,
                expires_at=now + timedelta(seconds=self._ttl_seconds),
                consent=request.consent,
                preferences=request.preferences,
                privacy=privacy,
                mock_mode=request.mock_mode,
            )
            self._sessions[session_id] = state
            return state.public()

    def get(self, session_id: str) -> SessionState:
        now = datetime.now(UTC)
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError("session does not exist or has been deleted")
            if state.expires_at <= now:
                state.clear()
                del self._sessions[session_id]
                raise SessionExpiredError("session expired and its in-memory data was deleted")
            return state

    def delete(self, session_id: str) -> bool:
        with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                raise SessionNotFoundError("session does not exist or was already deleted")
            state.clear()
            return True

    def set_last_analysis(self, session_id: str, analysis: AnalysisResult) -> None:
        with self._lock:
            state = self.get(session_id)
            state.last_analysis = analysis

    def get_privacy(self, session_id: str) -> PrivacySettings:
        return self.get(session_id).privacy

    def update_privacy(self, session_id: str, settings: PrivacySettings) -> PrivacySettings:
        with self._lock:
            state = self.get(session_id)
            if settings.store_raw_data and not state.consent.store_raw_data:
                raise ValueError("raw-data storage was not included in session consent")
            if settings.store_optional_attributes and not state.consent.store_optional_attributes:
                raise ValueError("optional-attribute storage was not included in session consent")
            # TTL changes are applied prospectively and remain bounded by schema validation.
            state.privacy = settings.model_copy(update={"cloud_upload": False})
            state.expires_at = datetime.now(UTC) + timedelta(seconds=settings.session_ttl_seconds)
            if not settings.store_derived_features:
                state.derived_features.clear()
            return state.privacy

    def prune_expired(self) -> int:
        with self._lock:
            return self._prune_locked(datetime.now(UTC))

    def _prune_locked(self, now: datetime) -> int:
        expired = [key for key, value in self._sessions.items() if value.expires_at <= now]
        for key in expired:
            self._sessions[key].clear()
            del self._sessions[key]
        return len(expired)

    @property
    def active_count(self) -> int:
        with self._lock:
            self._prune_locked(datetime.now(UTC))
            return len(self._sessions)
