from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from phantom.exceptions import ConsentRequiredError, SessionNotFoundError
from phantom.logging_config import PrivacyFilter, redact
from phantom.safety.privacy import require_consent, wipe_bytearray
from phantom.schemas import ConsentSettings, Modality, PrivacySettings, SessionCreateRequest
from phantom.service.session_manager import SessionManager


def test_optional_attributes_are_disabled_by_default() -> None:
    consent = ConsentSettings()
    assert consent.age_band_analysis is False
    assert consent.perceived_gender_presentation_analysis is False
    assert consent.store_raw_data is False
    assert consent.cloud_upload is False


def test_optional_attributes_require_camera_consent() -> None:
    with pytest.raises(ValidationError):
        ConsentSettings(age_band_analysis=True)


def test_remote_llm_requires_explicit_session_consent_but_cannot_enable_storage() -> None:
    consent = ConsentSettings(cloud_upload=True)
    assert consent.cloud_upload is True
    with pytest.raises(ValidationError):
        PrivacySettings(cloud_upload=True)


def test_consent_gate_is_modality_specific() -> None:
    consent = ConsentSettings(text_analysis=True)
    require_consent(consent, Modality.TEXT)
    with pytest.raises(ConsentRequiredError):
        require_consent(consent, Modality.AUDIO)


def test_session_deletion_removes_access() -> None:
    manager = SessionManager()
    session = manager.create(SessionCreateRequest())
    assert manager.active_count == 1
    manager.delete(session.session_id)
    assert manager.active_count == 0
    with pytest.raises(SessionNotFoundError):
        manager.get(session.session_id)


def test_privacy_cannot_expand_beyond_consent() -> None:
    manager = SessionManager()
    session = manager.create(SessionCreateRequest())
    with pytest.raises(ValueError, match="raw-data storage"):
        manager.update_privacy(
            session.session_id,
            PrivacySettings(store_raw_data=True),
        )


def test_mutable_raw_buffer_is_overwritten_and_cleared() -> None:
    media = bytearray(b"private-audio")
    wipe_bytearray(media)
    assert media == bytearray()


def test_logging_redacts_biometrics_and_transcripts() -> None:
    payload = {
        "audio": b"RIFF-private",
        "transcript": "very private",
        "quality": 0.8,
        "nested": {"face_embedding": [0.1, 0.2]},
    }
    cleaned = redact(payload)
    assert cleaned["audio"] == "[REDACTED]"
    assert cleaned["transcript"] == "[REDACTED]"
    assert cleaned["nested"]["face_embedding"] == "[REDACTED]"
    assert cleaned["quality"] == 0.8

    record = logging.LogRecord("test", logging.INFO, __file__, 1, payload, (), None)
    assert PrivacyFilter().filter(record)
    assert record.msg["audio"] == "[REDACTED]"
