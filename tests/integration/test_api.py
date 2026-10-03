from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from phantom.config import AppConfig


def client() -> TestClient:
    return TestClient(create_app(AppConfig(max_upload_bytes=1024)))


def create_session(test_client: TestClient, **consent: bool) -> str:
    response = test_client.post(
        "/session",
        json={"consent": consent, "mock_mode": True},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def test_health_and_openapi_are_available() -> None:
    with client() as test_client:
        assert test_client.get("/health").json()["status"] == "ok"
        assert test_client.get("/openapi.json").status_code == 200


def test_api_mock_end_to_end_then_delete() -> None:
    with client() as test_client:
        session_id = create_session(test_client, microphone=True, camera=True, text_analysis=True)
        response = test_client.post(
            "/analyze/multimodal",
            json={
                "session_id": session_id,
                "text": "The local mock is working.",
                "mock_audio": {"label": "happy", "confidence": 0.95, "quality": 0.95},
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body["modalities_available"]) == {"audio", "text"}
        assert body["optional_demographic_estimates"]["age_band"] == "analysis-disabled"
        assert body["safety"]["route"] == "normal"

        dialogue = test_client.post(
            "/dialogue/respond",
            json={"session_id": session_id, "estimate_is_incorrect": True},
        )
        assert dialogue.status_code == 200
        assert "correcting" in dialogue.json()["text"].lower()

        assert test_client.delete(f"/session/{session_id}").status_code == 200
        assert (
            test_client.get("/privacy/settings", headers={"X-Session-ID": session_id}).status_code
            == 404
        )


def test_consent_violation_returns_stable_error() -> None:
    with client() as test_client:
        session_id = create_session(test_client, text_analysis=True)
        response = test_client.post(
            "/analyze/multimodal",
            json={"session_id": session_id, "mock_audio": {"label": "neutral"}},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "consent_required"


def test_extra_and_malicious_input_is_rejected_without_echoing_value() -> None:
    secret_marker = "DO_NOT_ECHO_PRIVATE_TRANSCRIPT"
    with client() as test_client:
        response = test_client.post(
            "/session",
            json={"consent": {}, "unexpected": secret_marker},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
        assert secret_marker not in response.text


def test_upload_content_type_and_size_limits() -> None:
    with client() as test_client:
        session_id = create_session(test_client, microphone=True)
        wrong_type = test_client.post(
            "/analyze/audio",
            data={"session_id": session_id},
            files={"file": ("attack.exe", b"MZ", "application/octet-stream")},
        )
        assert wrong_type.status_code == 415
        assert wrong_type.json()["error"]["code"] == "invalid_media"

        too_large = test_client.post(
            "/analyze/audio",
            data={"session_id": session_id},
            files={"file": ("large.wav", b"RIFF" + b"x" * 2000, "audio/wav")},
        )
        assert too_large.status_code == 413


def test_privacy_settings_are_session_scoped_and_local_only() -> None:
    with client() as test_client:
        session_id = create_session(test_client, text_analysis=True)
        current = test_client.get("/privacy/settings", headers={"X-Session-ID": session_id})
        assert current.status_code == 200
        assert current.json()["cloud_upload"] is False
        rejected = test_client.put(
            "/privacy/settings",
            headers={"X-Session-ID": session_id},
            json={**current.json(), "cloud_upload": True},
        )
        assert rejected.status_code == 422
