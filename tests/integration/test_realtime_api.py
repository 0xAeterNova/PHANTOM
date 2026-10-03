from __future__ import annotations

import base64
import io
import math
import struct
import wave
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.realtime import create_app
from phantom.config import (
    AppConfig,
    AudioModelConfig,
    LLMConfig,
    RuntimeConfig,
    SpeechToTextConfig,
    TTSConfig,
    VisionModelConfig,
)
from phantom.runtime import RuntimeBundle, build_runtime_bundle
from phantom.tts import SpeechResult, TTSInfo, TTSMode


class PlayableWavTTS:
    """Return audible PCM WAV speech stand-ins for API serialization acceptance tests."""

    def __init__(self) -> None:
        self._info = TTSInfo(name="test-playable-wav", mode=TTSMode.REAL)
        self.calls = 0

    @property
    def info(self) -> TTSInfo:
        return self._info

    def synthesize(self, text: str) -> SpeechResult:
        assert text
        self.calls += 1
        sample_rate = 16_000
        frame_count = 3_200
        # A bounded 440 Hz waveform catches a regression where the API returns
        # a valid WAV container whose PCM payload is actually silent.
        frames = b"".join(
            struct.pack("<h", round(8_000 * math.sin(2 * math.pi * 440 * frame / sample_rate)))
            for frame in range(frame_count)
        )
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(frames)
        return SpeechResult(
            provider=self.info,
            audio_wav=output.getvalue(),
            mime_type="audio/wav",
            audio_available=True,
            text_only=False,
        )

    def stop(self) -> None:
        return None


@pytest.fixture
def demo_api() -> Iterator[tuple[TestClient, RuntimeBundle]]:
    """Run the deterministic profile without importing optional ML packages."""

    config = AppConfig(max_upload_bytes=1024 * 1024)
    bundle = build_runtime_bundle(config)
    with TestClient(create_app(config, bundle=bundle)) as client:
        yield client, bundle


def _create_demo_session(client: TestClient) -> str:
    response = client.post(
        "/api/session",
        json={
            "microphone": True,
            "camera": True,
            "text_analysis": True,
            "age_analysis": True,
            "cloud_llm": False,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["mode"] == "demo"
    assert body["mock_mode"] is True
    assert body["consent"]["age_band_analysis"] is True
    assert body["consent"]["store_raw_data"] is False
    assert body["consent"]["store_optional_attributes"] is False
    assert body["consent"]["cloud_upload"] is False
    return str(body["session_id"])


def test_health_discloses_demo_mode_and_in_memory_privacy(
    demo_api: tuple[TestClient, RuntimeBundle],
) -> None:
    client, _bundle = demo_api

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-request-id"]
    body = response.json()
    assert body["status"] == "ready"
    assert body["mode"] == "demo"
    assert body["active_sessions"] == 0
    assert body["privacy"] == {
        "save_audio": False,
        "save_camera": False,
        "in_memory_only": True,
    }
    assert set(body["components"]) == {
        "speech_to_text",
        "voice_emotion",
        "facial_emotion",
        "age",
        "llm",
        "tts",
    }
    for component in body["components"].values():
        assert component["mode"] == "demo"
        assert component["provider"] == "deterministic-demo"
        assert component["readiness"] == "ready"
        assert "simulated demo component" in component["detail"]


def test_session_requires_input_consent_and_enforces_each_modality(
    demo_api: tuple[TestClient, RuntimeBundle],
) -> None:
    client, _bundle = demo_api

    empty = client.post(
        "/api/session",
        json={
            "microphone": False,
            "camera": False,
            "text_analysis": False,
            "age_analysis": False,
            "cloud_llm": False,
        },
    )
    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "invalid_request"

    text_only = client.post(
        "/api/session",
        json={"text_analysis": True, "age_analysis": True},
    )
    assert text_only.status_code == 201
    text_only_body = text_only.json()
    assert text_only_body["consent"]["age_band_analysis"] is False
    session_id = str(text_only_body["session_id"])

    audio = client.post(
        "/api/demo/audio",
        json={"session_id": session_id, "transcript": "private speech"},
    )
    vision = client.post(
        "/api/demo/vision",
        json={"session_id": session_id, "label": "neutral"},
    )
    assert audio.status_code == 403
    assert audio.json()["error"]["code"] == "consent_required"
    assert vision.status_code == 403
    assert vision.json()["error"]["code"] == "consent_required"


def test_deterministic_demo_turn_has_synthetic_provenance_memory_and_cleanup(
    demo_api: tuple[TestClient, RuntimeBundle],
) -> None:
    client, bundle = demo_api
    session_id = _create_demo_session(client)

    vision = client.post(
        "/api/demo/vision",
        json={
            "session_id": session_id,
            "label": "angry",
            "confidence": 0.93,
            "quality": 0.91,
            "face_count": 1,
            "estimated_age": 27,
        },
    )
    assert vision.status_code == 200, vision.text
    vision_body = vision.json()
    assert vision_body["result"]["metadata"] == {
        "mode": "demo",
        "synthetic": True,
        "face_count": 1,
    }
    assert "simulated demonstration value" in vision_body["result"]["reason"]
    assert vision_body["optional_demographic_estimates"]["estimated_age"] == 27
    assert vision_body["optional_demographic_estimates"]["age_band"] == "young adult"
    assert "Simulated approximate age" in vision_body["optional_demographic_estimates"]["notice"]

    transcript = "I am happy today."
    audio = client.post(
        "/api/demo/audio",
        json={
            "session_id": session_id,
            "transcript": transcript,
            "label": "sad",
            "confidence": 0.94,
            "quality": 0.92,
        },
    )
    assert audio.status_code == 200, audio.text
    audio_body = audio.json()
    assert audio_body["language"] == "demo"
    assert audio_body["result"]["metadata"]["synthetic"] is True
    assert audio_body["result"]["metadata"]["mode"] == "demo"

    first = client.post(
        "/api/respond",
        json={"session_id": session_id, "user_text": transcript},
    )
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["transcript_source"] == "speech_to_text"
    assert first_body["audio"]["metadata"]["synthetic"] is True
    assert first_body["vision"]["metadata"]["synthetic"] is True
    assert first_body["fusion"]["agreement"] == "conflict"
    assert first_body["fusion"]["words_have_priority"] is True
    assert first_body["fusion"]["explicit_self_report"] is True
    assert "overrides conflicting sensors" in first_body["fusion"]["explanation"]
    assert first_body["assistant_text"].startswith("[DEMO RESPONSE]")
    assert "not a real language model" in first_body["assistant_text"]
    assert first_body["llm_provider"] == "deterministic-demo"
    assert first_body["tts_provider"] == "text-only"
    assert first_body["tts_audio_base64"] is None
    assert first_body["history_turns"] == 1

    second = client.post(
        "/api/respond",
        json={
            "session_id": session_id,
            "user_text": "Please continue; passing my test made me feel relieved.",
        },
    )
    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["assistant_text"] == first_body["assistant_text"]
    assert second_body["history_turns"] == 2
    assert second_body["transcript_source"] == "typed"
    assert bundle.health()["active_sessions"] == 1

    deleted = client.delete(f"/api/session/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True}
    assert bundle.health()["active_sessions"] == 0

    after_delete = client.post(
        "/api/respond",
        json={"session_id": session_id, "user_text": "Are you still there?"},
    )
    assert after_delete.status_code == 404
    assert after_delete.json()["error"]["code"] == "session_not_found"


def test_respond_serializes_playable_wav_and_mime_type_across_multiple_turns(
    demo_api: tuple[TestClient, RuntimeBundle],
) -> None:
    client, bundle = demo_api
    tts = PlayableWavTTS()
    bundle.runtime.tts = tts
    session_id = _create_demo_session(client)

    for turn in range(1, 11):
        response = client.post(
            "/api/respond",
            json={"session_id": session_id, "user_text": f"Acceptance turn {turn}"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["tts_provider"] == "test-playable-wav"
        assert body["tts_mime_type"] == "audio/wav"
        assert body["history_turns"] == min(turn, bundle.config.llm.max_history_turns)
        encoded = body["tts_audio_base64"]
        assert isinstance(encoded, str)
        wav_bytes = base64.b64decode(encoded, validate=True)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16_000
            assert wav.getnframes() == 3_200
            assert wav.getnframes() / wav.getframerate() == pytest.approx(0.2)
            pcm = wav.readframes(wav.getnframes())

        samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
        peak = max(abs(sample) for sample in samples)
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        non_silent_ratio = sum(sample != 0 for sample in samples) / len(samples)
        assert peak >= 7_900
        assert rms >= 5_000
        assert non_silent_ratio >= 0.95

    assert tts.calls == 10

    deleted = client.delete(f"/api/session/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True}
    assert bundle.sessions.active_count == 0

    after_delete = client.post(
        "/api/respond",
        json={"session_id": session_id, "user_text": "This must not synthesize"},
    )
    assert after_delete.status_code == 404
    assert after_delete.json()["error"]["code"] == "session_not_found"
    assert tts.calls == 10


def test_real_session_rejects_demo_signal_endpoints_without_loading_models() -> None:
    config = AppConfig(
        mock_mode=False,
        runtime=RuntimeConfig(mode="real", preload_models=False),
        audio=AudioModelConfig(backend="huggingface"),
        stt=SpeechToTextConfig(backend="faster-whisper"),
        vision=VisionModelConfig(backend="deepface"),
        llm=LLMConfig(backend="transformers"),
        tts=TTSConfig(backend="pyttsx3"),
    )
    bundle = build_runtime_bundle(config)
    client = TestClient(create_app(config, bundle=bundle))
    try:
        session = client.post(
            "/api/session",
            json={"microphone": True, "camera": True, "text_analysis": True},
        )
        assert session.status_code == 201, session.text
        assert session.json()["mode"] == "real"
        assert session.json()["mock_mode"] is False
        session_id = str(session.json()["session_id"])

        audio = client.post(
            "/api/demo/audio",
            json={"session_id": session_id, "transcript": "forged demo input"},
        )
        vision = client.post(
            "/api/demo/vision",
            json={"session_id": session_id, "label": "happy"},
        )
        assert audio.status_code == 400
        assert audio.json()["error"] == {
            "code": "invalid_request",
            "message": "this operation requires a demo session",
            "request_id": audio.json()["error"]["request_id"],
        }
        assert vision.status_code == 400
        assert vision.json()["error"]["code"] == "invalid_request"
        assert "synthetic" not in audio.text.lower()
        assert "synthetic" not in vision.text.lower()
    finally:
        client.close()
