"""Integrated browser camera + microphone + AI conversation application."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from phantom.config import AppConfig, load_config
from phantom.exceptions import InvalidMediaError, PayloadTooLargeError, PhantomError
from phantom.logging_config import configure_logging
from phantom.runtime import RuntimeBundle, build_runtime_bundle
from phantom.schemas import (
    AudioObservation,
    EmotionLabel,
    ErrorDetail,
    ErrorResponse,
    MockSignal,
    RuntimeTurnResponse,
    SessionCreateRequest,
    StrictModel,
    VisionObservation,
)
from phantom.service.turn_pipeline import consent_for_runtime

LOGGER = logging.getLogger("phantom.realtime")
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_AUDIO_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave"})
ALLOWED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class RealtimeSessionRequest(StrictModel):
    microphone: bool = False
    camera: bool = False
    text_analysis: bool = True
    age_analysis: bool = False
    cloud_llm: bool = False


class RuntimeRespondRequest(StrictModel):
    session_id: str = Field(min_length=16, max_length=128)
    user_text: str = Field(min_length=1, max_length=10_000)


class DemoAudioRequest(StrictModel):
    session_id: str = Field(min_length=16, max_length=128)
    transcript: str = Field(min_length=1, max_length=10_000)
    label: EmotionLabel = EmotionLabel.NEUTRAL
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    quality: float = Field(default=0.9, ge=0.0, le=1.0)


class DemoVisionRequest(StrictModel):
    session_id: str = Field(min_length=16, max_length=128)
    label: EmotionLabel = EmotionLabel.NEUTRAL
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    quality: float = Field(default=0.9, ge=0.0, le=1.0)
    face_count: int = Field(default=1, ge=0, le=20)
    estimated_age: float | None = Field(default=None, ge=0.0, le=120.0)


def _error(status_code: int, code: str, message: str, request_id: str | None) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


async def _bounded_read(upload: UploadFile, max_bytes: int) -> bytes:
    try:
        data = await upload.read(max_bytes + 1)
    finally:
        await upload.close()
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"upload exceeds the {max_bytes}-byte limit")
    if not data:
        raise InvalidMediaError("uploaded media is empty")
    return data


def create_app(
    config: AppConfig | None = None,
    *,
    bundle: RuntimeBundle | None = None,
) -> FastAPI:
    selected_config = config or load_config()
    selected_bundle = bundle or build_runtime_bundle(selected_config)
    configure_logging(selected_config.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        del application
        selected_bundle.warmup()
        yield

    application = FastAPI(
        title="Project PHANTOM Real-Time Assistant",
        version="0.2.0",
        description=(
            "Consent-gated multimodal conversational assistant. Facial expression, vocal "
            "emotion, and age outputs are uncertain estimates, not diagnoses."
        ),
        lifespan=lifespan,
    )
    application.state.bundle = selected_bundle
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.middleware("http")
    async def privacy_headers(request: Request, call_next: Any) -> Any:
        request_id = secrets.token_hex(8)
        request.state.request_id = request_id
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > selected_config.max_upload_bytes + 1_048_576:
            return _error(413, "payload_too_large", "request exceeds the upload limit", request_id)
        LOGGER.info(
            {
                "event": "request",
                "method": request.method,
                "path": request.url.path,
                "request_id": request_id,
            }
        )
        try:
            response = await asyncio.wait_for(
                call_next(request), timeout=selected_config.request_timeout_seconds
            )
        except TimeoutError:
            return _error(504, "request_timeout", "processing exceeded the time limit", request_id)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(self)"
        return response

    @application.exception_handler(PhantomError)
    async def phantom_error_handler(request: Request, exc: PhantomError) -> JSONResponse:
        statuses = {
            "session_not_found": 404,
            "session_expired": 410,
            "consent_required": 403,
            "invalid_media": 415,
            "payload_too_large": 413,
            "session_capacity_reached": 503,
        }
        return _error(
            statuses.get(exc.code, 400),
            exc.code,
            str(exc),
            getattr(request.state, "request_id", None),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        locations = [".".join(str(part) for part in item["loc"]) for item in exc.errors()]
        return _error(
            422,
            "validation_error",
            "invalid request fields: " + ", ".join(locations[:8]),
            getattr(request.state, "request_id", None),
        )

    @application.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError) -> JSONResponse:
        return _error(
            400,
            "invalid_request",
            str(exc),
            getattr(request.state, "request_id", None),
        )

    @application.exception_handler(RuntimeError)
    async def backend_error(request: Request, exc: RuntimeError) -> JSONResponse:
        del exc
        return _error(
            503,
            "backend_unavailable",
            "a configured real backend is unavailable; no simulated value was substituted",
            getattr(request.state, "request_id", None),
        )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return selected_bundle.health()

    @application.post("/api/session", status_code=status.HTTP_201_CREATED)
    def create_session(body: RealtimeSessionRequest) -> dict[str, Any]:
        if not any((body.microphone, body.camera, body.text_analysis)):
            raise ValueError("select at least one consented input")
        session = selected_bundle.sessions.create(
            SessionCreateRequest(
                consent=consent_for_runtime(
                    microphone=body.microphone,
                    camera=body.camera,
                    text_analysis=body.text_analysis,
                    age_analysis=body.age_analysis,
                    cloud_upload=body.cloud_llm,
                ),
                mock_mode=selected_bundle.runtime.mode == "demo",
            )
        )
        return {
            **session.model_dump(mode="json"),
            "mode": selected_bundle.runtime.mode,
            "camera_interval_ms": round(selected_config.vision.analyze_interval_seconds * 1000),
            "estimates_notice": (
                "Face emotion, voice emotion, and approximate age are probabilistic estimates."
            ),
        }

    @application.delete("/api/session/{session_id}")
    def delete_session(session_id: str) -> dict[str, bool]:
        state = selected_bundle.sessions.get(session_id)
        selected_bundle.runtime.clear_session_runtime(state)
        selected_bundle.sessions.delete(session_id)
        return {"deleted": True}

    @application.post("/api/vision", response_model=VisionObservation)
    async def analyze_vision(
        session_id: Annotated[str, Form(min_length=16, max_length=128)],
        file: Annotated[UploadFile, File(description="JPEG, PNG, or WebP camera snapshot")],
    ) -> VisionObservation:
        content_type = (file.content_type or "").partition(";")[0].lower()
        if content_type not in ALLOWED_IMAGE_TYPES:
            await file.close()
            raise InvalidMediaError("camera snapshot must be JPEG, PNG, or WebP")
        data = await _bounded_read(file, selected_config.max_upload_bytes)
        filename = Path(file.filename or "camera.jpg").name
        return selected_bundle.runtime.analyze_vision(session_id, data, filename, content_type)

    @application.post("/api/audio", response_model=AudioObservation)
    async def analyze_audio(
        session_id: Annotated[str, Form(min_length=16, max_length=128)],
        file: Annotated[UploadFile, File(description="Browser-recorded PCM WAV")],
    ) -> AudioObservation:
        content_type = (file.content_type or "").partition(";")[0].lower()
        if content_type not in ALLOWED_AUDIO_TYPES:
            await file.close()
            raise InvalidMediaError("microphone capture must be PCM WAV")
        data = await _bounded_read(file, selected_config.max_upload_bytes)
        filename = Path(file.filename or "microphone.wav").name
        return selected_bundle.runtime.analyze_audio(session_id, data, filename, content_type)

    @application.post("/api/respond", response_model=RuntimeTurnResponse)
    def respond(body: RuntimeRespondRequest) -> RuntimeTurnResponse:
        return selected_bundle.runtime.respond(body.session_id, body.user_text)

    @application.post("/api/demo/audio", response_model=AudioObservation)
    def demo_audio(body: DemoAudioRequest) -> AudioObservation:
        return selected_bundle.runtime.set_demo_audio(
            body.session_id,
            transcript=body.transcript,
            signal=MockSignal(
                label=body.label,
                confidence=body.confidence,
                quality=body.quality,
            ),
        )

    @application.post("/api/demo/vision", response_model=VisionObservation)
    def demo_vision(body: DemoVisionRequest) -> VisionObservation:
        return selected_bundle.runtime.set_demo_vision(
            body.session_id,
            signal=MockSignal(
                label=body.label,
                confidence=body.confidence,
                quality=body.quality,
                face_count=body.face_count,
            ),
            estimated_age=body.estimated_age,
        )

    return application


app = create_app()
