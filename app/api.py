"""Privacy-aware FastAPI surface for Project PHANTOM."""

from __future__ import annotations

import asyncio
import logging
import secrets
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Header, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from phantom import __version__
from phantom.config import AppConfig, load_config
from phantom.exceptions import InvalidMediaError, PayloadTooLargeError, PhantomError
from phantom.logging_config import configure_logging
from phantom.schemas import (
    AnalysisResult,
    DialogueRequest,
    DialogueResponse,
    ErrorDetail,
    ErrorResponse,
    MultimodalAnalysisRequest,
    PrivacySettings,
    SessionCreateRequest,
    SessionInfo,
    TextAnalysisRequest,
)
from phantom.service.orchestrator import PhantomOrchestrator
from phantom.service.session_manager import SessionManager

LOGGER = logging.getLogger("phantom.api")
ALLOWED_AUDIO_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave"})
ALLOWED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png"})


def _error(status_code: int, code: str, message: str, request_id: str | None) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


async def _bounded_read(upload: UploadFile, max_bytes: int) -> bytes:
    """Read no more than the configured upload limit and close the temporary file."""

    try:
        data = await upload.read(max_bytes + 1)
    finally:
        await upload.close()
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"upload exceeds the {max_bytes}-byte limit")
    if not data:
        raise ValueError("uploaded file is empty")
    return data


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()
    configure_logging(config.log_level)
    sessions = SessionManager(config.session_ttl_seconds, config.max_sessions)
    orchestrator = PhantomOrchestrator(config, sessions)

    application = FastAPI(
        title="Project PHANTOM API",
        version=__version__,
        description=(
            "Experimental multimodal affect-aware interaction API. Observations are uncertain and "
            "are not medical assessments or emergency services."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.state.config = config
    application.state.sessions = sessions
    application.state.orchestrator = orchestrator

    @application.middleware("http")
    async def privacy_and_limits(request: Request, call_next: Any) -> Any:
        request_id = secrets.token_hex(8)
        request.state.request_id = request_id
        content_length = request.headers.get("content-length")
        # Multipart framing adds overhead; bounded reads enforce the exact media limit.
        if content_length and int(content_length) > config.max_upload_bytes + 1_048_576:
            return _error(
                413, "payload_too_large", "request body exceeds configured limit", request_id
            )
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
                call_next(request), timeout=config.request_timeout_seconds
            )
        except TimeoutError:
            return _error(
                504, "request_timeout", "request exceeded the processing timeout", request_id
            )
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @application.exception_handler(PhantomError)
    async def phantom_error_handler(request: Request, exc: PhantomError) -> JSONResponse:
        code_to_status = {
            "session_not_found": 404,
            "session_expired": 410,
            "consent_required": 403,
            "invalid_media": 415,
            "payload_too_large": 413,
            "session_capacity_reached": 503,
        }
        return _error(
            code_to_status.get(exc.code, 400),
            exc.code,
            str(exc),
            getattr(request.state, "request_id", None),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Do not echo supplied values; they may contain transcript or other personal data.
        locations = [".".join(str(item) for item in error["loc"]) for error in exc.errors()]
        message = "invalid request fields: " + ", ".join(locations[:10])
        return _error(
            422,
            "validation_error",
            message,
            getattr(request.state, "request_id", None),
        )

    @application.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return _error(
            400,
            "invalid_request",
            str(exc),
            getattr(request.state, "request_id", None),
        )

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "mode": "mock-capable",
            "active_sessions": sessions.active_count,
        }

    @application.post("/session", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
    def create_session(body: SessionCreateRequest) -> SessionInfo:
        return sessions.create(body)

    @application.delete("/session/{session_id}")
    def delete_session(session_id: str) -> dict[str, bool]:
        sessions.delete(session_id)
        return {"deleted": True}

    @application.post("/analyze/text", response_model=AnalysisResult)
    def analyze_text(body: TextAnalysisRequest) -> AnalysisResult:
        return orchestrator.analyze_text(body.session_id, body.text)

    @application.post("/analyze/audio", response_model=AnalysisResult)
    async def analyze_audio(
        session_id: Annotated[str, Form(min_length=16, max_length=128)],
        file: Annotated[UploadFile, File(description="WAV audio only")],
    ) -> AnalysisResult:
        content_type = (file.content_type or "").lower()
        if content_type not in ALLOWED_AUDIO_TYPES:
            await file.close()
            raise InvalidMediaError("audio upload must use a WAV content type")
        data = await _bounded_read(file, config.max_upload_bytes)
        filename = Path(file.filename or "upload.wav").name
        return orchestrator.analyze_audio(session_id, data, filename, content_type)

    @application.post("/analyze/image", response_model=AnalysisResult)
    async def analyze_image(
        session_id: Annotated[str, Form(min_length=16, max_length=128)],
        file: Annotated[UploadFile, File(description="JPEG or PNG image only")],
    ) -> AnalysisResult:
        content_type = (file.content_type or "").lower()
        if content_type not in ALLOWED_IMAGE_TYPES:
            await file.close()
            raise InvalidMediaError("image upload must be JPEG or PNG")
        data = await _bounded_read(file, config.max_upload_bytes)
        filename = Path(file.filename or "upload.jpg").name
        return orchestrator.analyze_image(session_id, data, filename, content_type)

    @application.post("/analyze/multimodal", response_model=AnalysisResult)
    def analyze_multimodal(body: MultimodalAnalysisRequest) -> AnalysisResult:
        return orchestrator.analyze_multimodal(body)

    @application.post("/dialogue/respond", response_model=DialogueResponse)
    def dialogue_respond(body: DialogueRequest) -> DialogueResponse:
        return orchestrator.respond(body)

    @application.get("/privacy/settings", response_model=PrivacySettings)
    def get_privacy_settings(
        session_id: Annotated[str, Header(alias="X-Session-ID", min_length=16, max_length=128)],
    ) -> PrivacySettings:
        return sessions.get_privacy(session_id)

    @application.put("/privacy/settings", response_model=PrivacySettings)
    def put_privacy_settings(
        body: PrivacySettings,
        session_id: Annotated[str, Header(alias="X-Session-ID", min_length=16, max_length=128)],
    ) -> PrivacySettings:
        return sessions.update_privacy(session_id, body)

    return application


app = create_app()
