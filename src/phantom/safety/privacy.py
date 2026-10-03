"""Consent enforcement and best-effort in-memory cleanup helpers."""

from __future__ import annotations

from phantom.exceptions import ConsentRequiredError
from phantom.schemas import ConsentSettings, Modality


def require_consent(consent: ConsentSettings, modality: Modality) -> None:
    allowed = {
        Modality.AUDIO: consent.microphone,
        Modality.VISION: consent.camera,
        Modality.TEXT: consent.text_analysis,
    }[modality]
    if not allowed:
        raise ConsentRequiredError(f"explicit {modality.value} consent is required before analysis")


def wipe_bytearray(buffer: bytearray | None) -> None:
    """Overwrite a mutable raw-media buffer before releasing it."""

    if buffer is None:
        return
    for index in range(len(buffer)):
        buffer[index] = 0
    buffer.clear()
