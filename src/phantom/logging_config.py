"""Structured logging that redacts likely personal and biometric content."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

_SENSITIVE_KEYS = frozenset(
    {
        "audio",
        "image",
        "video",
        "transcript",
        "text",
        "raw_data",
        "face_embedding",
        "authorization",
        "cookie",
    }
)
_BASE64_PATTERN = re.compile(r"(?i)(?:[A-Za-z0-9+/]{80,}={0,2})")


def redact(value: Any) -> Any:
    """Recursively remove raw media, free text, credentials, and long base64 blobs."""

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in _SENSITIVE_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[REDACTED_BYTES:{len(value)}]"
    if isinstance(value, str):
        return _BASE64_PATTERN.sub("[REDACTED_BASE64]", value)
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class PrivacyFilter(logging.Filter):
    """Logging filter that sanitizes structured arguments before formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, Mapping):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(redact(item) for item in record.args)
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(PrivacyFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
