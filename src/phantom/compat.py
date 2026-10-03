"""Small compatibility helpers for supported Python runtimes."""

from __future__ import annotations

from datetime import timezone
from enum import Enum

# Keep these definitions identical on every supported interpreter.  Conditional
# aliases to the Python 3.11 stdlib names make Python 3.10 static analysis infer
# the enum values as ``Any`` and hide real type errors throughout the project.
UTC = timezone.utc


class StrEnum(str, Enum):
    """Small, typed Python 3.10-compatible string enum."""

    def __str__(self) -> str:
        return str(self.value)
