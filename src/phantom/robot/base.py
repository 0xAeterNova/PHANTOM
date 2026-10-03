"""Robot output protocol and command types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RobotCommand:
    text: str
    speech_speed: float = 1.0
    pause_seconds: float = 0.0
    gesture: str | None = None
    kiosk_mode: bool = False


class RobotAdapter(Protocol):
    def deliver(self, command: RobotCommand) -> None: ...

    def stop(self) -> None: ...
