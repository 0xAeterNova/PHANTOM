"""In-memory robot adapter used by tests and the default demo."""

from __future__ import annotations

from phantom.robot.base import RobotCommand


class MockRobotAdapter:
    def __init__(self) -> None:
        self.commands: list[RobotCommand] = []
        self.stopped = False

    def deliver(self, command: RobotCommand) -> None:
        if not command.text.strip():
            raise ValueError("robot output text cannot be empty")
        if not 0.5 <= command.speech_speed <= 2.0:
            raise ValueError("speech speed must be between 0.5 and 2.0")
        self.stopped = False
        self.commands.append(command)

    def stop(self) -> None:
        self.stopped = True

    @property
    def last_command(self) -> RobotCommand | None:
        return self.commands[-1] if self.commands else None
