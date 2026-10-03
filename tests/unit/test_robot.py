from __future__ import annotations

import pytest

from phantom.robot.base import RobotCommand
from phantom.robot.mock_robot import MockRobotAdapter


def test_mock_robot_records_bounded_output() -> None:
    robot = MockRobotAdapter()
    command = RobotCommand(
        text="Would you like to tell me more?", speech_speed=0.8, pause_seconds=1.0
    )
    robot.deliver(command)
    assert robot.last_command == command
    robot.stop()
    assert robot.stopped is True


def test_mock_robot_rejects_unsafe_speed() -> None:
    robot = MockRobotAdapter()
    with pytest.raises(ValueError):
        robot.deliver(RobotCommand(text="hello", speech_speed=5.0))
