from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from scripts.preview_external_camera import (
    CONNECTION_ERROR,
    build_parser,
    configure_console_output,
    open_camera,
    preview_camera,
)


def capture(*, opened: bool, frames: list[tuple[bool, object]] | None = None) -> Mock:
    result = Mock()
    result.isOpened.return_value = opened
    result.read.side_effect = frames or []
    return result


def cv2_mock(captures: list[Mock]) -> Mock:
    module = Mock()
    module.CAP_DSHOW = 700
    module.CAP_PROP_FRAME_WIDTH = 3
    module.CAP_PROP_FRAME_HEIGHT = 4
    module.VideoCapture.side_effect = captures
    return module


def test_parser_uses_requested_external_camera_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.camera_index == 1
    assert args.width == 1280
    assert args.height == 720


def test_console_uses_replacement_for_unsupported_arabic_glyphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = Mock()
    stderr = Mock()
    monkeypatch.setattr("scripts.preview_external_camera.sys.stdout", stdout)
    monkeypatch.setattr("scripts.preview_external_camera.sys.stderr", stderr)

    configure_console_output()

    stdout.reconfigure.assert_called_once_with(errors="replace")
    stderr.reconfigure.assert_called_once_with(errors="replace")


@pytest.mark.parametrize("argument", ["--camera-index=-1", "--width=0", "--height=-1"])
def test_parser_rejects_invalid_capture_values(argument: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([argument])


def test_windows_directshow_failure_falls_back_to_default_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directshow = capture(opened=False)
    fallback = capture(opened=True)
    module = cv2_mock([directshow, fallback])
    monkeypatch.setattr("scripts.preview_external_camera.os.name", "nt")

    opened = open_camera(module, 1)

    assert opened is fallback
    assert module.VideoCapture.call_args_list == [((1, 700),), ((1,),)]
    directshow.release.assert_called_once_with()


@pytest.mark.parametrize("exit_key", [ord("q"), 27])
def test_preview_sets_resolution_exits_and_always_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    exit_key: int,
) -> None:
    frame: Any = object()
    external_camera = capture(opened=True, frames=[(True, frame)])
    module = cv2_mock([external_camera])
    module.waitKey.return_value = exit_key
    monkeypatch.setattr("scripts.preview_external_camera.os.name", "posix")

    status = preview_camera(module, index=2, width=1280, height=720)

    assert status == 0
    assert external_camera.set.call_args_list == [((3, 1280),), ((4, 720),)]
    module.imshow.assert_called_once_with("PHANTOM external camera check - Iriun / USB", frame)
    external_camera.release.assert_called_once_with()
    module.destroyAllWindows.assert_called_once_with()


def test_preview_reports_bilingual_failure_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unavailable = capture(opened=False)
    module = cv2_mock([unavailable])
    monkeypatch.setattr("scripts.preview_external_camera.os.name", "posix")

    status = preview_camera(module, index=3, width=1280, height=720)

    assert status == 1
    error = capsys.readouterr().err
    assert CONNECTION_ERROR in error
    assert "Could not connect" in error
    assert "تعذر الاتصال" in error
    unavailable.release.assert_called_once_with()
    module.destroyAllWindows.assert_called_once_with()
