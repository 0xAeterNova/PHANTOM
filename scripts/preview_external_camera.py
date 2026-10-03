"""Preview an external or Iriun camera without starting the PHANTOM models."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from typing import Any

WINDOW_TITLE = "PHANTOM external camera check - Iriun / USB"
CONNECTION_ERROR = (
    "Could not connect to the external/Iriun camera. "
    "Try --camera-index 2 or 3 and close other apps using the camera.\n"
    "تعذر الاتصال بكاميرا Iriun أو الكاميرا الخارجية. "
    "جرّب --camera-index 2 أو 3 وأغلق البرامج الأخرى التي تستخدم الكاميرا."
)


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def camera_index(value: str) -> int:
    """Parse a non-negative OpenCV camera index."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("camera index must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the camera diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-index",
        type=camera_index,
        default=1,
        help="OpenCV camera index; Iriun is commonly 1 when a built-in camera is index 0",
    )
    parser.add_argument("--width", type=positive_integer, default=1280)
    parser.add_argument("--height", type=positive_integer, default=720)
    return parser


def configure_console_output() -> None:
    """Keep bilingual diagnostics from crashing legacy Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")


def open_camera(cv2_module: Any, index: int) -> Any:
    """Open a camera, preferring DirectShow on Windows and falling back safely."""
    if os.name == "nt" and hasattr(cv2_module, "CAP_DSHOW"):
        directshow_capture = cv2_module.VideoCapture(index, cv2_module.CAP_DSHOW)
        if directshow_capture.isOpened():
            return directshow_capture
        directshow_capture.release()
    return cv2_module.VideoCapture(index)


def preview_camera(cv2_module: Any, *, index: int, width: int, height: int) -> int:
    """Display frames until q/Escape and always release the camera resources."""
    capture = open_camera(cv2_module, index)
    try:
        if not capture.isOpened():
            print(CONNECTION_ERROR, file=sys.stderr)
            return 1

        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, height)
        print("External camera connected. Press q or Escape to close the preview.")
        print("تم الاتصال بالكاميرا الخارجية. اضغط q أو Escape لإغلاق المعاينة.")

        while True:
            received, frame = capture.read()
            if not received:
                print(CONNECTION_ERROR, file=sys.stderr)
                return 1

            # This is intentionally a camera-only diagnostic. PHANTOM model inference
            # continues through the consent-controlled browser application.
            cv2_module.imshow(WINDOW_TITLE, frame)
            key = cv2_module.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                return 0
    finally:
        capture.release()
        cv2_module.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the external-camera preview command."""
    configure_console_output()
    args = build_parser().parse_args(argv)
    try:
        import cv2
    except ImportError:
        print(
            "OpenCV is required. Install the PHANTOM realtime dependencies first.\n"
            "مكتبة OpenCV مطلوبة. ثبّت متطلبات PHANTOM realtime أولاً.",
            file=sys.stderr,
        )
        return 2
    return preview_camera(
        cv2,
        index=args.camera_index,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    raise SystemExit(main())
