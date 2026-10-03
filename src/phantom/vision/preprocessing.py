"""Safe local image decoding, color conversion, cropping, and resizing."""

from __future__ import annotations

import io
from importlib import import_module
from pathlib import Path
from typing import Any

from phantom.exceptions import InvalidMediaError


def _numpy() -> Any:
    try:
        return import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("NumPy is required for vision processing") from exc


def _coerce_rgb_array(image: Any) -> Any:
    np = _numpy()
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise InvalidMediaError("image must have grayscale, RGB, or RGBA channels")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        alpha = array[:, :, 3:4].astype(np.float32)
        if array.dtype.kind in {"f"} and float(np.nanmax(alpha, initial=0.0)) <= 1.0:
            alpha *= 255.0
        rgb = array[:, :, :3].astype(np.float32)
        array = rgb * (alpha / 255.0) + 255.0 * (1.0 - alpha / 255.0)
    if array.dtype.kind == "f":
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.max(array, initial=0.0)) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def decode_image(
    source: bytes | bytearray | memoryview | str | Path | Any,
    *,
    max_pixels: int = 16_000_000,
) -> Any:
    """Decode an image into an RGB uint8 array without retaining source bytes."""

    np = _numpy()
    if isinstance(source, (str, Path)):
        try:
            payload = Path(source).read_bytes()
        except OSError as exc:
            raise InvalidMediaError("image file could not be read") from exc
    elif isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
    else:
        direct_array = _coerce_rgb_array(source)
        if int(direct_array.shape[0]) * int(direct_array.shape[1]) > max_pixels:
            raise InvalidMediaError("decoded image exceeds the pixel limit")
        return direct_array
    if not payload:
        raise InvalidMediaError("image payload is empty")

    array: Any | None = None
    try:
        image_module = import_module("PIL.Image")
    except ModuleNotFoundError:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError("image decoding requires Pillow or OpenCV") from exc
        encoded = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise InvalidMediaError("image is malformed or uses an unsupported codec") from None
        if int(decoded.shape[0]) * int(decoded.shape[1]) > max_pixels:
            raise InvalidMediaError("decoded image exceeds the pixel limit") from None
        array = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    else:
        try:
            with image_module.open(io.BytesIO(payload)) as opened:
                width, height = opened.size
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise InvalidMediaError("decoded image exceeds the pixel limit")
                opened.load()
                array = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        except OSError as exc:
            raise InvalidMediaError("image is malformed or uses an unsupported codec") from exc
    return _coerce_rgb_array(array)


def crop_image(image: Any, box: tuple[int, int, int, int], *, padding: float = 0.10) -> Any:
    """Crop an ``(x, y, width, height)`` box, clipped to image boundaries."""

    if padding < 0.0:
        raise ValueError("padding cannot be negative")
    array = _coerce_rgb_array(image)
    height, width = array.shape[:2]
    x, y, box_width, box_height = (int(value) for value in box)
    if box_width <= 0 or box_height <= 0:
        raise InvalidMediaError("face box has non-positive dimensions")
    pad_x = round(box_width * padding)
    pad_y = round(box_height * padding)
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(width, x + box_width + pad_x), min(height, y + box_height + pad_y)
    if right <= left or bottom <= top:
        raise InvalidMediaError("face box lies outside the image")
    return array[top:bottom, left:right].copy()


def resize_image(image: Any, size: tuple[int, int] = (224, 224)) -> Any:
    """Resize RGB input with deterministic nearest-neighbor sampling."""

    target_height, target_width = (int(value) for value in size)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target dimensions must be positive")
    np = _numpy()
    array = _coerce_rgb_array(image)
    source_height, source_width = array.shape[:2]
    y_indices = np.minimum(
        (np.arange(target_height, dtype=np.float64) * source_height / target_height).astype(int),
        source_height - 1,
    )
    x_indices = np.minimum(
        (np.arange(target_width, dtype=np.float64) * source_width / target_width).astype(int),
        source_width - 1,
    )
    return array[y_indices[:, None], x_indices[None, :]].copy()


def preprocess_image(
    image: Any,
    *,
    size: tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> Any:
    """Resize an image and optionally return float RGB values in ``[0, 1]``."""

    np = _numpy()
    resized = resize_image(image, size)
    if normalize:
        return (resized.astype(np.float32) / 255.0).astype(np.float32, copy=False)
    return resized


# The vision model always receives a face crop, but this alias is convenient in tests.
preprocess_face = preprocess_image
