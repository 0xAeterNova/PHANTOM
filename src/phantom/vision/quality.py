"""Image and face-crop quality assessment for conservative abstention."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageQuality:
    overall: float
    brightness: float
    contrast: float
    sharpness: float
    clipping_fraction: float
    resolution_score: float
    sufficient: bool
    reason: str | None = None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def assess_image_quality(
    image: Any,
    *,
    minimum_side: int = 64,
    quality_floor: float = 0.25,
) -> ImageQuality:
    """Estimate exposure, contrast, blur, clipping, and usable resolution."""

    try:
        np = import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("NumPy is required for image quality assessment") from exc
    array = np.asarray(image)
    if array.ndim not in {2, 3} or array.size == 0:
        return ImageQuality(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False, "empty image")
    if array.ndim == 3:
        rgb = array[:, :, :3].astype(np.float32)
        if float(np.max(rgb, initial=0.0)) <= 1.0:
            rgb *= 255.0
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    else:
        gray = array.astype(np.float32)
        if float(np.max(gray, initial=0.0)) <= 1.0:
            gray *= 255.0
    gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)
    mean = float(np.mean(gray)) / 255.0
    std = float(np.std(gray)) / 64.0
    contrast_score = _clamp(std)
    exposure_score = _clamp(1.0 - abs(mean - 0.5) / 0.45)
    clipping = float(np.mean((gray <= 3.0) | (gray >= 252.0)))
    clipping_score = _clamp(1.0 - clipping / 0.65)
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        gradient_x = np.abs(np.diff(gray, axis=1))
        gradient_y = np.abs(np.diff(gray, axis=0))
        gradient = (float(np.mean(gradient_x)) + float(np.mean(gradient_y))) / 2.0
    else:
        gradient = 0.0
    sharpness_score = _clamp(gradient / 12.0)
    shortest = min(int(gray.shape[0]), int(gray.shape[1]))
    resolution_score = _clamp(shortest / float(minimum_side))
    overall = (
        0.20 * exposure_score
        + 0.20 * contrast_score
        + 0.30 * sharpness_score
        + 0.15 * clipping_score
        + 0.15 * resolution_score
    )
    reason: str | None = None
    if shortest < minimum_side // 2:
        reason = "image resolution is too low"
    elif mean < 0.04:
        reason = "image is too dark"
    elif mean > 0.96:
        reason = "image is too bright"
    elif sharpness_score < 0.04 and contrast_score < 0.08:
        reason = "image lacks usable detail"
    elif overall < quality_floor:
        reason = "image quality is insufficient"
    return ImageQuality(
        overall=_clamp(overall),
        brightness=_clamp(mean),
        contrast=contrast_score,
        sharpness=sharpness_score,
        clipping_fraction=clipping,
        resolution_score=resolution_score,
        sufficient=reason is None,
        reason=reason,
    )


estimate_image_quality = assess_image_quality
