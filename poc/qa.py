"""Photo quality checks: blur and brightness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class QAResult:
    valid: bool
    reason: Optional[str]
    blur_score: float
    brightness: float


def assess_photo(
    image_bgr: np.ndarray,
    *,
    min_blur: float = 40.0,
    min_brightness: float = 35.0,
    max_brightness: float = 245.0,
) -> QAResult:
    """Return whether the photo is usable for recognition."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    if blur_score < min_blur:
        return QAResult(False, "blur", blur_score, brightness)
    if brightness < min_brightness:
        return QAResult(False, "too_dark", blur_score, brightness)
    if brightness > max_brightness:
        return QAResult(False, "too_bright", blur_score, brightness)
    return QAResult(True, None, blur_score, brightness)
