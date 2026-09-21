"""Detect horizontal shelf bands from the image (before product detection)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ShelfBand:
    y1: int
    y2: int

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)


def _horizontal_edge_profile(gray: np.ndarray) -> np.ndarray:
    """Row-wise strength of horizontal structure (shelf lips / product tops)."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Emphasize horizontal edges
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(sobel_y)
    # Ignore left/right door frames
    x0, x1 = int(0.08 * w), int(0.92 * w)
    profile = mag[:, x0:x1].mean(axis=1)
    # Smooth
    k = max(5, (h // 80) | 1)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).ravel()
    return profile.astype(np.float64)


def _find_shelf_separators(profile: np.ndarray, h: int) -> List[int]:
    """Peak rows that likely separate shelves."""
    if h < 80:
        return []
    # Local peaks above adaptive threshold
    thr = float(np.percentile(profile, 75))
    min_dist = max(20, h // 14)
    peaks: List[int] = []
    for i in range(1, h - 1):
        if profile[i] < thr:
            continue
        if profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            if peaks and i - peaks[-1] < min_dist:
                if profile[i] > profile[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    # Drop peaks too close to top/bottom chrome
    return [p for p in peaks if 0.06 * h < p < 0.94 * h]


def estimate_shelf_bands(
    image_bgr: np.ndarray,
    *,
    max_bands: int = 5,
    min_bands: int = 3,
) -> List[ShelfBand]:
    """
    Split fridge/shelf photo into horizontal bands (product rows).
    Falls back to uniform slices if edge peaks are weak.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    profile = _horizontal_edge_profile(gray)
    seps = _find_shelf_separators(profile, h)

    bounds = [0] + seps + [h]
    merged = [bounds[0]]
    min_band = max(60, int(0.11 * h))
    for b in bounds[1:]:
        if b - merged[-1] < min_band:
            continue
        merged.append(b)
    if merged[-1] != h:
        if h - merged[-1] < min_band and len(merged) > 1:
            merged[-1] = h
        else:
            merged.append(h)

    bands = [ShelfBand(int(merged[i]), int(merged[i + 1])) for i in range(len(merged) - 1)]
    bands = [b for b in bands if b.height >= min_band]

    if len(bands) < min_bands:
        n = min(max_bands, max(min_bands, int(round(h / max(min_band, 1)))))
        n = max(min_bands, min(n, 5))
        step = h / n
        bands = [ShelfBand(int(i * step), int((i + 1) * step)) for i in range(n)]
        bands[-1] = ShelfBand(bands[-1].y1, h)

    if len(bands) > max_bands:
        while len(bands) > max_bands:
            heights = [b.height for b in bands]
            i = int(np.argmin(heights[:-1] if len(heights) > 1 else heights))
            if i >= len(bands) - 1:
                i = len(bands) - 2
            bands[i] = ShelfBand(bands[i].y1, bands[i + 1].y2)
            del bands[i + 1]

    return bands


def padded_band_crop(
    image_bgr: np.ndarray,
    band: ShelfBand,
    *,
    pad_ratio: float = 0.12,
) -> Tuple[np.ndarray, int, int]:
    """Return crop, y_offset into full image."""
    h, w = image_bgr.shape[:2]
    pad = int(pad_ratio * band.height)
    y1 = max(0, band.y1 - pad)
    y2 = min(h, band.y2 + pad)
    return image_bgr[y1:y2, :], y1, 0


def expand_bands_for_overlap(bands: Sequence[ShelfBand], h: int, overlap: float = 0.08) -> List[ShelfBand]:
    """Slight vertical overlap so products on the lip are not cut."""
    out: List[ShelfBand] = []
    for b in bands:
        pad = int(overlap * b.height)
        out.append(ShelfBand(max(0, b.y1 - pad), min(h, b.y2 + pad)))
    return out
