"""Group detections into shelf rows by Y coordinate clustering."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .brand import BrandPrediction
from .detect import Detection


def assign_shelf_rows(
    detections: Sequence[Detection],
    *,
    max_rows: int = 8,
    gap_ratio: float = 0.045,
) -> List[int]:
    """
    Assign each detection a 1-based shelf row (1 = top).
    Uses 1D clustering on center-y with adaptive gap.
    """
    if not detections:
        return []

    ys = np.array([d.cy for d in detections], dtype=np.float64)
    order = np.argsort(ys)
    sorted_y = ys[order]
    if len(sorted_y) == 1:
        return [1]

    img_span = float(sorted_y.max() - sorted_y.min()) or 1.0
    min_gap = max(12.0, gap_ratio * img_span)

    # Build clusters greedily top→bottom
    clusters: List[List[int]] = [[int(order[0])]]
    for idx in order[1:]:
        prev = clusters[-1][-1]
        if abs(ys[idx] - ys[prev]) <= min_gap:
            clusters[-1].append(int(idx))
        else:
            clusters.append([int(idx)])

    # Merge excess clusters by smallest gap
    while len(clusters) > max_rows:
        gaps = []
        for i in range(len(clusters) - 1):
            a = np.mean([ys[j] for j in clusters[i]])
            b = np.mean([ys[j] for j in clusters[i + 1]])
            gaps.append((b - a, i))
        _, merge_i = min(gaps, key=lambda t: t[0])
        clusters[merge_i].extend(clusters[merge_i + 1])
        del clusters[merge_i + 1]

    row_of = {}
    for row_i, members in enumerate(clusters, start=1):
        for det_i in members:
            row_of[det_i] = row_i

    return [row_of[i] for i in range(len(detections))]


def rows_brand_map(
    rows: Sequence[int],
    predictions: Sequence[BrandPrediction],
) -> Dict[str, List[str]]:
    """Map shelf_N -> ordered brand labels present on that row."""
    by_row: Dict[int, List[str]] = defaultdict(list)
    for row, pred in zip(rows, predictions):
        by_row[int(row)].append(pred.brand)
    return {f"shelf_{r}": brands for r, brands in sorted(by_row.items())}


def summarize_rows(
    rows: Sequence[int],
    predictions: Sequence[BrandPrediction],
) -> List[dict]:
    by_row: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row, pred in zip(rows, predictions):
        by_row[int(row)][pred.brand] += 1
    out = []
    for row in sorted(by_row):
        counts = dict(by_row[row])
        out.append({"shelf": row, "brands": counts, "total": sum(counts.values())})
    return out
