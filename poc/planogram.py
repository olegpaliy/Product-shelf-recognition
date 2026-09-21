"""Compare detected shelf brands against a demo planogram JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set


def load_planogram(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _unique_ordered(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def compare_planogram(
    planogram: dict,
    rows_brands: Dict[str, List[str]],
    all_detected_brands: Sequence[str],
) -> Dict[str, Any]:
    """
    Coarse compliance:
    - presence: expected brands found anywhere
    - row_match: brand on expected row or ±1
    - missing: expected − detected
    - competitor: detected brands outside own_brands
    """
    expected_rows: Dict[str, List[str]] = planogram.get("shelves", {})
    own_brands = set(planogram.get("own_brands", []))
    if not own_brands:
        own_brands = {b for brands in expected_rows.values() for b in brands}

    expected_all = _unique_ordered([b for brands in expected_rows.values() for b in brands])
    detected_set = {b for b in all_detected_brands if b != "unknown"}

    present = [b for b in expected_all if b in detected_set]
    missing = [b for b in expected_all if b not in detected_set]
    presence_pct = round(100.0 * len(present) / max(1, len(expected_all)), 2)

    if not expected_all:
        return {
            "enabled": False,
            "reason": "empty_planogram",
            "planogram_id": planogram.get("id"),
            "presence_pct": None,
            "row_match_pct": None,
            "coarse_compliance_pct": None,
            "present": [],
            "missing": [],
            "competitor": [],
            "unknown_count": sum(1 for b in all_detected_brands if b == "unknown"),
            "row_matches": [],
            "expected": expected_rows,
            "detected_by_row": {k: _unique_ordered(v) for k, v in rows_brands.items()},
        }

    # Build detected brand -> set of rows
    brand_rows: Dict[str, Set[int]] = {}
    for shelf_key, brands in rows_brands.items():
        try:
            row_num = int(shelf_key.split("_")[-1])
        except ValueError:
            continue
        for b in brands:
            if b == "unknown":
                continue
            brand_rows.setdefault(b, set()).add(row_num)

    expected_row_of: Dict[str, Set[int]] = {}
    for shelf_key, brands in expected_rows.items():
        try:
            row_num = int(shelf_key.split("_")[-1])
        except ValueError:
            continue
        for b in brands:
            expected_row_of.setdefault(b, set()).add(row_num)

    row_matches = []
    row_hits = 0
    row_total = 0
    for brand, exp_rows in expected_row_of.items():
        row_total += 1
        det_rows = brand_rows.get(brand, set())
        ok = False
        matched_row = None
        for er in exp_rows:
            for dr in det_rows:
                if abs(dr - er) <= 1:
                    ok = True
                    matched_row = dr
                    break
            if ok:
                break
        if ok:
            row_hits += 1
        row_matches.append(
            {
                "brand": brand,
                "expected_rows": sorted(exp_rows),
                "detected_rows": sorted(det_rows),
                "ok": ok,
                "matched_row": matched_row,
            }
        )

    row_match_pct = round(100.0 * row_hits / max(1, row_total), 2)
    coarse_compliance = round(0.6 * presence_pct + 0.4 * row_match_pct, 2)

    competitor = sorted(
        b for b in detected_set if b not in own_brands and b != "unknown"
    )
    unknown_count = sum(1 for b in all_detected_brands if b == "unknown")

    return {
        "enabled": True,
        "planogram_id": planogram.get("id", "demo"),
        "presence_pct": presence_pct,
        "row_match_pct": row_match_pct,
        "coarse_compliance_pct": coarse_compliance,
        "present": present,
        "missing": missing,
        "competitor": competitor,
        "unknown_count": unknown_count,
        "row_matches": row_matches,
        "expected": expected_rows,
        "detected_by_row": {k: _unique_ordered(v) for k, v in rows_brands.items()},
    }
