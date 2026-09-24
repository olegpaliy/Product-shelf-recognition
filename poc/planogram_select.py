"""Resolve which planogram to use for an analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
PLANOGRAMS_DIR = ROOT / "planograms"


def load_planogram_by_id(planogram_id: str) -> Dict[str, Any]:
    path = PLANOGRAMS_DIR / f"{planogram_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Planogram not found: {planogram_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def planogram_from_expected_brands(
    brands: Sequence[str],
    *,
    own_brands: Optional[Sequence[str]] = None,
    planogram_id: str = "custom",
) -> Dict[str, Any]:
    cleaned = []
    seen = set()
    for b in brands:
        name = str(b).strip().lower().replace(" ", "-")
        if not name or name in seen or name == "unknown":
            continue
        seen.add(name)
        cleaned.append(name)
    return {
        "id": planogram_id,
        "name": "Custom expected brands",
        "own_brands": list(own_brands) if own_brands is not None else cleaned,
        "shelves": {"shelf_1": cleaned} if cleaned else {},
        "notes": "Built from expected brands query param",
    }


def empty_planogram_result(reason: str = "no_planogram") -> Dict[str, Any]:
    return {
        "enabled": False,
        "reason": reason,
        "planogram_id": None,
        "presence_pct": None,
        "row_match_pct": None,
        "coarse_compliance_pct": None,
        "present": [],
        "missing": [],
        "competitor": [],
        "unknown_count": 0,
        "row_matches": [],
        "expected": {},
        "detected_by_row": {},
    }


def resolve_planogram(
    *,
    mode: str = "auto",
    sample_name: Optional[str] = None,
    expected_brands: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    mode:
      - auto: planograms/<sample_stem>.json if exists, else none
      - none: disabled
      - custom: from expected_brands list
      - <id>: load planograms/<id>.json
    """
    mode = (mode or "auto").strip().lower()

    if mode == "none":
        return {"enabled": False, "planogram": None, "source": "none"}

    if mode == "custom":
        plan = planogram_from_expected_brands(expected_brands or [])
        if not plan["shelves"]:
            return {"enabled": False, "planogram": plan, "source": "custom_empty"}
        return {"enabled": True, "planogram": plan, "source": "custom"}

    if mode == "auto":
        stem = Path(sample_name).stem if sample_name else None
        if stem:
            candidate = PLANOGRAMS_DIR / f"{stem}.json"
            if candidate.exists():
                plan = json.loads(candidate.read_text(encoding="utf-8"))
                return {"enabled": True, "planogram": plan, "source": f"auto:{stem}"}
        return {"enabled": False, "planogram": None, "source": "auto_none"}

    # explicit id
    plan = load_planogram_by_id(mode)
    if not plan.get("shelves"):
        return {"enabled": False, "planogram": plan, "source": mode}
    return {"enabled": True, "planogram": plan, "source": mode}
