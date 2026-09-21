"""Index precomputed out/ results for fast demo fallback."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def build_index() -> dict:
    by_image: dict[str, str] = {}
    results = []
    for d in sorted(OUT.iterdir()):
        report = d / "report.json"
        if not (d.is_dir() and report.exists()):
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        image = data.get("image")
        rid = data.get("result_id", d.name)
        if image and image not in by_image:
            by_image[image] = rid
        results.append(
            {
                "result_id": rid,
                "image": image,
                "detection_count": data.get("detection_count"),
                "valid": (data.get("qa") or {}).get("valid"),
                "compliance": (data.get("planogram") or {}).get("coarse_compliance_pct"),
            }
        )
    payload = {"by_image": by_image, "results": results}
    (OUT / "index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    idx = build_index()
    print(f"Indexed {len(idx['results'])} results → out/index.json")
