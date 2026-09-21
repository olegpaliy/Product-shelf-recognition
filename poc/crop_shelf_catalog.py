"""Seed catalog refs from shelf_planogram_02 (hand boxes for demo brands)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "samples" / "shelf_planogram_02.jpg"
CATALOG = ROOT / "catalog"

# 1024x616 — one clear facing per brand where possible
CROPS = [
    # shelf 5 (top)
    ("coca-cola", 210, 35, 255, 115),
    ("coca-cola", 260, 35, 305, 115),
    ("monster", 400, 30, 450, 115),
    ("fanta", 560, 25, 610, 110),
    ("pepsi", 655, 25, 705, 110),
    # shelf 4
    ("coca-cola", 200, 140, 255, 230),
    ("fanta", 310, 135, 365, 225),
    ("pepsi", 420, 135, 475, 225),
    ("lipton", 530, 130, 585, 225),
    ("lipton", 590, 130, 645, 225),
    ("sprite", 900, 125, 960, 220),
    # shelf 3
    ("pepsi", 80, 245, 140, 340),
    ("sprite", 360, 240, 420, 335),
    ("fanta", 480, 240, 540, 335),
    ("coca-cola", 700, 240, 760, 335),
    # shelf 2 — large bottles + water
    ("coca-cola", 280, 360, 350, 470),
    ("pepsi", 360, 355, 430, 470),
    ("fanta", 440, 355, 510, 470),
    # shelf 1 bottom
    ("fanta", 40, 490, 115, 600),
    ("sprite", 200, 485, 275, 600),
]


def main() -> None:
    img = cv2.imread(str(IMG))
    if img is None:
        raise SystemExit(f"Missing {IMG}")
    h, w = img.shape[:2]
    written: dict[str, int] = {}
    for i, (brand, x1, y1, x2, y2) in enumerate(CROPS, 1):
        crop = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            continue
        out_dir = CATALOG / brand
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"shelf02_{i:02d}.jpg"
        cv2.imwrite(str(out), crop)
        written[brand] = written.get(brand, 0) + 1
        print("wrote", out)

    manifest_path = CATALOG / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["shelf_planogram_02_manual"] = written
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done", written)


if __name__ == "__main__":
    main()
