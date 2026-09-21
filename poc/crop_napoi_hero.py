"""Crop brand references from samples/xo_napoi_hero.jpg into catalog/."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "samples" / "xo_napoi_hero.jpg"
CATALOG = ROOT / "catalog"

# Front-facing crops on 682x1024
CROPS = [
    ("red-bull", 155, 95, 235, 305),
    ("red-bull", 215, 100, 295, 310),
    ("monster", 305, 110, 385, 315),
    ("monster", 370, 115, 450, 320),
    ("fanta", 455, 100, 535, 295),
    ("fanta", 520, 95, 600, 290),
    ("fanta", 580, 90, 660, 285),
    ("coca-cola", 155, 385, 245, 595),
    ("coca-cola", 220, 390, 310, 600),
    ("coca-cola", 290, 395, 380, 605),
    ("sprite", 365, 395, 455, 605),
    ("sprite", 430, 395, 520, 605),
    ("schweppes", 500, 390, 590, 600),
    ("schweppes", 565, 385, 655, 595),
    ("schweppes", 620, 380, 680, 590),
    ("morshynska", 150, 620, 250, 930),
    ("morshynska", 230, 620, 330, 930),
    ("morshynska", 310, 620, 410, 930),
    ("morshynska", 390, 620, 490, 930),
    ("voda", 475, 615, 570, 935),
    ("voda", 545, 615, 640, 940),
    ("voda", 615, 615, 680, 940),
    ("juice", 180, 960, 250, 1023),
    ("juice", 250, 960, 320, 1023),
]


def main() -> None:
    img = cv2.imread(str(IMG))
    if img is None:
        raise SystemExit(f"Missing {IMG}")
    h, w = img.shape[:2]
    written = {}
    for i, (brand, x1, y1, x2, y2) in enumerate(CROPS, 1):
        crop = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            continue
        out_dir = CATALOG / brand
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"napoi_hero_{i:02d}.jpg"
        cv2.imwrite(str(out), crop)
        written[brand] = written.get(brand, 0) + 1
        print("wrote", out)

    manifest_path = CATALOG / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["napoi_hero_manual"] = written
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done", written)


if __name__ == "__main__":
    main()
