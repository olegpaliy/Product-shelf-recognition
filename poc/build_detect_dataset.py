"""Rebuild detect dataset from manual napoi boxes + cleaned live detections."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .crop_napoi_hero import CROPS as NAPOI_CROPS
from .detect import Detection, detect_products, draw_detections, ensure_sku110k_weights

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUT = ROOT / "datasets" / "detect"
REVIEW = ROOT / "out" / "label_review"

Box = Tuple[float, float, float, float]


def _clip(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Box | None:
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    return x1, y1, x2, y2


def boxes_to_yolo(boxes: Sequence[Box], w: int, h: int) -> List[str]:
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        nw = (x2 - x1) / w
        nh = (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def napoi_boxes(w: int, h: int) -> List[Box]:
    out = []
    for _, x1, y1, x2, y2 in NAPOI_CROPS:
        # slight widen for can/bottle body
        pad = 4
        b = _clip(x1 - pad, y1, x2 + pad, y2, w, h)
        if b:
            out.append(b)
    return out


def dets_to_boxes(dets: List[Detection]) -> List[Box]:
    return [(d.x1, d.y1, d.x2, d.y2) for d in dets]


def crop_fridge_panel(img: np.ndarray) -> Tuple[np.ndarray, int]:
    x0 = int(0.34 * img.shape[1])
    return img[:, x0:], x0


def make_tiles(
    image: np.ndarray, boxes: Sequence[Box], grid=(2, 3), overlap=0.25, min_boxes=2
):
    h, w = image.shape[:2]
    rows, cols = grid
    tile_h = int(h / (rows - overlap * (rows - 1)))
    tile_w = int(w / (cols - overlap * (cols - 1)))
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y1 = int(r * tile_h * (1 - overlap))
            x1 = int(c * tile_w * (1 - overlap))
            y2, x2 = min(h, y1 + tile_h), min(w, x1 + tile_w)
            local = []
            for bx1, by1, bx2, by2 in boxes:
                ix1, iy1 = max(bx1, x1), max(by1, y1)
                ix2, iy2 = min(bx2, x2), min(by2, y2)
                if ix2 - ix1 < 8 or iy2 - iy1 < 8:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                area = (bx2 - bx1) * (by2 - by1)
                if inter / (area + 1e-6) < 0.5:
                    continue
                local.append((ix1 - x1, iy1 - y1, ix2 - x1, iy2 - y1))
            if len(local) >= min_boxes:
                tiles.append((image[y1:y2, x1:x2].copy(), local, f"r{r}c{c}"))
    return tiles


def save_pair(img_dir: Path, lbl_dir: Path, stem: str, image: np.ndarray, boxes: Sequence[Box]):
    h, w = image.shape[:2]
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), image)
    (lbl_dir / f"{stem}.txt").write_text(
        "\n".join(boxes_to_yolo(boxes, w, h)) + ("\n" if boxes else ""), encoding="utf-8"
    )


def main() -> None:
    ensure_sku110k_weights()
    if OUT.exists():
        shutil.rmtree(OUT)
    train_img, train_lbl = OUT / "images" / "train", OUT / "labels" / "train"
    val_img, val_lbl = OUT / "images" / "val", OUT / "labels" / "val"
    for p in (train_img, train_lbl, val_img, val_lbl, REVIEW):
        p.mkdir(parents=True, exist_ok=True)

    sources = []

    hero = cv2.imread(str(SAMPLES / "xo_napoi_hero.jpg"))
    assert hero is not None
    hh, hw = hero.shape[:2]
    sources.append(("xo_napoi_hero", hero, napoi_boxes(hw, hh), "manual"))

    fridge_full = cv2.imread(str(SAMPLES / "xo_fridge_01.jpg"))
    if fridge_full is not None:
        panel, _ = crop_fridge_panel(fridge_full)
        boxes = dets_to_boxes(detect_products(panel, conf=0.14, imgsz=1280))
        sources.append(("xo_fridge_01_panel", panel, boxes, "cleaned"))

    shelf = cv2.imread(str(SAMPLES / "shelf_planogram_02.jpg"))
    if shelf is not None:
        boxes = dets_to_boxes(detect_products(shelf, conf=0.14, imgsz=1280))
        sources.append(("shelf_planogram_02", shelf, boxes, "cleaned"))

    records = []
    meta = []
    for stem, img, boxes, kind in sources:
        records.append((stem, img, boxes))
        w = img.shape[1]
        records.append((f"{stem}_flip", cv2.flip(img, 1), [(w - x2, y1, w - x1, y2) for x1, y1, x2, y2 in boxes]))
        for crop, local, tag in make_tiles(img, boxes):
            records.append((f"{stem}_{tag}", crop, local))
            fw = crop.shape[1]
            records.append(
                (
                    f"{stem}_{tag}_flip",
                    cv2.flip(crop, 1),
                    [(fw - x2, y1, fw - x1, y2) for x1, y1, x2, y2 in local],
                )
            )
        dets = [Detection(x1, y1, x2, y2, 1.0, 0, "product") for x1, y1, x2, y2 in boxes]
        cv2.imwrite(str(REVIEW / f"{stem}_{kind}.jpg"), draw_detections(img, dets))
        meta.append({"stem": stem, "kind": kind, "n_boxes": len(boxes)})

    val_stems = {s[0] for s in sources}
    n_train = n_val = 0
    for stem, img, boxes in records:
        if stem in val_stems:
            save_pair(val_img, val_lbl, stem, img, boxes)
            n_val += 1
        else:
            save_pair(train_img, train_lbl, stem, img, boxes)
            n_train += 1

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n",
        encoding="utf-8",
    )
    (OUT / "meta.json").write_text(
        json.dumps({"sources": meta, "train": n_train, "val": n_val}, indent=2),
        encoding="utf-8",
    )
    print(f"dataset train={n_train} val={n_val}")
    for m in meta:
        print(f"  {m['stem']}: {m['n_boxes']} ({m['kind']})")


if __name__ == "__main__":
    main()
