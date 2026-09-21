"""
Auto-seed catalog from sample detections using OpenCLIP text prompts.
Keeps only high-confidence crops per brand.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from .detect import detect_products

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"
SAMPLES = ROOT / "samples"

BRAND_PROMPTS = {
    "coca-cola": [
        "a Coca-Cola red soda bottle or can",
        "a red Coca-Cola soft drink package",
    ],
    "pepsi": [
        "a Pepsi blue soda bottle or can",
        "a blue Pepsi soft drink package",
    ],
    "fanta": [
        "an orange Fanta soda bottle or can",
        "a Fanta orange soft drink package",
    ],
    "sprite": [
        "a green Sprite lemon-lime soda bottle or can",
        "a Sprite soft drink package",
    ],
    "bonaqua": [
        "a clear plastic mineral water bottle",
        "a transparent water bottle with blue white label",
        "Bonaqua bottled water",
    ],
    "schweppes": [
        "a Schweppes tonic water bottle or can",
        "a dark labeled Schweppes soft drink",
    ],
    "lipton": [
        "a yellow Lipton iced tea bottle or can",
        "Lipton ice tea beverage package",
    ],
    "monster": [
        "a Monster Energy drink can with claw logo",
        "green black Monster energy can",
    ],
    "red-bull": [
        "a blue and silver Red Bull energy drink can",
        "Red Bull beverage can",
    ],
    "morshynska": [
        "a Morshynska mineral water bottle from Ukraine",
        "clear water bottle Ukrainian label Morshynska",
    ],
}


def main(
    *,
    per_brand: int = 10,
    min_score: float = 0.20,
    margin: float = 0.015,
    conf: float = 0.15,
    imgsz: int = 640,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    text_embs = {}
    with torch.no_grad():
        for brand, prompts in BRAND_PROMPTS.items():
            tokens = tokenizer(prompts).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            text_embs[brand] = emb.mean(dim=0)
            text_embs[brand] = text_embs[brand] / text_embs[brand].norm()

    brand_names = list(text_embs.keys())
    text_matrix = torch.stack([text_embs[b] for b in brand_names], dim=0)  # B,D

    # wipe old synthetic refs; keep directory structure fresh
    if CATALOG.exists():
        for brand_dir in list(CATALOG.iterdir()):
            if brand_dir.is_dir():
                shutil.rmtree(brand_dir)
            elif brand_dir.name == "manifest.json":
                brand_dir.unlink()

    buckets: dict[str, list[tuple[float, np.ndarray, str]]] = defaultdict(list)
    # also keep per-brand raw top scores even when margin fails (fill sparse brands)
    soft: dict[str, list[tuple[float, np.ndarray, str]]] = defaultdict(list)

    images = sorted(
        p
        for p in SAMPLES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "blur" not in p.name
        and "dark" not in p.name
    )

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        dets = detect_products(image, conf=conf, imgsz=imgsz)
        h, w = image.shape[:2]
        print(f"{img_path.name}: {len(dets)} detections")
        for i, det in enumerate(dets):
            x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
            x2, y2 = min(w, int(det.x2)), min(h, int(det.y2))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 10:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            sims = (text_matrix @ emb.squeeze(0)).float().cpu().numpy()
            order = np.argsort(-sims)
            best_i, second_i = int(order[0]), int(order[1])
            best_score = float(sims[best_i])
            second_score = float(sims[second_i])
            brand = brand_names[best_i]
            tag = f"{img_path.stem}_{i}"
            soft[brand].append((best_score, crop.copy(), tag))
            if best_score < min_score or (best_score - second_score) < margin:
                continue
            buckets[brand].append((best_score, crop.copy(), tag))

    # Fill sparse brands from soft candidates
    for brand, items in soft.items():
        if len(buckets[brand]) >= 3:
            continue
        items.sort(key=lambda t: t[0], reverse=True)
        for score, crop, tag in items:
            if score < 0.17:
                break
            if any(t[2] == tag for t in buckets[brand]):
                continue
            buckets[brand].append((score, crop, tag))
            if len(buckets[brand]) >= 5:
                break

    written = {}
    for brand, items in buckets.items():
        items.sort(key=lambda t: t[0], reverse=True)
        out_dir = CATALOG / brand
        out_dir.mkdir(parents=True, exist_ok=True)
        keep = items[:per_brand]
        for rank, (score, crop, tag) in enumerate(keep, start=1):
            path = out_dir / f"{tag}_{rank}.jpg"
            cv2.imwrite(str(path), crop)
        written[brand] = len(keep)
        print(f"{brand}: kept {len(keep)} (from soft={len(soft[brand])}) top={keep[0][0]:.3f}")

    manifest = {
        "method": "openclip_text_autoseed",
        "min_score": min_score,
        "margin": margin,
        "per_brand": per_brand,
        "brands": written,
    }
    (CATALOG / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest", written)


if __name__ == "__main__":
    main()
