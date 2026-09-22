"""Fine-tune SKU-110K YOLO11s on datasets/detect (local Mac CPU/MPS)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .detect import ensure_sku110k_weights

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "detect" / "data.yaml"
RUNS = ROOT / "runs" / "detect"
EXPORT = ROOT / "models" / "sku110k" / "sku110k-finetuned.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune product detector on sample dataset")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")  # Apple Silicon; use cpu if needed
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Missing {args.data}. Run: python -m poc.build_detect_dataset")

    from ultralytics import YOLO

    base = ensure_sku110k_weights()
    model = YOLO(str(base))

    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RUNS),
        name="finetune",
        exist_ok=True,
        patience=8,
        close_mosaic=6,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.05,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.05,
        workers=0,
        verbose=True,
    )
    if args.device:
        kwargs["device"] = args.device

    results = model.train(**kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        # fallback path
        best = RUNS / "finetune" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but best.pt not found under {RUNS}")

    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, EXPORT)
    print(f"exported → {EXPORT}")


if __name__ == "__main__":
    main()
