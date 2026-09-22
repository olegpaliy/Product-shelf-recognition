"""CLI: python -m poc.run --input samples/ --output out/"""

from __future__ import annotations

import argparse
from pathlib import Path

from .brand import BrandMatcher
from .pipeline import ROOT, analyze_image
from .report import write_batch_excel, write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shelf / XO recognition PoC")
    parser.add_argument("--input", type=Path, default=ROOT / "samples")
    parser.add_argument("--output", type=Path, default=ROOT / "out")
    parser.add_argument("--catalog", type=Path, default=ROOT / "catalog")
    parser.add_argument(
        "--planogram",
        type=Path,
        default=None,
        help="Optional explicit planogram JSON. Default: auto per image stem.",
    )
    parser.add_argument(
        "--planogram-mode",
        default="auto",
        help="auto|none|custom|<id> when --planogram file is not set",
    )
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args(argv)

    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    matcher = BrandMatcher(args.catalog)
    results = []
    for img in images:
        print(f"Analyzing {img.name} ...")
        payload = analyze_image(
            img,
            args.output,
            catalog_dir=args.catalog,
            planogram_path=args.planogram,
            planogram_mode=args.planogram_mode,
            conf=args.conf,
            imgsz=args.imgsz,
            matcher=matcher,
        )
        results.append(payload)
        plan = payload["planogram"]
        comp = plan.get("coarse_compliance_pct")
        print(
            f"  -> {payload['result_id']}: dets={payload['detection_count']} "
            f"valid={payload['qa']['valid']} "
            f"plan={plan.get('source')} compliance={comp}%"
        )

    write_json(args.output / "batch.json", {"results": results})
    write_batch_excel(args.output / "batch.xlsx", results)
    print(f"Done. {len(results)} images → {args.output}")


if __name__ == "__main__":
    main()
