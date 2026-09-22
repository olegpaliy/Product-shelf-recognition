"""End-to-end analysis pipeline for one image."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from .brand import BrandMatcher, aggregate_brand_counts, brand_share
from .detect import (
    default_detector_weights,
    detect_products,
    draw_detections,
    maybe_crop_product_roi,
    save_annotated,
)
from .planogram import compare_planogram
from .planogram_select import empty_planogram_result, resolve_planogram
from .qa import assess_photo
from .report import write_excel, write_json
from .shelves import assign_shelf_rows, rows_brand_map, summarize_rows

ROOT = Path(__file__).resolve().parents[1]


def analyze_image(
    image_path: Path,
    output_dir: Path,
    *,
    catalog_dir: Optional[Path] = None,
    planogram_mode: str = "auto",
    expected_brands: Optional[Sequence[str]] = None,
    planogram_path: Optional[Path] = None,
    result_id: Optional[str] = None,
    conf: float = 0.22,
    imgsz: int = 960,
    weights: Optional[str] = None,
    matcher: Optional[BrandMatcher] = None,
) -> Dict[str, Any]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rid = result_id or uuid.uuid4().hex[:12]
    result_dir = output_dir / rid
    result_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    qa = assess_photo(image)
    image, crop_x0 = maybe_crop_product_roi(image, image_path.stem)
    det_weights = weights or default_detector_weights()
    detections = detect_products(image, conf=conf, imgsz=imgsz, weights=det_weights)

    catalog_dir = Path(catalog_dir or ROOT / "catalog")
    if matcher is None:
        matcher = BrandMatcher(catalog_dir)
    predictions = matcher.predict_detections(image, detections)
    labels = [
        f"{p.brand} {p.confidence:.2f}" if p.brand != "unknown" else f"unknown {p.confidence:.2f}"
        for p in predictions
    ]

    rows = assign_shelf_rows(detections)
    row_map = rows_brand_map(rows, predictions)
    shelf_summary = summarize_rows(rows, predictions)
    counts = aggregate_brand_counts(predictions)
    share = brand_share(counts)

    # Legacy CLI path: explicit file still supported
    if planogram_path is not None:
        from .planogram import load_planogram

        path = Path(planogram_path)
        if path.exists():
            resolved = {
                "enabled": True,
                "planogram": load_planogram(path),
                "source": str(path.name),
            }
        else:
            resolved = {"enabled": False, "planogram": None, "source": "missing_file"}
    else:
        resolved = resolve_planogram(
            mode=planogram_mode,
            sample_name=image_path.name,
            expected_brands=expected_brands,
        )

    if resolved.get("enabled") and resolved.get("planogram"):
        plan_cmp = compare_planogram(
            resolved["planogram"],
            row_map,
            [p.brand for p in predictions],
        )
        plan_cmp["source"] = resolved.get("source")
    else:
        plan_cmp = empty_planogram_result(reason=str(resolved.get("source") or "disabled"))
        plan_cmp["source"] = resolved.get("source")

    annotated = draw_detections(image, detections, labels=labels)
    annotated_path = result_dir / "annotated.jpg"
    save_annotated(annotated_path, annotated)

    det_payload = []
    for det, pred, row in zip(detections, predictions, rows):
        item = det.to_dict()
        item["brand"] = pred.brand
        item["brand_confidence"] = round(pred.confidence, 4)
        item["shelf_row"] = row
        det_payload.append(item)

    payload: Dict[str, Any] = {
        "result_id": rid,
        "image": image_path.name,
        "image_path": str(image_path),
        "qa": {
            "valid": qa.valid,
            "reason": qa.reason,
            "blur_score": round(qa.blur_score, 2),
            "brightness": round(qa.brightness, 2),
        },
        "detection_count": len(detections),
        "detector_weights": det_weights,
        "detector_conf": conf,
        "detector_imgsz": imgsz,
        "roi_crop_x0": crop_x0,
        "brand_counts": counts,
        "brand_share": share,
        "shelf_summary": shelf_summary,
        "planogram": plan_cmp,
        "planogram_mode": planogram_mode,
        "detections": det_payload,
        "annotated_path": str(annotated_path),
        "catalog_brands": matcher.brands,
    }

    write_json(result_dir / "report.json", payload)
    write_excel(result_dir / "report.xlsx", payload)
    write_json(result_dir / "meta.json", {"result_id": rid, "image": image_path.name})
    return payload
