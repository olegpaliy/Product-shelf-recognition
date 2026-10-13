"""FastAPI app for the shelf recognition PoC."""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .brand import DEFAULT_MODEL as BRAND_MODEL, BrandMatcher
from .detect import default_detector_weights, detect_device
from .pipeline import ROOT, analyze_image
from .report import write_json

SAMPLES_DIR = ROOT / "samples"
OUT_DIR = ROOT / "out"
WEB_DIR = ROOT / "web"
UPLOADS_DIR = OUT_DIR / "uploads"
CATALOG_DIR = ROOT / "catalog"

app = FastAPI(title="Shelf XO Recognition PoC", version="0.7.0")
_matcher: Optional[BrandMatcher] = None
_matcher_lock = threading.Lock()

_BRAND_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def get_matcher() -> BrandMatcher:
    global _matcher
    if _matcher is not None:
        return _matcher
    with _matcher_lock:
        if _matcher is None:
            _matcher = BrandMatcher(CATALOG_DIR)
        return _matcher


def slugify_brand(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug


def _catalog_image_count(brand: str) -> int:
    brand_dir = CATALOG_DIR / brand
    if not brand_dir.is_dir():
        return 0
    return sum(
        1
        for p in brand_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and not p.name.startswith(".")
    )


def _list_catalog_brands() -> List[Dict[str, Any]]:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    disk = {
        p.name
        for p in CATALOG_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    }
    matcher_brands = set(get_matcher().brands)
    names = sorted(disk | matcher_brands)
    return [{"brand": b, "image_count": _catalog_image_count(b)} for b in names]


class CreateBrandBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class CatalogReferenceBody(BaseModel):
    result_id: str
    detection_index: int = Field(..., ge=0)
    brand: str = Field(..., min_length=1, max_length=64)
    action: str = Field(
        default="save",
        description="save | confirm | unknown | not_product",
    )


@app.get("/api/live")
def live():
    return {"ok": True}


@app.get("/api/health")
def health():
    matcher = get_matcher()
    weights = default_detector_weights()
    return {
        "ok": True,
        "version": "0.7.0",
        "detector": "sku110k",
        "detector_weights": weights,
        "detector_device": detect_device(),
        "brand_backend": "siglip2",
        "brand_model": BRAND_MODEL,
        "brand_device": matcher.device,
        "catalog_brands": matcher.brands,
    }


def _is_sample_image(name: str) -> bool:
    """Real demo photos only — skip macOS AppleDouble (._*) and hidden files."""
    if not name or name.startswith("."):
        return False
    return Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


@app.get("/api/samples")
def list_samples():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p.name for p in SAMPLES_DIR.iterdir() if p.is_file() and _is_sample_image(p.name)
    )
    return {"samples": files}


@app.get("/api/samples/{name}")
def get_sample(name: str):
    if not _is_sample_image(name) or "/" in name or "\\" in name:
        raise HTTPException(404, "Sample not found")
    path = SAMPLES_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Sample not found")
    return FileResponse(path)


@app.get("/api/catalog")
def list_catalog():
    return {"brands": _list_catalog_brands()}


@app.post("/api/catalog/brands")
def create_brand(body: CreateBrandBody):
    slug = slugify_brand(body.name)
    if not slug or not _BRAND_SLUG_RE.match(slug):
        raise HTTPException(400, "Invalid brand name (use letters, numbers, hyphens)")
    brand_dir = CATALOG_DIR / slug
    brand_dir.mkdir(parents=True, exist_ok=True)
    return {"brand": slug, "image_count": _catalog_image_count(slug)}


@app.post("/api/catalog/reload")
def reload_catalog():
    matcher = get_matcher()
    matcher.reload_catalog()
    return {"ok": True, "brands": _list_catalog_brands()}


@app.post("/api/catalog/references")
def add_catalog_reference(body: CatalogReferenceBody):
    rid = body.result_id.strip()
    if not rid or "/" in rid or "\\" in rid or ".." in rid:
        raise HTTPException(400, "Invalid result_id")

    report_path = OUT_DIR / rid / "report.json"
    source_path = OUT_DIR / rid / "source.jpg"
    if not report_path.exists() or not source_path.exists():
        raise HTTPException(404, "Result not found (missing report or source)")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    detections = report.get("detections") or []
    idx = body.detection_index
    if idx >= len(detections):
        raise HTTPException(400, "detection_index out of range")

    action = (body.action or "save").strip().lower()
    det = detections[idx]

    if action in {"unknown", "not_product"}:
        label = "unknown" if action == "unknown" else "not_product"
        det["brand"] = label
        det["brand_confidence"] = 0.0
        det["feedback"] = action
        detections[idx] = det
        report["detections"] = detections
        write_json(report_path, report)
        return {
            "ok": True,
            "saved": False,
            "brand": label,
            "detection_index": idx,
        }

    slug = slugify_brand(body.brand)
    if not slug or not _BRAND_SLUG_RE.match(slug):
        raise HTTPException(400, "Invalid brand name")
    if slug in {"unknown", "not-product", "not_product"}:
        raise HTTPException(400, "Use action=unknown/not_product instead")

    image = cv2.imread(str(source_path))
    if image is None:
        raise HTTPException(500, "Cannot read source image")
    h, w = image.shape[:2]
    x1 = max(0, int(det["x1"]))
    y1 = max(0, int(det["y1"]))
    x2 = min(w, int(det["x2"]))
    y2 = min(h, int(det["y2"]))
    if x2 - x1 < 8 or y2 - y1 < 12:
        raise HTTPException(400, "Crop too small")
    crop = image[y1:y2, x1:x2]

    brand_dir = CATALOG_DIR / slug
    brand_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"ref_{rid}_{idx}.jpg"
    out_path = brand_dir / out_name
    if not cv2.imwrite(str(out_path), crop):
        raise HTTPException(500, "Failed to write catalog crop")

    matcher = get_matcher()
    was_new = slug not in matcher.brand_embeddings
    try:
        matcher.add_reference(slug, crop)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if was_new:
        # Ensure disk + matrix stay aligned for brands that only had text before.
        matcher.reload_catalog()

    det["brand"] = slug
    det["brand_confidence"] = float(det.get("brand_confidence") or 1.0)
    det["feedback"] = "save" if action == "save" else action
    detections[idx] = det
    report["detections"] = detections
    report["catalog_brands"] = matcher.brands
    write_json(report_path, report)

    return {
        "ok": True,
        "saved": True,
        "brand": slug,
        "path": str(out_path.relative_to(ROOT)),
        "detection_index": idx,
        "image_count": _catalog_image_count(slug),
    }


@app.post("/api/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Query(None),
    planogram: str = Query("auto"),
    expected: Optional[str] = Query(
        None,
        description="Comma-separated expected brands for planogram=custom",
    ),
):
    if file is None and not sample:
        raise HTTPException(400, "Provide upload file or sample name")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    expected_brands = [x.strip() for x in (expected or "").split(",") if x.strip()]

    if sample:
        src = SAMPLES_DIR / sample
        if not src.exists():
            raise HTTPException(404, f"Sample not found: {sample}")
        image_path = src
    else:
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        image_path = UPLOADS_DIR / f"{rid}{suffix}"
        with image_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

    try:
        payload = analyze_image(
            image_path,
            OUT_DIR,
            result_id=rid,
            matcher=get_matcher(),
            planogram_mode=planogram,
            expected_brands=expected_brands or None,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc

    payload["annotated_url"] = f"/api/results/{rid}/annotated.jpg"
    payload["source_url"] = f"/api/results/{rid}/source.jpg"
    payload["json_url"] = f"/api/results/{rid}/report.json"
    payload["excel_url"] = f"/api/results/{rid}/report.xlsx"
    return payload


@app.get("/api/results/{result_id}/annotated.jpg")
def get_annotated(result_id: str):
    path = OUT_DIR / result_id / "annotated.jpg"
    if not path.exists():
        raise HTTPException(404, "Annotated image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/results/{result_id}/source.jpg")
def get_source(result_id: str):
    if "/" in result_id or "\\" in result_id or ".." in result_id:
        raise HTTPException(404, "Source image not found")
    path = OUT_DIR / result_id / "source.jpg"
    if not path.exists():
        raise HTTPException(404, "Source image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/results/{result_id}/report.json")
def get_json(result_id: str):
    path = OUT_DIR / result_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "JSON report not found")
    return FileResponse(path, media_type="application/json", filename=f"{result_id}.json")


@app.get("/api/results/{result_id}/report.xlsx")
def get_excel(result_id: str):
    path = OUT_DIR / result_id / "report.xlsx"
    if not path.exists():
        raise HTTPException(404, "Excel report not found")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
