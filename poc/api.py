"""FastAPI app for the shelf recognition PoC."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .brand import BrandMatcher
from .detect import default_detector_weights
from .pipeline import ROOT, analyze_image
from .planogram_select import list_planograms

SAMPLES_DIR = ROOT / "samples"
OUT_DIR = ROOT / "out"
WEB_DIR = ROOT / "web"
UPLOADS_DIR = OUT_DIR / "uploads"

app = FastAPI(title="Shelf XO Recognition PoC", version="0.6.0")
_matcher: Optional[BrandMatcher] = None


def get_matcher() -> BrandMatcher:
    global _matcher
    if _matcher is None:
        _matcher = BrandMatcher(ROOT / "catalog")
    return _matcher


def reset_matcher() -> None:
    """Force reload after catalog refresh."""
    global _matcher
    _matcher = None


@app.get("/api/health")
def health():
    weights = default_detector_weights()
    return {
        "ok": True,
        "version": "0.6.0",
        "detector": "sku110k",
        "detector_weights": weights,
        "brand_backend": "siglip2",
        "brand_model": "google/siglip2-base-patch16-224",
        "catalog_brands": get_matcher().brands,
    }


@app.get("/api/planograms")
def api_list_planograms():
    return {"planograms": list_planograms()}


@app.get("/api/samples")
def list_samples():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p.name
        for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return {"samples": files}


@app.get("/api/samples/{name}")
def get_sample(name: str):
    path = SAMPLES_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Sample not found")
    return FileResponse(path)


@app.post("/api/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Query(None),
    use_cache: bool = Query(False),
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
        # Cache only when planogram mode is default auto and no custom expected
        if use_cache and planogram == "auto" and not expected_brands:
            index_path = OUT_DIR / "index.json"
            if index_path.exists():
                import json

                index = json.loads(index_path.read_text(encoding="utf-8"))
                cached_id = (index.get("by_image") or {}).get(sample)
                cached_report = OUT_DIR / cached_id / "report.json" if cached_id else None
                if cached_report and cached_report.exists():
                    payload = json.loads(cached_report.read_text(encoding="utf-8"))
                    payload["cached"] = True
                    payload["annotated_url"] = f"/api/results/{cached_id}/annotated.jpg"
                    payload["json_url"] = f"/api/results/{cached_id}/report.json"
                    payload["excel_url"] = f"/api/results/{cached_id}/report.xlsx"
                    return payload
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

    payload["cached"] = False
    payload["annotated_url"] = f"/api/results/{rid}/annotated.jpg"
    payload["json_url"] = f"/api/results/{rid}/report.json"
    payload["excel_url"] = f"/api/results/{rid}/report.xlsx"
    return payload


@app.get("/api/results/{result_id}")
def get_result(result_id: str):
    path = OUT_DIR / result_id / "report.json"
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(path, media_type="application/json")


@app.get("/api/results/{result_id}/annotated.jpg")
def get_annotated(result_id: str):
    path = OUT_DIR / result_id / "annotated.jpg"
    if not path.exists():
        raise HTTPException(404, "Annotated image not found")
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


@app.get("/api/export.xlsx")
def get_batch_excel():
    path = OUT_DIR / "batch.xlsx"
    if not path.exists():
        raise HTTPException(404, "Batch Excel not found — run CLI batch first")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="batch.xlsx",
    )


@app.get("/api/results")
def list_results():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for d in sorted(OUT_DIR.iterdir()):
        report = d / "report.json"
        if d.is_dir() and report.exists():
            items.append(
                {
                    "result_id": d.name,
                    "json_url": f"/api/results/{d.name}/report.json",
                    "annotated_url": f"/api/results/{d.name}/annotated.jpg",
                    "excel_url": f"/api/results/{d.name}/report.xlsx",
                }
            )
    return {"results": items}


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
