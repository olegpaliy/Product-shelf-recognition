"""YOLO product detection — default SKU-110K retail weights + optional fine-tune."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKU110K_DIR = ROOT / "models" / "sku110k"
SKU110K_BASE = SKU110K_DIR / "sku110k-yolo11-s640.pt"
SKU110K_FINETUNED = SKU110K_DIR / "sku110k-finetuned.pt"

# COCO classes that often appear on beverage shelves
RETAIL_CLASS_IDS = {39, 40, 41, 46, 47}  # bottle, wine glass, cup, banana, apple (loose)
RETAIL_CLASS_NAMES = {"bottle", "wine glass", "cup", "can", "object", "product"}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_model = None
_model_weights: Optional[str] = None


def ensure_sku110k_weights() -> Path:
    """Return path to base SKU-110K YOLO11s weights (must already be on disk)."""
    if SKU110K_BASE.exists():
        return SKU110K_BASE
    raise FileNotFoundError(
        f"Missing {SKU110K_BASE}. Place sku110k-yolo11-s640.pt under models/sku110k/"
    )


def default_detector_weights() -> str:
    """Prefer fine-tuned sample weights, else SKU-110K base."""
    if SKU110K_FINETUNED.exists():
        return str(SKU110K_FINETUNED)
    if SKU110K_BASE.exists():
        return str(SKU110K_BASE)
    return "yolov8n.pt"


def get_model(weights: Optional[str] = None):
    global _model, _model_weights
    path = str(weights or default_detector_weights())
    if _model is None or _model_weights != path:
        from ultralytics import YOLO

        _model = YOLO(path)
        _model_weights = path
    return _model


def reset_model() -> None:
    global _model, _model_weights
    _model = None
    _model_weights = None


def crop_fridge_panel(image_bgr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Keep cooler half of marketing-slide samples (UI strip on the left)."""
    x0 = int(0.34 * image_bgr.shape[1])
    return image_bgr[:, x0:], x0


def maybe_crop_product_roi(
    image_bgr: np.ndarray, stem: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    if stem and "fridge" in stem.lower():
        return crop_fridge_panel(image_bgr)
    return image_bgr, 0


def _nms(dets: List[Detection], iou_thresh: float = 0.45) -> List[Detection]:
    if not dets:
        return []
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
    scores = np.array([d.confidence for d in dets], dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return [dets[i] for i in keep]


def detect_by_columns(image_bgr: np.ndarray) -> List[Detection]:
    """
    Heuristic proposals for dense bottle facings when YOLO under-detects.
    Finds tall vertical blobs that look like bottles/cans.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets: List[Detection] = []
    min_h = h * 0.04
    max_h = h * 0.35
    min_w = w * 0.012
    max_w = w * 0.12
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < min_h or bh > max_h or bw < min_w or bw > max_w:
            continue
        aspect = bh / max(bw, 1)
        if aspect < 1.2 or aspect > 6.5:
            continue
        area_ratio = (bw * bh) / float(h * w)
        if area_ratio < 0.0004 or area_ratio > 0.05:
            continue
        dets.append(
            Detection(
                x1=float(x),
                y1=float(y),
                x2=float(x + bw),
                y2=float(y + bh),
                confidence=0.35,
                class_id=-1,
                class_name="product",
            )
        )
    return _nms(dets, iou_thresh=0.35)


def _looks_like_fan_or_fixture(det: Detection, h: int, w: int) -> bool:
    """Reject cooler fan grills / top fixtures mistaken for SKUs."""
    cy_n = det.cy / max(h, 1)
    cx_n = det.cx / max(w, 1)
    aspect = det.height / max(det.width, 1.0)
    width_n = det.width / max(w, 1)
    # Circular/square object high and relatively wide (fan grill)
    if cy_n < 0.14 and 0.55 <= aspect <= 1.55 and width_n >= 0.07:
        return True
    # Tiny lone blob glued to the very top center
    if cy_n < 0.08 and 0.35 <= cx_n <= 0.65 and width_n >= 0.05 and aspect < 1.8:
        return True
    # Very top wide/flat box (fan often boxed as a squat rectangle)
    if cy_n < 0.10 and aspect < 0.85 and width_n >= 0.08 and 0.25 <= cx_n <= 0.75:
        return True
    return False


def _filter_non_products(dets: List[Detection], h: int, w: int) -> List[Detection]:
    return [d for d in dets if not _looks_like_fan_or_fixture(d, h, w)]


def detect_products(
    image_bgr: np.ndarray,
    *,
    conf: float = 0.2,
    imgsz: int = 960,
    weights: Optional[str] = None,
    retail_only: bool = True,
) -> List[Detection]:
    """Run retail detector and return product-like boxes; merge column heuristic if sparse."""
    h, w = image_bgr.shape[:2]
    wpath = weights or default_detector_weights()
    model = get_model(wpath)
    # Single-class retail checkpoints (SKU-110K) — keep all classes
    is_retail_ckpt = "sku110k" in Path(wpath).name.lower()
    results = model.predict(
        source=image_bgr,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    detections: List[Detection] = []
    if results:
        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls.item())
                name = str(names.get(cls_id, cls_id))
                if (
                    retail_only
                    and not is_retail_ckpt
                    and cls_id not in RETAIL_CLASS_IDS
                    and name.lower() not in RETAIL_CLASS_NAMES
                ):
                    continue
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=float(box.conf.item()),
                        class_id=cls_id,
                        class_name=name,
                    )
                )

    # Dense shelves: widen to all COCO classes if too few retail hits
    if retail_only and not is_retail_ckpt and len(detections) < 5:
        return detect_products(
            image_bgr,
            conf=conf,
            imgsz=imgsz,
            weights=wpath,
            retail_only=False,
        )

    detections = _nms(detections, iou_thresh=0.5)

    if len(detections) < 12:
        detections = _nms(detections + detect_by_columns(image_bgr), iou_thresh=0.4)

    detections = _filter_non_products(detections, h, w)
    detections.sort(key=lambda d: (d.cy, d.cx))
    return detections


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Detection],
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw boxes and optional brand labels."""
    out = image_bgr.copy()
    for i, det in enumerate(detections):
        color = (40, 160, 70)
        label = labels[i] if labels and i < len(labels) else det.class_name
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        text = f"{label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text = max(0, p1[1] - 4)
        cv2.rectangle(out, (p1[0], y_text - th - 4), (p1[0] + tw + 4, y_text + 2), color, -1)
        cv2.putText(
            out,
            text,
            (p1[0] + 2, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def save_annotated(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)
