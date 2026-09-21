"""YOLO product detection (COCO bottle-related classes + fallback)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

# COCO classes that often appear on beverage shelves
RETAIL_CLASS_IDS = {39, 40, 41, 46, 47}  # bottle, wine glass, cup, banana, apple (loose)
RETAIL_CLASS_NAMES = {"bottle", "wine glass", "cup", "can"}


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


def get_model(weights: str = "yolov8n.pt"):
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(weights)
    return _model


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


def detect_products(
    image_bgr: np.ndarray,
    *,
    conf: float = 0.15,
    imgsz: int = 640,
    weights: str = "yolov8n.pt",
    retail_only: bool = True,
) -> List[Detection]:
    """Run YOLO and return product-like boxes; merge column heuristic if sparse."""
    model = get_model(weights)
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
    if retail_only and len(detections) < 5:
        detections = detect_products(
            image_bgr,
            conf=conf,
            imgsz=imgsz,
            weights=weights,
            retail_only=False,
        )
        return detections

    if len(detections) < 12:
        detections = _nms(detections + detect_by_columns(image_bgr), iou_thresh=0.4)

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
