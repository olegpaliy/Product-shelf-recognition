"""Brand matching via Hugging Face SigLIP2 (image catalog + text prototypes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .detect import Detection


@dataclass
class BrandPrediction:
    brand: str
    confidence: float


TEXT_PROMPTS: Dict[str, List[str]] = {
    "coca-cola": ["a Coca-Cola red soda can", "Coca-Cola soft drink can"],
    "pepsi": ["a Pepsi blue soda can", "Pepsi soft drink can"],
    "fanta": ["an orange Fanta soda can", "Fanta orange soft drink"],
    "sprite": ["a green Sprite lemon-lime soda can", "Sprite soft drink can"],
    "schweppes": ["a Schweppes tonic water can", "Schweppes soft drink can"],
    "monster": ["a Monster Energy drink can with green claw logo"],
    "red-bull": ["a blue and silver Red Bull energy drink can"],
    "morshynska": [
        "Morshynska Ukrainian mineral water bottle with red Cyrillic моршинська text",
        "моршинська clear PET water bottle red brand name blue pine trees logo",
        "Morshynska glass bottle vertical red label silver cap",
        "Morshynska Sport water bottle red sport cap athlete label",
        "моршинська сильногазована or слабогазована or негазована mineral water",
    ],
    "luzhanska": [
        "dark blue tinted Luzhanska Ukrainian mineral water plastic bottle",
        "лужанська-7 dark blue PET bottle mountain label red Cyrillic text",
    ],
    "san-benedetto": [
        "San Benedetto Italian mineral water clear plastic bottle pink cap",
        "SAN BENEDETTO light blue label with bird logo Product of Italy",
    ],
    "voda": ["clear water bottle with vertical white VODA text label"],
}

DEFAULT_MODEL = "google/siglip2-so400m-patch16-256"


def _as_feature_tensor(feats: Any) -> torch.Tensor:
    """Normalize get_*_features output across transformers versions."""
    if torch.is_tensor(feats):
        return feats
    for key in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        val = getattr(feats, key, None)
        if val is None and isinstance(feats, dict):
            val = feats.get(key)
        if torch.is_tensor(val):
            if key == "last_hidden_state":
                return val[:, 0]
            return val
    raise TypeError(f"Unsupported features type: {type(feats)}")


class BrandMatcher:
    """Match product crops to brands with SigLIP2 image+text scores."""

    def __init__(
        self,
        catalog_dir: Path,
        *,
        threshold: float = 0.16,
        margin: float = 0.025,
        device: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        image_weight: float = 0.75,
        text_weight: float = 0.25,
        batch_size: int = 16,
    ) -> None:
        self.catalog_dir = Path(catalog_dir)
        self.threshold = threshold
        self.margin = margin
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.batch_size = max(1, int(batch_size))
        self.model_id = model_id
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained(model_id)
        # Avoid accelerate meta-init (.to(mps) fails with "copy out of meta tensor").
        self.model = AutoModel.from_pretrained(
            model_id, low_cpu_mem_usage=False
        ).eval()
        self.model.to(self.device)

        self.brand_embeddings: Dict[str, np.ndarray] = {}
        self.text_embeddings: Dict[str, np.ndarray] = {}
        self._load_text_prototypes()
        self._load_catalog()

    @property
    def embed_dim(self) -> int:
        cfg = getattr(self.model, "config", None)
        dim = getattr(cfg, "projection_dim", None) or getattr(cfg, "hidden_size", None)
        return int(dim or 1152)

    def _embed_pils(self, images: List[Image.Image]) -> np.ndarray:
        """Embed PIL images in batches; returns (N, D) float32 L2-normalized."""
        if not images:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        chunks: List[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(images), bs):
                batch = images[i : i + bs]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = _as_feature_tensor(self.model.get_image_features(**inputs))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                chunks.append(feats.float().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def _embed_pil(self, image: Image.Image) -> np.ndarray:
        return self._embed_pils([image])[0]

    def _embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self._embed_pil(Image.fromarray(rgb))

    def _embed_bgrs(self, images_bgr: List[Optional[np.ndarray]]) -> np.ndarray:
        """Embed BGR crops; empty/tiny slots get zero vectors without a model call."""
        n = len(images_bgr)
        out = np.zeros((n, self.embed_dim), dtype=np.float32)
        pil_batch: List[Image.Image] = []
        indices: List[int] = []
        for i, image_bgr in enumerate(images_bgr):
            if image_bgr is None or image_bgr.size == 0:
                continue
            h, w = image_bgr.shape[:2]
            if h < 12 or w < 8:
                continue
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_batch.append(Image.fromarray(rgb))
            indices.append(i)
        if pil_batch:
            embs = self._embed_pils(pil_batch)
            for row, idx in enumerate(indices):
                out[idx] = embs[row]
        return out

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = _as_feature_tensor(self.model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy().astype(np.float32)

    def _load_text_prototypes(self) -> None:
        for brand, prompts in TEXT_PROMPTS.items():
            embs = self._embed_texts(prompts)
            mean = embs.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-8)
            self.text_embeddings[brand] = mean.astype(np.float32)

    def _load_catalog(self) -> None:
        if not self.catalog_dir.exists():
            return
        # Collect all catalog images, embed in one batched pass, group by brand.
        brand_indices: Dict[str, List[int]] = {}
        pils: List[Image.Image] = []
        for brand_dir in sorted(self.catalog_dir.iterdir()):
            if not brand_dir.is_dir() or brand_dir.name.startswith("."):
                continue
            for img_path in sorted(brand_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                brand_indices.setdefault(brand_dir.name, []).append(len(pils))
                pils.append(Image.fromarray(rgb))
        if not pils:
            return
        embs = self._embed_pils(pils)
        for brand, idxs in brand_indices.items():
            self.brand_embeddings[brand] = embs[idxs]

    @property
    def brands(self) -> List[str]:
        return sorted(set(self.brand_embeddings) | set(self.text_embeddings))

    def _looks_dark_blue_bottle(self, crop_bgr: np.ndarray) -> bool:
        """Luzhanska facings are blue-tinted PET; clear Morshynska should not match."""
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # OpenCV hue: blue ~100–130
        blue = (h >= 95) & (h <= 135) & (s >= 35) & (v <= 200)
        return float(blue.mean()) >= 0.06

    def _score_embedding(
        self,
        emb: np.ndarray,
        *,
        crop_bgr: Optional[np.ndarray] = None,
    ) -> BrandPrediction:
        brands = self.brands
        if not brands:
            return BrandPrediction("unknown", 0.0)
        if emb is None or emb.size == 0 or float(np.linalg.norm(emb)) < 1e-8:
            return BrandPrediction("unknown", 0.0)

        scores: Dict[str, float] = {}
        img_scores: Dict[str, float] = {}
        for brand in brands:
            parts: List[float] = []
            weights: List[float] = []
            refs = self.brand_embeddings.get(brand)
            if refs is not None:
                img_s = float(np.max(refs @ emb))
                img_scores[brand] = img_s
                parts.append(img_s)
                weights.append(self.image_weight)
            text = self.text_embeddings.get(brand)
            if text is not None:
                parts.append(float(np.dot(text, emb)))
                weights.append(self.text_weight if refs is not None else 1.0)
            if not parts:
                continue
            wsum = sum(weights) or 1.0
            scores[brand] = sum(p * wt for p, wt in zip(parts, weights)) / wsum

        if not scores:
            return BrandPrediction("unknown", 0.0)

        # Clear water bottles must not become Luzhanska.
        if (
            "luzhanska" in scores
            and crop_bgr is not None
            and not self._looks_dark_blue_bottle(crop_bgr)
        ):
            scores.pop("luzhanska", None)
            if not scores:
                return BrandPrediction("unknown", 0.0)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_brand, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        # Extra bar for luzhanska: need a strong visual match to catalog blues.
        if best_brand == "luzhanska":
            img_s = img_scores.get("luzhanska", 0.0)
            if img_s < 0.45 or (best_score - second) < max(self.margin, 0.03):
                return BrandPrediction("unknown", float(best_score))
        if best_score < self.threshold or (best_score - second) < self.margin:
            return BrandPrediction("unknown", float(best_score))
        return BrandPrediction(best_brand, float(best_score))

    def predict_crop(self, crop_bgr: np.ndarray) -> BrandPrediction:
        if crop_bgr is None or crop_bgr.size == 0:
            return BrandPrediction("unknown", 0.0)
        h, w = crop_bgr.shape[:2]
        if h < 12 or w < 8:
            return BrandPrediction("unknown", 0.0)
        emb = self._embed_bgr(crop_bgr)
        return self._score_embedding(emb, crop_bgr=crop_bgr)

    def predict_detections(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
    ) -> List[BrandPrediction]:
        if not detections:
            return []
        h, w = image_bgr.shape[:2]
        crops: List[Optional[np.ndarray]] = []
        for det in detections:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(w, int(det.x2))
            y2 = min(h, int(det.y2))
            crops.append(image_bgr[y1:y2, x1:x2])
        embs = self._embed_bgrs(crops)
        return [
            self._score_embedding(embs[i], crop_bgr=crops[i])
            for i in range(len(crops))
        ]


def aggregate_brand_counts(predictions: List[BrandPrediction]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in predictions:
        counts[p.brand] = counts.get(p.brand, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def brand_share(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: round(100.0 * v / total, 2) for k, v in counts.items()}
