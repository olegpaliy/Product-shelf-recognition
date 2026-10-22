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
    alternatives: Optional[List[Dict[str, Any]]] = None


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
        # Stricter accept: prefer unknown over weak nearest-neighbor matches.
        threshold: float = 0.28,
        margin: float = 0.07,
        sparse_ref_limit: int = 3,
        sparse_image_floor: float = 0.60,
        sparse_margin: float = 0.09,
        single_ref_image_floor: float = 0.72,
        min_image_floor: float = 0.52,
        device: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        image_weight: float = 0.75,
        text_weight: float = 0.25,
        batch_size: int = 16,
    ) -> None:
        self.catalog_dir = Path(catalog_dir)
        self.threshold = threshold
        self.margin = margin
        self.sparse_ref_limit = max(1, int(sparse_ref_limit))
        self.sparse_image_floor = sparse_image_floor
        self.sparse_margin = sparse_margin
        self.single_ref_image_floor = single_ref_image_floor
        self.min_image_floor = min_image_floor
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

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        # Avoid accelerate meta-init (.to() fails with "copy out of meta tensor").
        self.model = AutoModel.from_pretrained(
            model_id,
            low_cpu_mem_usage=False,
            dtype=torch.float32,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

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

    def _catalog_brand_names(self) -> List[str]:
        if not self.catalog_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.catalog_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def _prompts_for_brand(self, brand: str) -> List[str]:
        if brand in TEXT_PROMPTS:
            return TEXT_PROMPTS[brand]
        label = brand.replace("-", " ")
        return [
            f"a photo of {label} product packaging",
            f"{label} brand shelf product",
            label,
        ]

    def _load_text_prototypes(self) -> None:
        # Only brands present in this catalog dir (keeps drinks vs Eva separate).
        brands = self._catalog_brand_names()
        if not brands:
            brands = sorted(TEXT_PROMPTS.keys())
        for brand in brands:
            embs = self._embed_texts(self._prompts_for_brand(brand))
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
                if img_path.name.startswith("."):
                    continue
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

    def reload_catalog(self) -> None:
        """Re-embed catalog images + text prototypes for brands on disk."""
        self.brand_embeddings = {}
        self.text_embeddings = {}
        self._load_text_prototypes()
        self._load_catalog()

    def add_reference(self, brand: str, image_bgr: np.ndarray) -> None:
        """Embed one crop and append to the brand's reference matrix."""
        emb = self._embed_bgrs([image_bgr])[0]
        if float(np.linalg.norm(emb)) < 1e-8:
            raise ValueError("Could not embed reference crop")
        emb = emb.astype(np.float32)
        existing = self.brand_embeddings.get(brand)
        if existing is None:
            self.brand_embeddings[brand] = emb[None, :]
        else:
            self.brand_embeddings[brand] = np.vstack([existing, emb[None, :]])

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
            return BrandPrediction("unknown", 0.0, alternatives=[])
        if emb is None or emb.size == 0 or float(np.linalg.norm(emb)) < 1e-8:
            return BrandPrediction("unknown", 0.0, alternatives=[])

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
            return BrandPrediction("unknown", 0.0, alternatives=[])

        # Clear water bottles must not become Luzhanska.
        if (
            "luzhanska" in scores
            and crop_bgr is not None
            and not self._looks_dark_blue_bottle(crop_bgr)
        ):
            scores.pop("luzhanska", None)
            if not scores:
                return BrandPrediction("unknown", 0.0, alternatives=[])

        ranked_all = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        alternatives = [
            {"brand": b, "score": round(float(s), 4)} for b, s in ranked_all[:3]
        ]

        # Image floor before win/margin: weak few-shot brands must not spoil
        # near-seconds; multi-ref still needs a real visual hit.
        eligible = dict(scores)
        for brand in list(eligible):
            refs = self.brand_embeddings.get(brand)
            n_refs = 0 if refs is None else int(refs.shape[0])
            # Text-only brands (empty catalog folder) cannot win.
            if n_refs <= 0:
                eligible.pop(brand, None)
                continue
            if n_refs == 1:
                floor = self.single_ref_image_floor
            elif n_refs < self.sparse_ref_limit:
                floor = self.sparse_image_floor
            else:
                floor = self.min_image_floor
            if img_scores.get(brand, 0.0) < floor:
                eligible.pop(brand, None)

        if not eligible:
            return BrandPrediction(
                "unknown", float(ranked_all[0][1]), alternatives=alternatives
            )

        ranked = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)
        best_brand, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        gap = best_score - second
        img_s = img_scores.get(best_brand, 0.0)
        refs = self.brand_embeddings.get(best_brand)
        n_refs = 0 if refs is None else int(refs.shape[0])
        sparse = n_refs > 0 and n_refs < self.sparse_ref_limit
        need_margin = self.sparse_margin if sparse else self.margin
        need_score = self.threshold
        if sparse:
            floor = (
                self.single_ref_image_floor
                if n_refs == 1
                else self.sparse_image_floor
            )
            need_score = max(need_score, floor * 0.85)
        # Extra bar for luzhanska: need a strong visual match to catalog blues.
        if best_brand == "luzhanska":
            if img_s < 0.45 or gap < max(need_margin, 0.03):
                return BrandPrediction(
                    "unknown", float(best_score), alternatives=alternatives
                )
        if best_score < need_score or gap < need_margin:
            return BrandPrediction(
                "unknown", float(best_score), alternatives=alternatives
            )
        return BrandPrediction(
            best_brand, float(best_score), alternatives=alternatives
        )

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
