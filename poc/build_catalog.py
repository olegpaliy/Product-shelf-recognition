"""Create simple brand reference images for the PoC catalog."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# BGR color themes approximating beverage packaging
BRANDS = {
    "coca-cola": [(0, 0, 180), (20, 20, 220), (40, 40, 160)],
    "fanta": [(0, 140, 255), (0, 165, 255), (30, 180, 255)],
    "sprite": [(60, 180, 60), (40, 160, 40), (80, 200, 100)],
    "pepsi": [(180, 60, 20), (200, 40, 40), (160, 80, 30)],
    "bonaqua": [(240, 240, 240), (220, 220, 230), (200, 210, 220)],
    "schweppes": [(40, 40, 40), (20, 20, 20), (60, 60, 60)],
    "lipton": [(0, 180, 220), (0, 160, 200), (20, 190, 230)],
    "monster": [(0, 220, 80), (20, 200, 40), (0, 180, 60)],
}


def make_ref(colors: list, size: int = 224) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    band = size // len(colors)
    for i, color in enumerate(colors):
        img[i * band : (i + 1) * band, :] = color
    # label-like center stripe
    cv2.rectangle(img, (40, 70), (size - 40, size - 70), (255, 255, 255), -1)
    cv2.rectangle(img, (50, 80), (size - 50, size - 80), colors[0], -1)
    return img


def main() -> None:
    catalog = ROOT / "catalog"
    for brand, colors in BRANDS.items():
        d = catalog / brand
        d.mkdir(parents=True, exist_ok=True)
        for i, shift in enumerate((0, 15, -10)):
            cols = [tuple(int(np.clip(c + shift, 0, 255)) for c in color) for color in colors]
            path = d / f"ref_{i+1}.jpg"
            cv2.imwrite(str(path), make_ref(cols))
            print("wrote", path)


if __name__ == "__main__":
    main()
