"""Generate a tiny *synthetic* sample dataset for the notebooks.

The real datasets (HAM10000, ISIC 2018) require kagglehub credentials and are
multi-GB, so they cannot live in the repo or build on Binder. This script
writes a handful of small synthetic dermoscopy-like images (a coloured blob
"lesion" over a textured "skin" background) plus their ground-truth masks into
``data/sample/``. It is enough to exercise every cell of both notebooks
(training, thresholding, mask post-processing, lesion-focused compositing) in a
few seconds on CPU.

The images are obviously synthetic and are NOT used for any reported result;
they exist purely so the notebooks are runnable by a fresh clone / container.

Run with::

    PYTHONPATH=src python -m data_setup.make_sample_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = REPO_ROOT / "data" / "sample"

N_ISIC = 24
N_HAM = 12
IMG_SIZE = 256
SEED = 42


def _skin_background(rng: np.random.Generator) -> np.ndarray:
    """A warm, mildly textured background standing in for skin."""
    base = np.array(
        [rng.integers(180, 225), rng.integers(140, 180), rng.integers(120, 160)],
        dtype=np.float32,
    )
    noise = rng.normal(0, 10, size=(IMG_SIZE, IMG_SIZE, 1)).astype(np.float32)
    img = np.clip(base[None, None, :] + noise, 0, 255)
    return img


def _make_pair(rng: np.random.Generator) -> tuple[Image.Image, Image.Image]:
    """Return (RGB image, L-mode binary mask) with one elliptical lesion."""
    img = _skin_background(rng)

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cy, cx = rng.integers(90, 166, size=2)
    ry, rx = rng.integers(35, 70, size=2)
    angle = rng.uniform(0, np.pi)

    xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    ellipse = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0

    lesion_color = np.array(
        [rng.integers(70, 130), rng.integers(40, 90), rng.integers(40, 90)],
        dtype=np.float32,
    )
    lesion_noise = rng.normal(0, 14, size=(IMG_SIZE, IMG_SIZE, 1)).astype(np.float32)
    lesion = np.clip(lesion_color[None, None, :] + lesion_noise, 0, 255)
    img[ellipse] = lesion[ellipse]

    image = Image.fromarray(img.astype(np.uint8), mode="RGB")
    mask = Image.fromarray((ellipse * 255).astype(np.uint8), mode="L")
    return image, mask


def main() -> None:
    rng = np.random.default_rng(SEED)

    isic_images = SAMPLE_ROOT / "isic" / "images"
    isic_masks = SAMPLE_ROOT / "isic" / "masks"
    ham_images = SAMPLE_ROOT / "ham10000" / "images"
    for d in (isic_images, isic_masks, ham_images):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(N_ISIC):
        image, mask = _make_pair(rng)
        name = f"ISIC_{i:07d}"
        image.save(isic_images / f"{name}.jpg", quality=90)
        mask.save(isic_masks / f"{name}_segmentation.png")

    for i in range(N_HAM):
        image, _ = _make_pair(rng)
        image.save(ham_images / f"HAM_{i:07d}.jpg", quality=90)

    print(f"Wrote {N_ISIC} ISIC image/mask pairs to {isic_images.parent}")
    print(f"Wrote {N_HAM} HAM-style images to {ham_images}")


if __name__ == "__main__":
    main()
