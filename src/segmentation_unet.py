"""Shared logic for the lesion-segmentation notebooks.

This module holds the *real code* behind ``notebooks/01_segmentation.ipynb``
and ``notebooks/02_lesion_focused.ipynb`` so that the notebooks themselves stay
thin (glue + narration + figures). It implements a small from-scratch U-Net for
binary lesion segmentation on ISIC-2018-style data, the Dice loss/metrics used
to train and evaluate it, mask post-processing, and the lesion-on-white
compositing step that turns a raw dermoscopic image into a background-suppressed
("lesion-focused") image.

This is a deliberately self-contained alternative to the DeepLabV3 segmenter in
``src/training/train_ham10000_segmentation_model.py``; it is the pipeline the
notebooks document and is kept separate on purpose.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

IMAGE_SIZE = 256
"""Side length the segmenter operates at; images are resized to this square."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2, the basic U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNet(nn.Module):
    """A compact 3-level U-Net producing a single-channel logit map."""

    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        c3 = self.down3(self.pool2(c2))
        b = self.bottleneck(self.pool3(c3))

        x = self.conv3(torch.cat([self.up3(b), c3], dim=1))
        x = self.conv2(torch.cat([self.up2(x), c2], dim=1))
        x = self.conv1(torch.cat([self.up1(x), c1], dim=1))
        return self.out(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ISICSegmentationDataset(Dataset):
    """ISIC-2018-Task-1-style dataset.

    Expects ``<image_dir>/<name>.jpg`` paired with
    ``<mask_dir>/<name>_segmentation.png``.
    """

    def __init__(self, image_dir: str, mask_dir: str, image_size: int = IMAGE_SIZE):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))

        self.image_transform = T.Compose(
            [T.Resize((image_size, image_size)), T.ToTensor()]
        )
        self.mask_transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size),
                    interpolation=T.InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_files[idx]
        base_name = image_name[: -len(".jpg")]
        mask_name = base_name + "_segmentation.png"

        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")

        image = self.image_transform(image)
        mask = (self.mask_transform(mask) > 0.5).float()
        return image, mask


# ---------------------------------------------------------------------------
# Losses & metrics
# ---------------------------------------------------------------------------
def dice_loss_from_logits(logits: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(logits).view(logits.size(0), -1)
    masks = masks.view(masks.size(0), -1)
    intersection = (probs * masks).sum(dim=1)
    total = probs.sum(dim=1) + masks.sum(dim=1)
    return 1 - ((2 * intersection + eps) / (total + eps)).mean()


def dice_score_from_logits(
    logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
):
    preds = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    masks = masks.view(masks.size(0), -1)
    intersection = (preds * masks).sum(dim=1)
    total = preds.sum(dim=1) + masks.sum(dim=1)
    return ((2 * intersection + eps) / (total + eps)).mean()


def dice_score_binary_numpy(
    pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-6
) -> float:
    pred_mask = pred_mask.astype(np.float32)
    true_mask = true_mask.astype(np.float32)
    intersection = (pred_mask * true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    return float((2 * intersection + eps) / (total + eps))


# ---------------------------------------------------------------------------
# Mask post-processing & lesion-focused compositing
# ---------------------------------------------------------------------------
def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground component and fill holes.

    ``mask`` is a 2D array of 0/1. Returns float32 0/1 of the same shape.
    """
    labelled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask.astype(np.float32)

    component_sizes = ndimage.sum(mask, labelled, range(1, num_features + 1))
    largest_component = int(np.argmax(component_sizes)) + 1
    cleaned = ndimage.binary_fill_holes(labelled == largest_component)
    return cleaned.astype(np.float32)


@torch.no_grad()
def predict_lesion_mask(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run the segmenter on a PIL image and return a cleaned 0/1 mask.

    The returned mask is resized back to the *original* image resolution so it
    can be composited with the full-resolution image.
    """
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    tensor = transform(image).unsqueeze(0).to(device)

    probs = torch.sigmoid(model(tensor))
    raw_mask = (probs > threshold).float()[0, 0].cpu().numpy()
    cleaned = clean_binary_mask(raw_mask)

    mask_img = Image.fromarray((cleaned * 255).astype(np.uint8))
    mask_img = mask_img.resize(image.size, resample=Image.NEAREST)
    return np.array(mask_img).astype(np.float32) / 255.0


def composite_on_white(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Keep lesion pixels, replace background with white."""
    image_np = np.array(image.convert("RGB")).astype(np.uint8)
    mask_3d = mask[:, :, None]
    white = np.full_like(image_np, 255)
    out = (image_np * mask_3d + white * (1 - mask_3d)).astype(np.uint8)
    return Image.fromarray(out)
