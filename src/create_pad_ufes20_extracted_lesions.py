import os
import shutil
import zipfile

import kagglehub
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch import nn
from torchvision import transforms


# -----------------------
# Global config
# -----------------------

os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"

DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")

DATASET_DIR = os.path.join(DATA_ROOT, "datasets", "pad-ufes-20")
MODEL_PATH = os.path.join(RUN_DIR, "models", "segmentation_unet_v2.pth")
OUTPUT_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-extracted-lesions")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

segmentation_image_size = 256
mask_threshold = 0.5


# -----------------------
# Model
# -----------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)


# -----------------------
# Helpers
# -----------------------

def download_dataset():
    if os.path.exists(DATASET_DIR):
        extract_zip_files(DATASET_DIR)
        return DATASET_DIR

    path = kagglehub.dataset_download("mahdavi1202/skin-cancer")
    os.makedirs(DATASET_DIR, exist_ok=True)

    for item in os.listdir(path):
        source_path = os.path.join(path, item)
        target_path = os.path.join(DATASET_DIR, item)

        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        elif not os.path.exists(target_path):
            shutil.copy2(source_path, target_path)

    extract_zip_files(DATASET_DIR)

    return DATASET_DIR


def extract_zip_files(dataset_dir):
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if not filename.lower().endswith(".zip"):
                continue

            zip_path = os.path.join(root, filename)
            extract_dir = os.path.join(root, os.path.splitext(filename)[0])

            if os.path.exists(extract_dir):
                continue

            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(extract_dir)


def find_metadata_csv(dataset_dir):
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if not filename.lower().endswith(".csv"):
                continue

            csv_path = os.path.join(root, filename)
            try:
                df = pd.read_csv(csv_path, nrows=5)
            except Exception:
                continue

            columns = set(df.columns)
            if {"img_id", "diagnostic"}.issubset(columns):
                return csv_path

    raise FileNotFoundError("Could not find PAD-UFES-20 metadata CSV with img_id and diagnostic columns.")


def make_image_index(dataset_dir):
    image_index = {}
    image_extensions = {".png", ".jpg", ".jpeg"}

    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in image_extensions:
                continue

            path = os.path.join(root, filename)
            image_index[filename] = path
            image_index[stem] = path

    return image_index


def clean_state_dict(state_dict):
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        for prefix in ["module.", "model.", "unet."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned_state_dict[new_key] = value

    return cleaned_state_dict


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_segmentation_model(device):
    checkpoint = load_checkpoint(MODEL_PATH, device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = {
                key: value
                for key, value in checkpoint.items()
                if torch.is_tensor(value)
            }
    else:
        raise ValueError("Expected segmentation checkpoint to contain a state dict.")

    state_dict = clean_state_dict(state_dict)

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    base_channels_to_try = []

    if "base_channels" in config:
        base_channels_to_try.append(config["base_channels"])

    base_channels_to_try.extend([32, 64, 16])

    last_error = None
    for base_channels in base_channels_to_try:
        model = UNet(base_channels=base_channels).to(device)
        try:
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            print(f"Loaded segmentation model with base_channels={base_channels}")
            return model
        except RuntimeError as error:
            last_error = error

    raise RuntimeError(
        "Could not load segmentation_unet_v2.pth into the U-Net architecture in this script. "
        "The checkpoint may have been trained with different layer names or widths."
    ) from last_error


def main():
    dataset_dir = download_dataset()
    metadata_path = find_metadata_csv(dataset_dir)
    image_index = make_image_index(dataset_dir)

    df = pd.read_csv(metadata_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = load_segmentation_model(device)

    segment_transform = transforms.Compose([
        transforms.Resize((segmentation_image_size, segmentation_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    rows = []
    created = 0
    skipped_existing = 0
    missing_images = 0

    for _, row in df.iterrows():
        img_id = str(row["img_id"])
        img_key = os.path.splitext(img_id)[0]
        image_path = image_index.get(img_id) or image_index.get(img_key) or image_index.get(img_key + ".png")

        if image_path is None:
            missing_images += 1
            continue

        output_filename = img_key + ".png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_path):
            skipped_existing += 1
        else:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                original_size = image.size

                tensor = segment_transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    probs = torch.sigmoid(logits)

                mask_array = (probs.squeeze().cpu().numpy() > mask_threshold).astype(np.uint8) * 255
                mask = Image.fromarray(mask_array).convert("L")
                mask = mask.resize(original_size, Image.NEAREST)

                white_background = Image.new("RGB", original_size, (255, 255, 255))
                extracted_image = Image.composite(image, white_background, mask)
                extracted_image.save(output_path)

            created += 1

        output_row = row.to_dict()
        output_row["original_image_path"] = image_path
        output_row["extracted_image_path"] = output_path
        rows.append(output_row)

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    print(f"Metadata path: {metadata_path}")
    print(f"Created extracted lesion images: {created}")
    print(f"Already existed: {skipped_existing}")
    print(f"Missing original images: {missing_images}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
