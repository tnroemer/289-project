import os

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch import nn
from torchvision import transforms


DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")

INPUT_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-images")
INPUT_METADATA_PATH = os.path.join(INPUT_DIR, "metadata.csv")
MODEL_PATH = os.path.join(RUN_DIR, "models", "segmentation_unet_v2.pth")
OUTPUT_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-lesion-white-images")
OUTPUT_METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

segmentation_image_size = 256
mask_threshold = 0.5


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        self.down1 = ConvBlock(in_channels, base_channels)
        self.down2 = ConvBlock(base_channels, base_channels * 2)
        self.down3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(self.pool(down1))
        down3 = self.down3(self.pool(down2))
        bottleneck = self.bottleneck(self.pool(down3))

        x = self.up3(bottleneck)
        x = torch.cat((x, down3), dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat((x, down2), dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat((x, down1), dim=1)
        x = self.conv1(x)

        return self.out(x)


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

    if "down1.net.0.weight" in state_dict:
        base_channels_to_try.append(state_dict["down1.net.0.weight"].shape[0])

    base_channels_to_try.extend([32, 64, 16])
    base_channels_to_try = list(dict.fromkeys(base_channels_to_try))

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
    if not os.path.exists(INPUT_METADATA_PATH):
        raise FileNotFoundError(
            f"Missing prepared PAD-UFES-20 metadata: {INPUT_METADATA_PATH}. "
            "Run `sbatch submit/submit_create_data.sh` first."
        )

    df = pd.read_csv(INPUT_METADATA_PATH)
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
        image_path = row["image_path"]
        output_path = os.path.join(OUTPUT_DIR, img_key + ".jpg")

        if not os.path.exists(image_path):
            missing_images += 1
            continue

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
                extracted_image.save(output_path, quality=95)

            created += 1

        output_row = row.to_dict()
        output_row["original_image_path"] = image_path
        output_row["lesion_white_image_path"] = output_path
        output_row["image_path"] = output_path
        rows.append(output_row)

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(OUTPUT_METADATA_PATH, index=False)

    print(f"Input metadata: {INPUT_METADATA_PATH}")
    print(f"Created PAD lesion-white images: {created}")
    print(f"Already existed: {skipped_existing}")
    print(f"Missing prepared PAD images: {missing_images}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Saved metadata: {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()
