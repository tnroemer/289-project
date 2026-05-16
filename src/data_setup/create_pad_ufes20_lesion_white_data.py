import os

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms

from models.model_architectures import build_segmentation_model


# Configurable for reproducibility; defaults keep the repo self-contained.
# Override on a cluster via SKIN_LESIONS_DATA_ROOT / SKIN_LESIONS_RUN_DIR.
DATA_ROOT = os.environ.get(
    "SKIN_LESIONS_DATA_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
RUN_DIR = os.environ.get("SKIN_LESIONS_RUN_DIR", DATA_ROOT)

INPUT_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-images")
INPUT_METADATA_PATH = os.path.join(INPUT_DIR, "metadata.csv")
MODEL_PATH = os.path.join(RUN_DIR, "models", "segmentation_deeplabv3_resnet50.pth")
OUTPUT_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-lesion-white-images")
OUTPUT_METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

default_image_size = 320
mask_threshold = 0.5


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def segmentation_logits(output):
    if isinstance(output, dict):
        return output["out"]
    return output


def load_segmentation_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing segmentation model: {MODEL_PATH}. "
            "Run `sbatch submit/submit_train_ham10000_segmentation_model.sh` first."
        )

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

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    config = {**config, "use_pretrained_backbone": False}
    model = build_segmentation_model(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded segmentation model: {MODEL_PATH}")
    return model, config


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

    model, model_config = load_segmentation_model(device)
    model_image_size = int(model_config.get("image_size", default_image_size))

    segment_transform = transforms.Compose([
        transforms.Resize((model_image_size, model_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    rows = []
    created = 0
    missing_images = 0

    for _, row in df.iterrows():
        img_id = str(row["img_id"])
        img_key = os.path.splitext(img_id)[0]
        image_path = row["image_path"]
        output_path = os.path.join(OUTPUT_DIR, img_key + ".jpg")

        if not os.path.exists(image_path):
            missing_images += 1
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_size = image.size

            tensor = segment_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = segmentation_logits(model(tensor))
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
    print(f"Missing prepared PAD images: {missing_images}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Saved metadata: {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()
