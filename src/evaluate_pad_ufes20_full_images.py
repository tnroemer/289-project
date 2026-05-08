import os
import shutil
import zipfile

import kagglehub
import pandas as pd
import torch

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -----------------------
# Global config
# -----------------------

DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")

os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"

DATASET_DIR = os.path.join(DATA_ROOT, "datasets", "pad-ufes-20")
MODEL_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")

batch_size = 32
num_workers = 4

PAD_MALIGNANT_LABELS = {"bcc", "mel", "ack", "scc"}

MODEL_SPECS = [
    {
        "name": "basic_cnn_binary",
        "path": os.path.join(MODEL_DIR, "basic_cnn_binary_best.pt"),
        "type": "cnn",
    },
    {
        "name": "vit_binary",
        "path": os.path.join(MODEL_DIR, "vit_binary_best.pt"),
        "type": "vit",
    },
]


# -----------------------
# Models
# -----------------------

class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class BasicVIT(nn.Module):
    def __init__(
        self,
        num_classes,
        image_size=64,
        patch_size=8,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.classifier(x)

        return x


class PadFullImagesDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "pixel_values": image,
            "img_id": row["img_id"],
            "diagnostic": row["diagnostic"],
            "image_path": row["image_path"],
        }


# -----------------------
# Helpers
# -----------------------

def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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


def build_model(model_type, checkpoint, num_outputs):
    config = checkpoint.get("config", {})

    if model_type == "cnn":
        return BasicCNN(num_classes=num_outputs)

    if model_type == "vit":
        return BasicVIT(
            num_classes=num_outputs,
            image_size=config.get("image_size", 64),
            patch_size=config.get("patch_size", 8),
            embed_dim=config.get("embed_dim", 128),
            depth=config.get("depth", 4),
            num_heads=config.get("num_heads", 4),
            mlp_dim=config.get("mlp_dim", 256),
            dropout=config.get("dropout", 0.1),
        )

    raise ValueError(f"Unknown model type: {model_type}")


def make_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def is_pad_malignant(label):
    return str(label).lower() in PAD_MALIGNANT_LABELS


def save_metrics(pred_df, metrics_path):
    true_positive = ((pred_df["true_malignant"] == 1) & (pred_df["pred_malignant"] == 1)).sum()
    false_positive = ((pred_df["true_malignant"] == 0) & (pred_df["pred_malignant"] == 1)).sum()
    false_negative = ((pred_df["true_malignant"] == 1) & (pred_df["pred_malignant"] == 0)).sum()
    true_negative = ((pred_df["true_malignant"] == 0) & (pred_df["pred_malignant"] == 0)).sum()

    total = len(pred_df)
    accuracy = (true_positive + true_negative) / total if total > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity > 0 else 0.0

    metrics_rows = [
        {"metric": "num_examples", "value": total},
        {"metric": "binary_accuracy", "value": accuracy},
        {"metric": "binary_precision", "value": precision},
        {"metric": "binary_sensitivity", "value": sensitivity},
        {"metric": "binary_specificity", "value": specificity},
        {"metric": "binary_f1", "value": f1},
        {"metric": "true_positive", "value": true_positive},
        {"metric": "false_positive", "value": false_positive},
        {"metric": "false_negative", "value": false_negative},
        {"metric": "true_negative", "value": true_negative},
    ]

    for label, count in pred_df["diagnostic"].value_counts().items():
        metrics_rows.append({"metric": f"true_count_{label}", "value": count})

    for label, count in pred_df["pred_dx"].value_counts().items():
        metrics_rows.append({"metric": f"pred_count_{label}", "value": count})

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Binary accuracy: {accuracy:.4f}")
    print(f"Binary sensitivity: {sensitivity:.4f}")
    print(f"Binary specificity: {specificity:.4f}")
    print(f"Saved metrics to {metrics_path}")


def evaluate_model(model_spec, manifest_df, device):
    if not os.path.exists(model_spec["path"]):
        print(f"Skipping missing model: {model_spec['path']}")
        return

    checkpoint = load_checkpoint(model_spec["path"], device)
    labels = checkpoint["labels"]
    config = checkpoint.get("config", {})
    image_size = config.get("image_size", 128 if model_spec["type"] == "cnn" else 64)

    model = build_model(model_spec["type"], checkpoint, num_outputs=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = make_transform(image_size)
    dataset = PadFullImagesDataset(manifest_df, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            probs_cpu = probs.cpu()
            preds_cpu = preds.cpu()

            for i in range(images.size(0)):
                pred_malignant = int(preds_cpu[i])
                pred_label = labels[pred_malignant]
                true_label = batch["diagnostic"][i]

                row = {
                    "img_id": batch["img_id"][i],
                    "image_path": batch["image_path"][i],
                    "diagnostic": true_label,
                    "true_malignant": int(is_pad_malignant(true_label)),
                    "pred_label": pred_malignant,
                    "pred_dx": pred_label,
                    "pred_malignant": pred_malignant,
                    "prob_malignant": float(probs_cpu[i]),
                }

                rows.append(row)

    pred_df = pd.DataFrame(rows)

    os.makedirs(PRED_DIR, exist_ok=True)
    predictions_path = os.path.join(
        PRED_DIR,
        f"pad_ufes_20_full_images_{model_spec['name']}_predictions.csv",
    )
    metrics_path = os.path.join(
        PRED_DIR,
        f"pad_ufes_20_full_images_{model_spec['name']}_metrics.csv",
    )

    pred_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")

    save_metrics(pred_df, metrics_path)


def main():
    dataset_dir = download_dataset()
    metadata_path = find_metadata_csv(dataset_dir)
    image_index = make_image_index(dataset_dir)

    manifest_df = pd.read_csv(metadata_path)
    manifest_df["image_path"] = manifest_df["img_id"].apply(
        lambda x: image_index.get(str(x)) or image_index.get(os.path.splitext(str(x))[0])
    )
    missing_images = manifest_df["image_path"].isna().sum()
    manifest_df = manifest_df.dropna(subset=["image_path"]).reset_index(drop=True)

    print(f"Metadata path: {metadata_path}")
    print(f"Found PAD-UFES-20 full images: {len(manifest_df)}")
    print(f"Missing PAD-UFES-20 full images: {missing_images}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for model_spec in MODEL_SPECS:
        print(f"Evaluating {model_spec['name']}")
        evaluate_model(model_spec, manifest_df, device)


if __name__ == "__main__":
    main()
