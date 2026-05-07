import os

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

EXTRACTED_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-extracted-lesions")
MANIFEST_PATH = os.path.join(EXTRACTED_DIR, "metadata.csv")
MODEL_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")

batch_size = 32
num_workers = 4

PAD_MALIGNANT_LABELS = {"bcc", "mel", "scc", "bod"}
HAM_MALIGNANT_LABELS = {"akiec", "bcc", "mel"}

MODEL_SPECS = [
    {
        "name": "basic_cnn_lesion_white",
        "path": os.path.join(MODEL_DIR, "basic_cnn_lesion_white_best.pt"),
        "type": "cnn",
    },
    {
        "name": "vit_lesion_white",
        "path": os.path.join(MODEL_DIR, "vit_lesion_white_best.pt"),
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


class PadExtractedLesionsDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(row["extracted_image_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "pixel_values": image,
            "img_id": row["img_id"],
            "diagnostic": row["diagnostic"],
            "extracted_image_path": row["extracted_image_path"],
        }


# -----------------------
# Helpers
# -----------------------

def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(model_type, checkpoint, num_classes):
    config = checkpoint.get("config", {})

    if model_type == "cnn":
        return BasicCNN(num_classes=num_classes)

    if model_type == "vit":
        return BasicVIT(
            num_classes=num_classes,
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


def is_ham_malignant(label):
    return str(label).lower() in HAM_MALIGNANT_LABELS


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

    model = build_model(model_spec["type"], checkpoint, num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = make_transform(image_size)
    dataset = PadExtractedLesionsDataset(manifest_df, transform)
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
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            probs_cpu = probs.cpu()
            preds_cpu = preds.cpu()

            for i in range(images.size(0)):
                pred_label = labels[int(preds_cpu[i])]
                true_label = batch["diagnostic"][i]

                row = {
                    "img_id": batch["img_id"][i],
                    "extracted_image_path": batch["extracted_image_path"][i],
                    "diagnostic": true_label,
                    "true_malignant": int(is_pad_malignant(true_label)),
                    "pred_label": int(preds_cpu[i]),
                    "pred_dx": pred_label,
                    "pred_malignant": int(is_ham_malignant(pred_label)),
                }

                for label_index, label_name in enumerate(labels):
                    row[f"prob_{label_name}"] = float(probs_cpu[i, label_index])

                rows.append(row)

    pred_df = pd.DataFrame(rows)

    os.makedirs(PRED_DIR, exist_ok=True)
    predictions_path = os.path.join(
        PRED_DIR,
        f"pad_ufes_20_extracted_lesions_{model_spec['name']}_predictions.csv",
    )
    metrics_path = os.path.join(
        PRED_DIR,
        f"pad_ufes_20_extracted_lesions_{model_spec['name']}_metrics.csv",
    )

    pred_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")

    save_metrics(pred_df, metrics_path)


def main():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            "Could not find extracted PAD-UFES-20 lesion manifest. "
            "Run create_pad_ufes20_extracted_lesions.py first."
        )

    manifest_df = pd.read_csv(MANIFEST_PATH)
    manifest_df = manifest_df[manifest_df["extracted_image_path"].apply(os.path.exists)].reset_index(drop=True)

    print(f"Found extracted PAD-UFES-20 lesion images: {len(manifest_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for model_spec in MODEL_SPECS:
        print(f"Evaluating {model_spec['name']}")
        evaluate_model(model_spec, manifest_df, device)


if __name__ == "__main__":
    main()
