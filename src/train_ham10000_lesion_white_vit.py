import os

import pandas as pd
import torch
import wandb

from datasets import Dataset, Image, ClassLabel
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms


# -----------------------
# Global config
# -----------------------

os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"

DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")
DATASET_DIR = os.path.join(DATA_ROOT, "datasets", "skin-cancer-mnist-ham10000")
LESION_IMAGE_DIR = os.path.join(RUN_DIR, "data", "lesion-white-images")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")
MODEL_NAME = "vit_lesion_white_binary"
MALIGNANT_LABELS = {"mel", "bcc", "akiec"}
BINARY_LABELS = ["benign", "malignant"]

image_size = 64
batch_size = 32
learning_rate = 3e-4
num_epochs = 50
num_workers = 4


# -----------------------
# Transforms
# -----------------------

# have to use mean and sd for our particular dataset in normalize

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def train_transforms(examples):
    examples["pixel_values"] = [
        train_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples


def val_transforms(examples):
    examples["pixel_values"] = [
        val_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples


def collate_fn(batch):
    output = {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.float32),
    }

    if "image_id" in batch[0]:
        output["image_id"] = [x["image_id"] for x in batch]
    if "image_path" in batch[0]:
        output["image_path"] = [x["image_path"] for x in batch]
    if "dx" in batch[0]:
        output["dx"] = [x["dx"] for x in batch]

    return output


# -----------------------
# Model
# -----------------------

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

        # Split image into patches and turn each patch into an embedding vector

        self.patch_embed = nn.Conv2d(

            in_channels=3,

            out_channels=embed_dim,

            kernel_size=patch_size,

            stride=patch_size,

        )

        # Extra token that learns to summarize the whole image

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings tell the transformer where each patch came from

        self.pos_embed = nn.Parameter(

            torch.zeros(1, self.num_patches + 1, embed_dim)

        )

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

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=depth,

        )

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

        # [B, embed_dim, H/patch_size, W/patch_size]

        x = x.flatten(2)

        # [B, embed_dim, num_patches]

        x = x.transpose(1, 2)

        # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # [B, num_patches + 1, embed_dim]

        x = x + self.pos_embed

        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]

        # Use class token for classification

        x = self.classifier(x)

        return x


# -----------------------
# Data setup
# -----------------------

def prepare_dataset():
    csv_path = "/ocean/projects/mth250011p/troemer/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_metadata.csv"

    df = pd.read_csv(csv_path)
    df["image"] = df["image_id"].apply(
        lambda x: os.path.join(LESION_IMAGE_DIR, x + ".jpg")
    )
    df["image_path"] = df["image"]

    missing_lesion_images = (~df["image"].apply(os.path.exists)).sum()
    df = df[df["image"].apply(os.path.exists)].reset_index(drop=True)

    print(f"Using lesion-white image folder: {LESION_IMAGE_DIR}")
    print(f"Found lesion-white images: {len(df)}")
    print(f"Missing lesion-white images: {missing_lesion_images}")

    if len(df) == 0:
        raise FileNotFoundError(
            "No lesion-white images found. Run create_ham10000_lesion_white_images.py first."
        )

    labels = BINARY_LABELS
    label_feature = ClassLabel(names=labels)

    df["label"] = df["dx"].str.lower().isin(MALIGNANT_LABELS).astype(int)

    dataset = Dataset.from_pandas(df[["image_id", "image_path", "dx", "image", "label"]])
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", label_feature)

    splits = dataset.train_test_split(
        test_size=0.2,
        stratify_by_column="label",
        seed=42,
    )

    train_labels = splits["train"]["label"]
    positive_count = sum(train_labels)
    negative_count = len(train_labels) - positive_count
    pos_weight = negative_count / positive_count

    train_ds = splits["train"].with_transform(train_transforms)
    val_ds = splits["test"].with_transform(val_transforms)

    return train_ds, val_ds, labels, pos_weight


# -----------------------
# Training / evaluation
# -----------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        true_positive += ((preds == 1) & (labels == 1)).sum().item()
        false_positive += ((preds == 1) & (labels == 0)).sum().item()
        false_negative += ((preds == 0) & (labels == 1)).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return avg_loss, accuracy, precision, recall, f1


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            true_positive += ((preds == 1) & (labels == 1)).sum().item()
            false_positive += ((preds == 1) & (labels == 0)).sum().item()
            false_negative += ((preds == 0) & (labels == 1)).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return avg_loss, accuracy, precision, recall, f1


def save_predictions(model, loader, criterion, device, labels, predictions_path, metrics_path):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels_batch = batch["labels"].to(device, non_blocking=True)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels_batch).sum().item()
            true_positive += ((preds == 1) & (labels_batch == 1)).sum().item()
            false_positive += ((preds == 1) & (labels_batch == 0)).sum().item()
            false_negative += ((preds == 0) & (labels_batch == 1)).sum().item()
            true_negative += ((preds == 0) & (labels_batch == 0)).sum().item()
            total += labels_batch.size(0)

            probs_cpu = probs.cpu()
            preds_cpu = preds.cpu()
            labels_cpu = labels_batch.cpu()

            for i in range(images.size(0)):
                row = {
                    "image_id": batch["image_id"][i],
                    "image_path": batch["image_path"][i],
                    "true_label": int(labels_cpu[i]),
                    "true_dx": batch["dx"][i],
                    "pred_label": int(preds_cpu[i]),
                    "pred_dx": labels[int(preds_cpu[i])],
                    "prob_malignant": float(probs_cpu[i]),
                    "correct": int(preds_cpu[i] == labels_cpu[i]),
                }

                rows.append(row)

    avg_loss = total_loss / total
    accuracy = correct / total
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(predictions_path, index=False)

    metrics_rows = [
        {"metric": "loss", "class": "overall", "value": avg_loss},
        {"metric": "accuracy", "class": "overall", "value": accuracy},
        {"metric": "precision", "class": "malignant", "value": precision},
        {"metric": "recall", "class": "malignant", "value": recall},
        {"metric": "specificity", "class": "benign", "value": specificity},
        {"metric": "f1", "class": "malignant", "value": f1},
        {"metric": "num_examples", "class": "overall", "value": total},
        {"metric": "true_positive", "class": "malignant", "value": true_positive},
        {"metric": "false_positive", "class": "malignant", "value": false_positive},
        {"metric": "false_negative", "class": "malignant", "value": false_negative},
        {"metric": "true_negative", "class": "benign", "value": true_negative},
    ]

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Acc: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")

    return avg_loss, accuracy, precision, recall, f1


# -----------------------
# Main
# -----------------------

def main():
    train_ds, val_ds, labels, pos_weight = prepare_dataset()

    config = {
        "model": "ViT",
        "dataset": "HAM10000-lesion-white",
        "target": "binary_malignant",
        "malignant_labels": sorted(MALIGNANT_LABELS),
        "pos_weight": pos_weight,
        "run_dir": RUN_DIR,
        "lesion_image_dir": LESION_IMAGE_DIR,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "image_size": image_size,
        "num_outputs": 1,
        "num_workers": num_workers,
        "patch_size": 8,
        "embed_dim": 128,
        "depth": 4,
        "num_heads": 4,
        "mlp_dim": 256,
        "dropout": 0.1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    wandb.login(key=os.environ["WANDB_API_KEY"])

    run = wandb.init(
        entity="tnroemer-berk",
        project="skin-cancer-cnn",
        name="vit-lesion-white-binary",
        config=config,
    )

    model = BasicVIT(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = AdamW(model.parameters(), 
                      lr=learning_rate,
                      weight_decay=0.05)

    # Optional. This can add overhead, but it is useful for debugging.
    # wandb.watch(model, log="gradients", log_freq=100)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    best_model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pt")
    predictions_path = os.path.join(PRED_DIR, f"{MODEL_NAME}_test_predictions.csv")
    metrics_path = os.path.join(PRED_DIR, f"{MODEL_NAME}_test_metrics.csv")

    best_val_f1 = -1.0
    patience = 8
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "train/precision": train_precision,
            "train/recall": train_recall,
            "train/f1": train_f1,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/precision": val_precision,
            "val/recall": val_recall,
            "val/f1": val_f1,
        })

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Precision: {train_precision:.4f} | "
            f"Train Recall: {train_recall:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Precision: {val_precision:.4f} | "
            f"Val Recall: {val_recall:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_loss": val_loss,
                "labels": labels,
                "malignant_labels": sorted(MALIGNANT_LABELS),
                "pos_weight": pos_weight,
                "config": config,
            }

            torch.save(checkpoint, best_model_path)

            wandb.run.summary["best_val_f1"] = best_val_f1
            wandb.run.summary["best_val_accuracy"] = val_acc
            wandb.run.summary["best_val_precision"] = val_precision
            wandb.run.summary["best_val_recall"] = val_recall
            wandb.run.summary["best_epoch"] = epoch + 1

            wandb.save(best_model_path)

            print(f"Saved new best model: val_f1={val_f1:.4f}")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_precision, test_recall, test_f1 = save_predictions(
        model,
        val_loader,
        criterion,
        device,
        labels,
        predictions_path,
        metrics_path,
    )

    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_acc,
        "test/precision": test_precision,
        "test/recall": test_recall,
        "test/f1": test_f1,
    })
    wandb.run.summary["test_loss"] = test_loss
    wandb.run.summary["test_accuracy"] = test_acc
    wandb.run.summary["test_precision"] = test_precision
    wandb.run.summary["test_recall"] = test_recall
    wandb.run.summary["test_f1"] = test_f1

    wandb.finish()


if __name__ == "__main__":
    main()
