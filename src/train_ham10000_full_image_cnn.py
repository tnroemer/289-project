import os
import shutil

import kagglehub
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
IMAGE_DIR = os.path.join(DATASET_DIR, "HAM10000_images")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")
MODEL_NAME = "basic_cnn"

image_size = 128
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
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
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


# -----------------------
# Data setup
# -----------------------

def prepare_dataset():
    if not os.path.exists(DATASET_DIR):
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

        os.makedirs(DATASET_DIR, exist_ok=True)

        shutil.copytree(
            os.path.join(path, "HAM10000_images_part_1"),
            IMAGE_DIR,
            dirs_exist_ok=True,
        )

        shutil.copytree(
            os.path.join(path, "HAM10000_images_part_2"),
            IMAGE_DIR,
            dirs_exist_ok=True,
        )

        csv_path = os.path.join(path, "HAM10000_metadata.csv")
    else:
        csv_path = "/ocean/projects/mth250011p/troemer/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/HAM10000_metadata.csv"

    df = pd.read_csv(csv_path)
    df["image"] = df["image_id"].apply(
        lambda x: os.path.join(IMAGE_DIR, x + ".jpg")
    )
    df["image_path"] = df["image"]

    labels = sorted(df["dx"].unique())
    label_feature = ClassLabel(names=labels)

    df["label"] = df["dx"].apply(lambda x: label_feature.str2int(x))

    dataset = Dataset.from_pandas(df[["image_id", "image_path", "dx", "image", "label"]])
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", label_feature)

    splits = dataset.train_test_split(
        test_size=0.2,
        stratify_by_column="label",
        seed=42,
    )

    train_ds = splits["train"].with_transform(train_transforms)
    val_ds = splits["test"].with_transform(val_transforms)

    return train_ds, val_ds, labels


# -----------------------
# Training / evaluation
# -----------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def save_predictions(model, loader, criterion, device, labels, predictions_path, metrics_path):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels_batch = batch["labels"].to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels_batch).sum().item()
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
                    "correct": int(preds_cpu[i] == labels_cpu[i]),
                }

                for label_index, label_name in enumerate(labels):
                    row[f"prob_{label_name}"] = float(probs_cpu[i, label_index])

                rows.append(row)

    avg_loss = total_loss / total
    accuracy = correct / total

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(predictions_path, index=False)

    metrics_rows = [
        {"metric": "loss", "class": "overall", "value": avg_loss},
        {"metric": "accuracy", "class": "overall", "value": accuracy},
        {"metric": "num_examples", "class": "overall", "value": total},
    ]

    for label_name in labels:
        true_positive = ((pred_df["true_dx"] == label_name) & (pred_df["pred_dx"] == label_name)).sum()
        false_positive = ((pred_df["true_dx"] != label_name) & (pred_df["pred_dx"] == label_name)).sum()
        false_negative = ((pred_df["true_dx"] == label_name) & (pred_df["pred_dx"] != label_name)).sum()
        true_negative = ((pred_df["true_dx"] != label_name) & (pred_df["pred_dx"] != label_name)).sum()

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        metrics_rows.extend([
            {"metric": "support", "class": label_name, "value": true_positive + false_negative},
            {"metric": "precision", "class": label_name, "value": precision},
            {"metric": "recall", "class": label_name, "value": recall},
            {"metric": "specificity", "class": label_name, "value": specificity},
            {"metric": "f1", "class": label_name, "value": f1},
        ])

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Acc: {accuracy:.4f}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")

    return avg_loss, accuracy


# -----------------------
# Main
# -----------------------

def main():
    train_ds, val_ds, labels = prepare_dataset()

    num_classes = len(labels)

    config = {
        "model": "BasicCNN",
        "dataset": "HAM10000",
        "run_dir": RUN_DIR,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "image_size": image_size,
        "num_classes": num_classes,
        "num_workers": num_workers,
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
        name="basic-cnn",
        config=config,
    )

    model = BasicCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Optional. This can add overhead, but it is useful for debugging.
    # wandb.watch(model, log="gradients", log_freq=100)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    best_model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pt")
    predictions_path = os.path.join(PRED_DIR, f"{MODEL_NAME}_test_predictions.csv")
    metrics_path = os.path.join(PRED_DIR, f"{MODEL_NAME}_test_metrics.csv")

    best_val_acc = -1.0
    patience = 8
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        })

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "labels": labels,
                "config": config,
            }

            torch.save(checkpoint, best_model_path)

            wandb.run.summary["best_val_accuracy"] = best_val_acc
            wandb.run.summary["best_epoch"] = epoch + 1

            wandb.save(best_model_path)

            print(f"Saved new best model: val_acc={val_acc:.4f}")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = save_predictions(
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
    })
    wandb.run.summary["test_loss"] = test_loss
    wandb.run.summary["test_accuracy"] = test_acc

    wandb.finish()


if __name__ == "__main__":
    main()
