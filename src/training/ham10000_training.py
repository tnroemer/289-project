import os

import pandas as pd
import torch
import wandb

from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.model_architectures import build_model


DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")
SPLIT_DIR = os.path.join(RUN_DIR, "data", "splits")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")

COMMON_LABELS = ["akiec", "bcc", "bkl", "mel", "nv"]
MALIGNANT_LABELS = {"akiec", "bcc", "mel"}
TRAIN_AUGMENTATION_NOTE = (
    "resize, random resized crop, flips, rotation, affine translation/scale/shear, "
    "mild perspective, color jitter, occasional grayscale, and mild blur"
)

batch_size = 32
num_epochs = 60
num_workers = 4
seed = 42

MODEL_SETTINGS = {
    "cnn": {
        "display_name": "basic-cnn",
        "checkpoint_name": "basic_cnn",
        "model": "BasicCNN",
        "image_size": 128,
        "learning_rate": 3e-4,
    },
    "vit": {
        "display_name": "vit",
        "checkpoint_name": "vit",
        "model": "ViT",
        "image_size": 96,
        "learning_rate": 2e-4,
        "patch_size": 8,
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 6,
        "mlp_dim": 384,
        "dropout": 0.15,
    },
    "resnet": {
        "display_name": "resnet",
        "checkpoint_name": "resnet",
        "model": "SimpleResNet",
        "image_size": 128,
        "learning_rate": 3e-4,
    },
}


class SkinLesionDataset(Dataset):
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
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "dx": row["dx"],
        }


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.80, 1.0),
            ratio=(0.90, 1.10),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.90, 1.10),
            shear=5,
        ),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ], p=0.20),
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

    return train_transform, val_transform


def load_split(split_name, image_source):
    split_path = os.path.join(SPLIT_DIR, f"ham10000_{split_name}.csv")

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Run `sbatch submit/submit_create_data.sh` before training."
        )

    df = pd.read_csv(split_path)

    if image_source == "full_image":
        df["image_path"] = df["full_image_path"]
    elif image_source == "lesion_white":
        df["image_path"] = df["lesion_image_path"]
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    missing_images = (~df["image_path"].apply(os.path.exists)).sum()
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    print(f"{split_name} examples found: {len(df)}")
    print(f"{split_name} missing images: {missing_images}")

    if len(df) == 0:
        raise FileNotFoundError(f"No images found for split {split_name}.")

    return df


def compute_class_weights(train_df, labels):
    train_labels = train_df["label"].tolist()
    counts = pd.Series(train_labels).value_counts().to_dict()
    raw_weights = []

    for i, label in enumerate(labels):
        count = counts.get(i, 0)
        if count == 0:
            raise ValueError(f"No training examples for class {label}")

        frequency_weight = len(train_labels) / (len(labels) * count)
        raw_weights.append(frequency_weight)

    mean_weight = sum(raw_weights) / len(raw_weights)
    class_weights = [weight / mean_weight for weight in raw_weights]

    return class_weights


def prepare_datasets(image_source, image_size):
    train_df = load_split("train", image_source)
    val_df = load_split("val", image_source)
    test_df = load_split("test", image_source)

    train_transform, val_transform = make_transforms(image_size)

    train_ds = SkinLesionDataset(train_df, train_transform)
    val_ds = SkinLesionDataset(val_df, val_transform)
    test_ds = SkinLesionDataset(test_df, val_transform)

    class_weights = compute_class_weights(train_df, COMMON_LABELS)

    print("Train counts:")
    print(train_df["dx"].value_counts().sort_index())
    print("Class weights:")
    for label, weight in zip(COMMON_LABELS, class_weights):
        print(f"{label}: {weight:.4f}")

    return train_ds, val_ds, test_ds, train_df, val_df, test_df, class_weights


def compute_metrics(targets, preds, labels):
    num_classes = len(labels)
    total = len(targets)
    correct = sum(int(t == p) for t, p in zip(targets, preds))

    per_class = {}
    recalls = []
    precisions = []
    f1s = []

    for class_index, label in enumerate(labels):
        true_positive = sum(int(t == class_index and p == class_index) for t, p in zip(targets, preds))
        false_positive = sum(int(t != class_index and p == class_index) for t, p in zip(targets, preds))
        false_negative = sum(int(t == class_index and p != class_index) for t, p in zip(targets, preds))
        true_negative = sum(int(t != class_index and p != class_index) for t, p in zip(targets, preds))
        support = sum(int(t == class_index) for t in targets)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    malignant_indices = [i for i, label in enumerate(labels) if label in MALIGNANT_LABELS]
    true_malignant = [t in malignant_indices for t in targets]
    pred_malignant = [p in malignant_indices for p in preds]

    true_positive = sum(int(t and p) for t, p in zip(true_malignant, pred_malignant))
    false_positive = sum(int((not t) and p) for t, p in zip(true_malignant, pred_malignant))
    false_negative = sum(int(t and (not p)) for t, p in zip(true_malignant, pred_malignant))
    true_negative = sum(int((not t) and (not p)) for t, p in zip(true_malignant, pred_malignant))

    binary_accuracy = (true_positive + true_negative) / total if total > 0 else 0.0
    malignant_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    malignant_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    malignant_specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    malignant_f1 = (
        2 * malignant_precision * malignant_recall / (malignant_precision + malignant_recall)
        if malignant_precision + malignant_recall > 0
        else 0.0
    )
    benign_precision = true_negative / (true_negative + false_negative) if true_negative + false_negative > 0 else 0.0
    benign_recall = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    benign_specificity = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if benign_precision + benign_recall > 0
        else 0.0
    )
    binary_macro_precision = (malignant_precision + benign_precision) / 2
    binary_macro_recall = (malignant_recall + benign_recall) / 2
    binary_macro_f1 = (malignant_f1 + benign_f1) / 2

    confusion = {}
    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            confusion[f"{true_label}_pred_{pred_label}"] = sum(
                int(t == true_index and p == pred_index)
                for t, p in zip(targets, preds)
            )

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "macro_precision": sum(precisions) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
        "macro_f1": sum(f1s) / num_classes,
        "balanced_accuracy": sum(recalls) / num_classes,
        "binary_accuracy": binary_accuracy,
        "binary_macro_precision": binary_macro_precision,
        "binary_macro_recall": binary_macro_recall,
        "binary_macro_f1": binary_macro_f1,
        "binary_balanced_accuracy": binary_macro_recall,
        "malignant_precision": malignant_precision,
        "malignant_recall": malignant_recall,
        "malignant_specificity": malignant_specificity,
        "malignant_f1": malignant_f1,
        "benign_precision": benign_precision,
        "benign_recall": benign_recall,
        "benign_specificity": benign_specificity,
        "benign_f1": benign_f1,
        "binary_true_positive": true_positive,
        "binary_false_positive": false_positive,
        "binary_false_negative": false_negative,
        "binary_true_negative": true_negative,
        "malignant_support": true_positive + false_negative,
        "benign_support": true_negative + false_positive,
        "num_examples": total,
        "per_class": per_class,
        "confusion": confusion,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, labels):
    model.train()

    total_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels_batch = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total += labels_batch.size(0)
        all_targets.extend(labels_batch.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds, labels)
    metrics["loss"] = total_loss / total

    return metrics


def evaluate(model, loader, criterion, device, labels):
    model.eval()

    total_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels_batch = batch["labels"].to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total += labels_batch.size(0)
            all_targets.extend(labels_batch.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds, labels)
    metrics["loss"] = total_loss / total

    return metrics


def write_metrics_csv(metrics, labels, metrics_path):
    rows = [
        {"metric": "loss", "class": "overall", "value": metrics["loss"]},
        {"metric": "accuracy", "class": "overall", "value": metrics["accuracy"]},
        {"metric": "macro_precision", "class": "overall", "value": metrics["macro_precision"]},
        {"metric": "macro_recall", "class": "overall", "value": metrics["macro_recall"]},
        {"metric": "macro_f1", "class": "overall", "value": metrics["macro_f1"]},
        {"metric": "balanced_accuracy", "class": "overall", "value": metrics["balanced_accuracy"]},
        {"metric": "binary_accuracy", "class": "overall", "value": metrics["binary_accuracy"]},
        {"metric": "binary_macro_precision", "class": "overall", "value": metrics["binary_macro_precision"]},
        {"metric": "binary_macro_recall", "class": "overall", "value": metrics["binary_macro_recall"]},
        {"metric": "binary_macro_f1", "class": "overall", "value": metrics["binary_macro_f1"]},
        {"metric": "binary_balanced_accuracy", "class": "overall", "value": metrics["binary_balanced_accuracy"]},
        {"metric": "malignant_precision", "class": "malignant", "value": metrics["malignant_precision"]},
        {"metric": "malignant_recall", "class": "malignant", "value": metrics["malignant_recall"]},
        {"metric": "malignant_specificity", "class": "benign", "value": metrics["malignant_specificity"]},
        {"metric": "malignant_f1", "class": "malignant", "value": metrics["malignant_f1"]},
        {"metric": "benign_precision", "class": "benign", "value": metrics["benign_precision"]},
        {"metric": "benign_recall", "class": "benign", "value": metrics["benign_recall"]},
        {"metric": "benign_specificity", "class": "malignant", "value": metrics["benign_specificity"]},
        {"metric": "benign_f1", "class": "benign", "value": metrics["benign_f1"]},
        {"metric": "binary_true_positive", "class": "malignant", "value": metrics["binary_true_positive"]},
        {"metric": "binary_false_positive", "class": "malignant", "value": metrics["binary_false_positive"]},
        {"metric": "binary_false_negative", "class": "malignant", "value": metrics["binary_false_negative"]},
        {"metric": "binary_true_negative", "class": "benign", "value": metrics["binary_true_negative"]},
        {"metric": "support", "class": "malignant", "value": metrics["malignant_support"]},
        {"metric": "support", "class": "benign", "value": metrics["benign_support"]},
        {"metric": "num_examples", "class": "overall", "value": metrics["num_examples"]},
    ]

    for label in labels:
        class_metrics = metrics["per_class"][label]
        for metric_name, value in class_metrics.items():
            rows.append({"metric": metric_name, "class": label, "value": value})

    for name, value in metrics["confusion"].items():
        rows.append({"metric": f"confusion_{name}", "class": "overall", "value": value})

    pd.DataFrame(rows).to_csv(metrics_path, index=False)


def save_predictions(model, loader, criterion, device, labels, predictions_path, metrics_path):
    model.eval()

    total_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []
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
            total += labels_batch.size(0)
            all_targets.extend(labels_batch.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            probs_cpu = probs.cpu()
            preds_cpu = preds.cpu()
            labels_cpu = labels_batch.cpu()

            for i in range(images.size(0)):
                pred_index = int(preds_cpu[i])
                true_index = int(labels_cpu[i])

                row = {
                    "image_id": batch["image_id"][i],
                    "image_path": batch["image_path"][i],
                    "true_label": true_index,
                    "true_dx": batch["dx"][i],
                    "pred_label": pred_index,
                    "pred_dx": labels[pred_index],
                    "correct": int(pred_index == true_index),
                    "true_binary": "malignant" if labels[true_index] in MALIGNANT_LABELS else "benign",
                    "pred_binary": "malignant" if labels[pred_index] in MALIGNANT_LABELS else "benign",
                }
                row["binary_correct"] = int(row["true_binary"] == row["pred_binary"])

                for j, label in enumerate(labels):
                    row[f"prob_{label}"] = float(probs_cpu[i, j])

                rows.append(row)

    metrics = compute_metrics(all_targets, all_preds, labels)
    metrics["loss"] = total_loss / total

    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    write_metrics_csv(metrics, labels, metrics_path)

    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    print(f"Test Binary Accuracy: {metrics['binary_accuracy']:.4f}")
    print(f"Test Binary Macro F1: {metrics['binary_macro_f1']:.4f}")
    print(f"Test Malignant Recall: {metrics['malignant_recall']:.4f}")
    print(f"Test Malignant Specificity: {metrics['malignant_specificity']:.4f}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")

    return metrics


def wandb_metrics(prefix, metrics):
    return {
        f"{prefix}/loss": metrics["loss"],
        f"{prefix}/accuracy": metrics["accuracy"],
        f"{prefix}/macro_precision": metrics["macro_precision"],
        f"{prefix}/macro_recall": metrics["macro_recall"],
        f"{prefix}/macro_f1": metrics["macro_f1"],
        f"{prefix}/balanced_accuracy": metrics["balanced_accuracy"],
        f"{prefix}/binary_accuracy": metrics["binary_accuracy"],
        f"{prefix}/binary_macro_precision": metrics["binary_macro_precision"],
        f"{prefix}/binary_macro_recall": metrics["binary_macro_recall"],
        f"{prefix}/binary_macro_f1": metrics["binary_macro_f1"],
        f"{prefix}/binary_balanced_accuracy": metrics["binary_balanced_accuracy"],
        f"{prefix}/malignant_precision": metrics["malignant_precision"],
        f"{prefix}/malignant_recall": metrics["malignant_recall"],
        f"{prefix}/malignant_specificity": metrics["malignant_specificity"],
        f"{prefix}/malignant_f1": metrics["malignant_f1"],
        f"{prefix}/benign_precision": metrics["benign_precision"],
        f"{prefix}/benign_recall": metrics["benign_recall"],
        f"{prefix}/benign_f1": metrics["benign_f1"],
    }


def print_epoch(epoch, train_metrics, val_metrics):
    print(
        f"Epoch {epoch}/{num_epochs} | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['accuracy']:.4f} | "
        f"Train Macro F1: {train_metrics['macro_f1']:.4f} | "
        f"Train Malig Recall: {train_metrics['malignant_recall']:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
        f"Val Malig Recall: {val_metrics['malignant_recall']:.4f}"
    )


def train_ham10000_model(model_type, image_source):
    torch.manual_seed(seed)

    if model_type not in MODEL_SETTINGS:
        raise ValueError(f"Unknown model_type: {model_type}")

    if image_source not in {"full_image", "lesion_white"}:
        raise ValueError(f"Unknown image_source: {image_source}")

    settings = MODEL_SETTINGS[model_type].copy()
    image_size = settings["image_size"]
    learning_rate = settings["learning_rate"]

    if image_source == "full_image":
        model_name = settings["checkpoint_name"]
        wandb_name = settings["display_name"]
        dataset_name = "HAM10000"
    else:
        model_name = f"{settings['checkpoint_name']}_lesion_white"
        wandb_name = f"{settings['display_name']}-lesion-white"
        dataset_name = "HAM10000-lesion-white"

    train_ds, val_ds, test_ds, train_df, val_df, test_df, class_weights = prepare_datasets(
        image_source,
        image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    config = {
        "model": settings["model"],
        "model_type": model_type,
        "dataset": dataset_name,
        "target": "multiclass_common_ham_pad_labels",
        "labels": COMMON_LABELS,
        "malignant_labels": sorted(MALIGNANT_LABELS),
        "image_source": image_source,
        "run_dir": RUN_DIR,
        "split_dir": SPLIT_DIR,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "image_size": image_size,
        "num_classes": len(COMMON_LABELS),
        "num_workers": num_workers,
        "class_weights": class_weights,
        "class_weight_note": "inverse frequency weights normalized to mean 1",
        "train_augmentation": TRAIN_AUGMENTATION_NOTE,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

    for key in ["patch_size", "embed_dim", "depth", "num_heads", "mlp_dim", "dropout"]:
        if key in settings:
            config[key] = settings[key]

    model = build_model(model_type, num_classes=len(COMMON_LABELS), config=config).to(device)

    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(
        project="skin-cancer-cnn",
        name=wandb_name,
        config=config,
    )

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    predictions_path = os.path.join(PRED_DIR, f"{model_name}_test_predictions.csv")
    metrics_path = os.path.join(PRED_DIR, f"{model_name}_test_metrics.csv")

    best_val_macro_f1 = -1.0
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            COMMON_LABELS,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            COMMON_LABELS,
        )

        log_row = {"epoch": epoch}
        log_row.update(wandb_metrics("train", train_metrics))
        log_row.update(wandb_metrics("val", val_metrics))
        wandb.log(log_row)

        print_epoch(epoch, train_metrics, val_metrics)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "labels": COMMON_LABELS,
                "malignant_labels": sorted(MALIGNANT_LABELS),
                "class_weights": class_weights,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_binary_accuracy": val_metrics["binary_accuracy"],
                "val_binary_macro_f1": val_metrics["binary_macro_f1"],
                "val_binary_balanced_accuracy": val_metrics["binary_balanced_accuracy"],
                "val_malignant_precision": val_metrics["malignant_precision"],
                "val_malignant_recall": val_metrics["malignant_recall"],
                "val_malignant_specificity": val_metrics["malignant_specificity"],
                "val_malignant_f1": val_metrics["malignant_f1"],
                "val_benign_recall": val_metrics["benign_recall"],
                "config": config,
            }

            torch.save(checkpoint, best_model_path)

            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_val_macro_f1"] = val_metrics["macro_f1"]
            wandb.run.summary["best_val_accuracy"] = val_metrics["accuracy"]
            wandb.run.summary["best_val_balanced_accuracy"] = val_metrics["balanced_accuracy"]
            wandb.run.summary["best_val_binary_accuracy"] = val_metrics["binary_accuracy"]
            wandb.run.summary["best_val_binary_macro_f1"] = val_metrics["binary_macro_f1"]
            wandb.run.summary["best_val_binary_balanced_accuracy"] = val_metrics["binary_balanced_accuracy"]
            wandb.run.summary["best_val_malignant_recall"] = val_metrics["malignant_recall"]
            wandb.run.summary["best_val_malignant_specificity"] = val_metrics["malignant_specificity"]
            wandb.run.summary["best_val_benign_recall"] = val_metrics["benign_recall"]

            wandb.save(best_model_path)

            print(f"Saved new best model: val_macro_f1={val_metrics['macro_f1']:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    checkpoint = load_checkpoint(best_model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = save_predictions(
        model,
        test_loader,
        criterion,
        device,
        COMMON_LABELS,
        predictions_path,
        metrics_path,
    )

    wandb.log(wandb_metrics("test", test_metrics))
    wandb.run.summary["test_loss"] = test_metrics["loss"]
    wandb.run.summary["test_accuracy"] = test_metrics["accuracy"]
    wandb.run.summary["test_macro_f1"] = test_metrics["macro_f1"]
    wandb.run.summary["test_balanced_accuracy"] = test_metrics["balanced_accuracy"]
    wandb.run.summary["test_binary_accuracy"] = test_metrics["binary_accuracy"]
    wandb.run.summary["test_binary_macro_f1"] = test_metrics["binary_macro_f1"]
    wandb.run.summary["test_binary_balanced_accuracy"] = test_metrics["binary_balanced_accuracy"]
    wandb.run.summary["test_malignant_recall"] = test_metrics["malignant_recall"]
    wandb.run.summary["test_malignant_specificity"] = test_metrics["malignant_specificity"]
    wandb.run.summary["test_benign_recall"] = test_metrics["benign_recall"]

    wandb.finish()
