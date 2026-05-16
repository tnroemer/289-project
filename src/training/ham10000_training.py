import os

import pandas as pd
import torch
import wandb

from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.lora import is_lora_parameter_name
from models.model_architectures import build_model


# Configurable for reproducibility; defaults keep the repo self-contained.
# Override on a cluster via SKIN_LESIONS_DATA_ROOT / SKIN_LESIONS_RUN_DIR.
DATA_ROOT = os.environ.get(
    "SKIN_LESIONS_DATA_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
RUN_DIR = os.environ.get("SKIN_LESIONS_RUN_DIR", DATA_ROOT)
SPLIT_DIR = os.path.join(RUN_DIR, "data", "splits")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")
METRICS_DIR = os.path.join(RUN_DIR, "metrics")

OVERLAP_LABELS = ["akiec", "bcc", "bkl", "mel", "nv"]
BINARY_LABELS = ["benign", "malignant"]
MALIGNANT_LABELS = {"malignant"}
HAM_BINARY_LABELS = {
    "akiec": "malignant",
    "bcc": "malignant",
    "bkl": "benign",
    "mel": "malignant",
    "nv": "benign",
}
TARGET_SENSITIVITY = 0.90
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
        "image_size": 224,
        "learning_rate": 3e-4,
    },
    "vit": {
        "display_name": "vit",
        "checkpoint_name": "vit",
        "model": "ViT",
        "image_size": 224,
        "learning_rate": 2e-4,
        "patch_size": 16,
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
        "image_size": 224,
        "learning_rate": 3e-4,
    },
    "pretrained_resnet50": {
        "display_name": "pretrained-resnet50",
        "checkpoint_name": "pretrained_resnet50",
        "model": "ImageNetResNet50HeadThenLayer4",
        "image_size": 224,
        "learning_rate": 1e-3,
        "backbone_learning_rate": 1e-5,
        "unfreeze_layer4_epoch": 6,
        "use_pretrained_backbone": True,
        "finetune": "head_then_layer4",
    },
    # ------------------------------------------------------------------
    # ImageNet-pretrained backbones, two finetune variants each.
    # checkpoint_name == model_type so the filename suffix stays unambiguous
    # for bootstrap.py's parser (which strips a known suffix list, not a
    # greedy split). display_name uses hyphens for W&B readability.
    # head : freeze backbone, replace head with fresh Linear, train head only.
    # lora : freeze backbone, inject LoRA on a backbone-specific subtree,
    #        train LoRA A/B + fresh head.
    # ------------------------------------------------------------------
    "resnet18_head": {
        "display_name": "resnet18-head",
        "checkpoint_name": "resnet18_head",
        "model": "ImageNetResNet18HeadOnly",
        "image_size": 224,
        "learning_rate": 1e-3,
        "use_pretrained_backbone": True,
        "finetune": "head_only",
    },
    "resnet18_lora": {
        "display_name": "resnet18-lora",
        "checkpoint_name": "resnet18_lora",
        "model": "ImageNetResNet18LoRA(layer4)",
        "image_size": 224,
        "learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
        "use_pretrained_backbone": True,
        "finetune": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
    },
    "alexnet_head": {
        "display_name": "alexnet-head",
        "checkpoint_name": "alexnet_head",
        "model": "ImageNetAlexNetHeadOnly",
        "image_size": 224,
        "learning_rate": 1e-3,
        "use_pretrained_backbone": True,
        "finetune": "head_only",
    },
    "alexnet_lora": {
        "display_name": "alexnet-lora",
        "checkpoint_name": "alexnet_lora",
        "model": "ImageNetAlexNetLoRA(conv8,conv10,fc1,fc4)",
        "image_size": 224,
        "learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
        "use_pretrained_backbone": True,
        "finetune": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
    },
    "densenet121_head": {
        "display_name": "densenet121-head",
        "checkpoint_name": "densenet121_head",
        "model": "ImageNetDenseNet121HeadOnly",
        "image_size": 224,
        "learning_rate": 1e-3,
        "use_pretrained_backbone": True,
        "finetune": "head_only",
    },
    "densenet121_lora": {
        "display_name": "densenet121-lora",
        "checkpoint_name": "densenet121_lora",
        "model": "ImageNetDenseNet121LoRA(denseblock4)",
        "image_size": 224,
        "learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
        "use_pretrained_backbone": True,
        "finetune": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
    },
    "efficientnet_b0_head": {
        "display_name": "efficientnet_b0-head",
        "checkpoint_name": "efficientnet_b0_head",
        "model": "ImageNetEfficientNetB0HeadOnly",
        "image_size": 224,
        "learning_rate": 1e-3,
        "use_pretrained_backbone": True,
        "finetune": "head_only",
    },
    "efficientnet_b0_lora": {
        "display_name": "efficientnet_b0-lora",
        "checkpoint_name": "efficientnet_b0_lora",
        "model": "ImageNetEfficientNetB0LoRA(features7+8)",
        "image_size": 224,
        "learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
        "use_pretrained_backbone": True,
        "finetune": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
    },
    "convnext_tiny_head": {
        "display_name": "convnext_tiny-head",
        "checkpoint_name": "convnext_tiny_head",
        "model": "ImageNetConvNeXtTinyHeadOnly",
        "image_size": 224,
        "learning_rate": 1e-3,
        "use_pretrained_backbone": True,
        "finetune": "head_only",
    },
    "convnext_tiny_lora": {
        "display_name": "convnext_tiny-lora",
        "checkpoint_name": "convnext_tiny_lora",
        "model": "ImageNetConvNeXtTinyLoRA(features7)",
        "image_size": 224,
        "learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
        "use_pretrained_backbone": True,
        "finetune": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
    },
    "vgg16_head": {
        "display_name": "vgg16-head",
        "checkpoint_name": "vgg16_head",
        "model": "ImageNetVGG16HeadOnly",
        "image_size": 224,
        "learning_rate": 1e-3,
        "use_pretrained_backbone": True,
        "finetune": "head_only",
        "batch_size": 16,  # 138M params -> halve batch to fit single GPU
    },
    "vgg16_lora": {
        "display_name": "vgg16-lora",
        "checkpoint_name": "vgg16_lora",
        "model": "ImageNetVGG16LoRA(conv26,conv28,fc0,fc3)",
        "image_size": 224,
        "learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
        "use_pretrained_backbone": True,
        "finetune": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
        "batch_size": 16,
    },
    "googlenet_head": {
        "display_name": "googlenet-head",
        "checkpoint_name": "googlenet_head",
        "model": "ImageNetGoogLeNetHeadOnly",
        "image_size": 224,
        "learning_rate": 1e-3,
        "use_pretrained_backbone": True,
        "finetune": "head_only",
    },
    "googlenet_lora": {
        "display_name": "googlenet-lora",
        "checkpoint_name": "googlenet_lora",
        "model": "ImageNetGoogLeNetLoRA(inception5a+5b)",
        "image_size": 224,
        "learning_rate": 1e-3,
        "backbone_learning_rate": 1e-4,
        "use_pretrained_backbone": True,
        "finetune": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
    },
    # ------------------------------------------------------------------
    # Small-capacity baseline trained from scratch (no torchvision weights).
    # Uses BasicCNN's normalization (ImageNet stats) for input consistency.
    # ------------------------------------------------------------------
    "lenet5": {
        "display_name": "lenet5",
        "checkpoint_name": "lenet5",
        "model": "LeNet5",
        "image_size": 224,
        "learning_rate": 3e-4,
        "use_pretrained_backbone": False,
        "finetune": "scratch",
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
            "binary_class": row["binary_class"],
        }


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def scores_from_logits(logits):
    return torch.sigmoid(logits.view(-1))


def binary_metrics_from_scores(targets, scores, labels, malignant_labels, threshold):
    malignant_indices = {i for i, label in enumerate(labels) if label in malignant_labels}
    true_malignant = [target in malignant_indices for target in targets]
    pred_malignant = [float(score) >= threshold for score in scores]
    total = len(targets)

    true_positive = sum(int(t and p) for t, p in zip(true_malignant, pred_malignant))
    false_positive = sum(int((not t) and p) for t, p in zip(true_malignant, pred_malignant))
    false_negative = sum(int(t and (not p)) for t, p in zip(true_malignant, pred_malignant))
    true_negative = sum(int((not t) and (not p)) for t, p in zip(true_malignant, pred_malignant))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "threshold": threshold,
        "target_sensitivity": TARGET_SENSITIVITY,
        "accuracy": (true_positive + true_negative) / total if total > 0 else 0.0,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "malignant_support": true_positive + false_negative,
        "benign_support": true_negative + false_positive,
        "num_examples": total,
    }


def choose_threshold_for_sensitivity(targets, scores, labels, malignant_labels, target_sensitivity=TARGET_SENSITIVITY):
    candidate_thresholds = sorted(
        {0.0, 1.0}.union({float(score) for score in scores}),
        reverse=True,
    )
    best_metrics = None

    for threshold in candidate_thresholds:
        metrics = binary_metrics_from_scores(targets, scores, labels, malignant_labels, threshold)
        metrics["target_sensitivity"] = target_sensitivity
        if metrics["sensitivity"] >= target_sensitivity:
            if best_metrics is None or metrics["specificity"] > best_metrics["specificity"]:
                best_metrics = metrics

    if best_metrics is None:
        best_metrics = binary_metrics_from_scores(targets, scores, labels, malignant_labels, 0.0)
        best_metrics["target_sensitivity"] = target_sensitivity

    return best_metrics["threshold"], best_metrics


def metrics_to_rows(metrics):
    metric_names = [
        "loss",
        "threshold",
        "target_sensitivity",
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "malignant_support",
        "benign_support",
        "num_examples",
    ]

    return [
        {"metric": name, "value": metrics[name]}
        for name in metric_names
        if name in metrics
    ]


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

    if "dx" not in df.columns:
        raise ValueError(
            f"Missing dx column in {split_path}. "
            "Run `sbatch submit/submit_create_data.sh` to rebuild binary metadata."
        )

    df["dx"] = df["dx"].astype(str).str.lower()
    df = df[df["dx"].isin(OVERLAP_LABELS)].copy().reset_index(drop=True)
    label_to_id = {label: i for i, label in enumerate(BINARY_LABELS)}
    df["binary_class"] = df["dx"].map(HAM_BINARY_LABELS)
    df["binary_label"] = df["binary_class"].map(label_to_id)
    df["label"] = df["binary_label"]

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


def compute_pos_weight(train_df):
    positives = int((train_df["label"] == 1).sum())
    negatives = int((train_df["label"] == 0).sum())

    if positives == 0:
        raise ValueError("No malignant training examples found.")
    if negatives == 0:
        raise ValueError("No benign training examples found.")

    return negatives / positives


def prepare_datasets(image_source, image_size):
    train_df = load_split("train", image_source)
    val_df = load_split("val", image_source)
    test_df = load_split("test", image_source)

    train_transform, val_transform = make_transforms(image_size)

    train_ds = SkinLesionDataset(train_df, train_transform)
    val_ds = SkinLesionDataset(val_df, val_transform)
    test_ds = SkinLesionDataset(test_df, val_transform)

    pos_weight = compute_pos_weight(train_df)

    print("Train counts:")
    print(train_df["dx"].value_counts().sort_index())
    print("Binary train counts:")
    print(train_df["binary_class"].value_counts().sort_index())
    print(f"BCE pos_weight (negatives / positives): {pos_weight:.4f}")

    return train_ds, val_ds, test_ds, train_df, val_df, test_df, pos_weight


def compute_metrics(targets, preds, labels):
    total = len(targets)
    malignant_indices = [i for i, label in enumerate(labels) if label in MALIGNANT_LABELS]
    true_malignant = [t in malignant_indices for t in targets]
    pred_malignant = [p in malignant_indices for p in preds]

    true_positive = sum(int(t and p) for t, p in zip(true_malignant, pred_malignant))
    false_positive = sum(int((not t) and p) for t, p in zip(true_malignant, pred_malignant))
    false_negative = sum(int(t and (not p)) for t, p in zip(true_malignant, pred_malignant))
    true_negative = sum(int((not t) and (not p)) for t, p in zip(true_malignant, pred_malignant))

    accuracy = (true_positive + true_negative) / total if total > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "malignant_support": true_positive + false_negative,
        "benign_support": true_negative + false_positive,
        "num_examples": total,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, labels):
    model.train()

    if getattr(model, "freeze_backbone", False):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    total_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels_batch = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images).view(-1)
        loss = criterion(logits, labels_batch.float())

        loss.backward()
        optimizer.step()

        preds = (scores_from_logits(logits) >= 0.5).long()

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

            logits = model(images).view(-1)
            loss = criterion(logits, labels_batch.float())
            preds = (scores_from_logits(logits) >= 0.5).long()

            total_loss += loss.item() * images.size(0)
            total += labels_batch.size(0)
            all_targets.extend(labels_batch.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds, labels)
    metrics["loss"] = total_loss / total

    return metrics


def collect_logits(model, loader, device):
    model.eval()

    logits = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels_batch = batch["labels"].to(device, non_blocking=True)
            outputs = model(images).view(-1)

            logits.append(outputs.cpu())
            targets.extend(labels_batch.cpu().tolist())

    return torch.cat(logits, dim=0), targets


def write_metrics_csv(metrics, metrics_path):
    pd.DataFrame(metrics_to_rows(metrics)).to_csv(metrics_path, index=False)


def save_predictions(
    model,
    loader,
    criterion,
    device,
    labels,
    predictions_path,
    metrics_path,
    malignant_threshold=None,
):
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

            logits = model(images).view(-1)
            loss = criterion(logits, labels_batch.float())
            malignant_scores = scores_from_logits(logits)
            default_preds = (malignant_scores >= 0.5).long()
            if malignant_threshold is None:
                preds = default_preds
            else:
                preds = (malignant_scores >= malignant_threshold).long()

            total_loss += loss.item() * images.size(0)
            total += labels_batch.size(0)
            all_targets.extend(labels_batch.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            default_preds_cpu = default_preds.cpu()
            preds_cpu = preds.cpu()
            labels_cpu = labels_batch.cpu()
            malignant_scores_cpu = malignant_scores.cpu()

            for i in range(images.size(0)):
                default_index = int(default_preds_cpu[i])
                pred_index = int(preds_cpu[i])
                true_index = int(labels_cpu[i])
                malignant_score = float(malignant_scores_cpu[i])

                row = {
                    "image_id": batch["image_id"][i],
                    "image_path": batch["image_path"][i],
                    "true_label": true_index,
                    "true_dx": batch["dx"][i],
                    "true_class": batch["binary_class"][i],
                    "default_label": default_index,
                    "default_class": labels[default_index],
                    "default_correct": int(default_index == true_index),
                    "pred_label": pred_index,
                    "pred_class": labels[pred_index],
                    "correct": int(pred_index == true_index),
                    "malignant_score": malignant_score,
                    "prob_benign": 1.0 - malignant_score,
                    "prob_malignant": malignant_score,
                    "threshold": malignant_threshold,
                }

                rows.append(row)

    metrics = compute_metrics(all_targets, all_preds, labels)
    metrics["loss"] = total_loss / total

    if malignant_threshold is not None:
        metrics["threshold"] = malignant_threshold
        metrics["target_sensitivity"] = TARGET_SENSITIVITY

    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    write_metrics_csv(metrics, metrics_path)

    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {metrics['specificity']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Test Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    if malignant_threshold is not None:
        print(f"Threshold: {malignant_threshold:.4f}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")

    return metrics


def wandb_metrics(prefix, metrics):
    rows = {
        f"{prefix}/loss": metrics["loss"],
        f"{prefix}/accuracy": metrics["accuracy"],
        f"{prefix}/precision": metrics["precision"],
        f"{prefix}/sensitivity": metrics["sensitivity"],
        f"{prefix}/specificity": metrics["specificity"],
        f"{prefix}/f1": metrics["f1"],
        f"{prefix}/balanced_accuracy": metrics["balanced_accuracy"],
    }

    return rows


def print_epoch(epoch, train_metrics, val_metrics, val_threshold, val_threshold_metrics, total_epochs=None):
    total = total_epochs if total_epochs is not None else num_epochs
    print(
        f"Epoch {epoch}/{total} | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Threshold: {val_threshold:.4f} | "
        f"Val Sens: {val_threshold_metrics['sensitivity']:.4f} | "
        f"Val Spec: {val_threshold_metrics['specificity']:.4f}"
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

    # Per-model overrides for previously-hardcoded knobs. Defaults match the
    # module-level constants so existing entries (cnn, vit, resnet,
    # pretrained_resnet50) keep their current behavior unchanged.
    local_batch_size = settings.get("batch_size", batch_size)
    local_num_epochs = settings.get("num_epochs", num_epochs)
    local_num_workers = settings.get("num_workers", num_workers)
    local_weight_decay = settings.get("weight_decay", 1e-4)
    local_patience = settings.get("early_stopping_patience", 20)
    finetune_mode = settings.get("finetune")

    if image_source == "full_image":
        model_name = settings["checkpoint_name"]
        wandb_name = settings["display_name"]
        dataset_name = "HAM10000"
    else:
        model_name = f"{settings['checkpoint_name']}_lesion_white"
        wandb_name = f"{settings['display_name']}-lesion-white"
        dataset_name = "HAM10000-lesion-white"

    train_ds, val_ds, test_ds, train_df, val_df, test_df, pos_weight = prepare_datasets(
        image_source,
        image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=local_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=local_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=local_num_workers,
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
        "target": "binary_benign_malignant_overlap_labels",
        "overlap_labels": OVERLAP_LABELS,
        "labels": BINARY_LABELS,
        "malignant_labels": sorted(MALIGNANT_LABELS),
        "image_source": image_source,
        "run_dir": RUN_DIR,
        "split_dir": SPLIT_DIR,
        "batch_size": local_batch_size,
        "learning_rate": learning_rate,
        "epochs": local_num_epochs,
        "image_size": image_size,
        "num_classes": 1,
        "num_workers": local_num_workers,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": pos_weight,
        "pos_weight_note": "number of benign training examples divided by number of malignant training examples",
        "optimizer": "AdamW",
        "weight_decay": local_weight_decay,
        "scheduler": "cosine_annealing_lr",
        "early_stopping_patience": local_patience,
        "target_sensitivity": TARGET_SENSITIVITY,
        "threshold_selection": "validation_threshold_for_target_sensitivity_after_training",
        "model_selection": "highest_validation_specificity_at_target_sensitivity",
        "train_augmentation": TRAIN_AUGMENTATION_NOTE,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

    for key in [
        "patch_size",
        "embed_dim",
        "depth",
        "num_heads",
        "mlp_dim",
        "dropout",
        "use_pretrained_backbone",
        "backbone_learning_rate",
        "unfreeze_layer4_epoch",
        "finetune",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
    ]:
        if key in settings:
            config[key] = settings[key]

    model = build_model(model_type, num_classes=1, config=config).to(device)

    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(
        project="skin-cancer-cnn",
        name=wandb_name,
        config=config,
    )

    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    if finetune_mode == "head_then_layer4":
        head_parameters = [parameter for parameter in model.fc.parameters() if parameter.requires_grad]
        layer4_parameters = list(model.layer4.parameters())
        optimizer = AdamW(
            [
                {"params": head_parameters, "lr": learning_rate},
                {"params": layer4_parameters, "lr": settings["backbone_learning_rate"]},
            ],
            weight_decay=local_weight_decay,
        )
        trainable_parameters = head_parameters
    elif finetune_mode == "lora":
        head_params = []
        lora_params = []
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if is_lora_parameter_name(parameter_name):
                lora_params.append(parameter)
            else:
                head_params.append(parameter)
        if not lora_params:
            raise RuntimeError(
                f"finetune=lora but no LoRA parameters found in {model_type}. "
                "Check that build_model injected LoRA wrappers."
            )
        optimizer = AdamW(
            [
                {"params": head_params, "lr": learning_rate},
                {"params": lora_params, "lr": settings["backbone_learning_rate"]},
            ],
            weight_decay=local_weight_decay,
        )
        trainable_parameters = head_params + lora_params
    else:
        # head_only, scratch, or unset: one parameter group at head LR.
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = AdamW(trainable_parameters, lr=learning_rate, weight_decay=local_weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=local_num_epochs)

    print(f"Trainable parameters: {sum(parameter.numel() for parameter in trainable_parameters)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    predictions_path = os.path.join(PRED_DIR, f"{model_name}_test_predictions.csv")
    metrics_path = os.path.join(METRICS_DIR, f"{model_name}_test_metrics.csv")

    best_val_specificity = -1.0
    patience = local_patience
    epochs_without_improvement = 0
    layer4_unfrozen = False

    for epoch in range(1, local_num_epochs + 1):
        if (
            finetune_mode == "head_then_layer4"
            and not layer4_unfrozen
            and epoch >= settings["unfreeze_layer4_epoch"]
        ):
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
            layer4_unfrozen = True
            print(f"Unfroze pretrained {model_type} layer4 at epoch {epoch}.")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            BINARY_LABELS,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            BINARY_LABELS,
        )
        val_logits, val_targets = collect_logits(model, val_loader, device)
        val_scores = scores_from_logits(val_logits)
        val_threshold, val_threshold_metrics = choose_threshold_for_sensitivity(
            val_targets,
            val_scores.tolist(),
            BINARY_LABELS,
            MALIGNANT_LABELS,
            TARGET_SENSITIVITY,
        )

        log_row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        log_row["train/loss"] = train_metrics["loss"]
        log_row["val/loss"] = val_metrics["loss"]
        log_row.update({
            f"val/{name}": value
            for name, value in val_threshold_metrics.items()
            if isinstance(value, (int, float))
        })
        wandb.log(log_row)

        print_epoch(epoch, train_metrics, val_metrics, val_threshold, val_threshold_metrics, total_epochs=local_num_epochs)

        val_specificity = val_threshold_metrics["specificity"]
        if val_specificity > best_val_specificity:
            best_val_specificity = val_specificity
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "labels": BINARY_LABELS,
                "malignant_labels": sorted(MALIGNANT_LABELS),
                "pos_weight": pos_weight,
                "malignant_threshold": val_threshold,
                "target_sensitivity": TARGET_SENSITIVITY,
                "val_metrics": val_threshold_metrics,
                "selection_metric": "val_specificity_at_target_sensitivity",
                "val_loss": val_metrics["loss"],
                "config": {
                    **config,
                    "malignant_threshold": val_threshold,
                    "selection_metric": "val_specificity_at_target_sensitivity",
                },
            }

            torch.save(checkpoint, best_model_path)

            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_val_threshold"] = val_threshold
            wandb.run.summary["best_val_sensitivity"] = val_threshold_metrics["sensitivity"]
            wandb.run.summary["best_val_specificity"] = val_threshold_metrics["specificity"]
            wandb.run.summary["best_val_accuracy"] = val_threshold_metrics["accuracy"]
            wandb.run.summary["best_val_precision"] = val_threshold_metrics["precision"]
            wandb.run.summary["best_val_f1"] = val_threshold_metrics["f1"]
            wandb.run.summary["best_val_balanced_accuracy"] = val_threshold_metrics["balanced_accuracy"]

            wandb.save(best_model_path)

            print(
                "Saved new best model: "
                f"val_specificity={val_specificity:.4f}"
            )
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    checkpoint = load_checkpoint(best_model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_logits, val_targets = collect_logits(model, val_loader, device)
    val_scores = scores_from_logits(val_logits)
    malignant_threshold, val_threshold_metrics = choose_threshold_for_sensitivity(
        val_targets,
        val_scores.tolist(),
        BINARY_LABELS,
        MALIGNANT_LABELS,
        TARGET_SENSITIVITY,
    )

    checkpoint["malignant_threshold"] = malignant_threshold
    checkpoint["target_sensitivity"] = TARGET_SENSITIVITY
    checkpoint["val_metrics"] = val_threshold_metrics
    checkpoint["config"]["malignant_threshold"] = malignant_threshold
    torch.save(checkpoint, best_model_path)

    print(f"Validation threshold: {malignant_threshold:.4f}")
    print(f"Validation sensitivity: {val_threshold_metrics['sensitivity']:.4f}")
    print(f"Validation specificity: {val_threshold_metrics['specificity']:.4f}")

    val_log = {
        f"val/{name}": value
        for name, value in val_threshold_metrics.items()
        if isinstance(value, (int, float))
    }
    wandb.log(val_log)

    test_metrics = save_predictions(
        model,
        test_loader,
        criterion,
        device,
        BINARY_LABELS,
        predictions_path,
        metrics_path,
        malignant_threshold=malignant_threshold,
    )

    wandb.log(wandb_metrics("test", test_metrics))
    wandb.run.summary["test_loss"] = test_metrics["loss"]
    wandb.run.summary["test_accuracy"] = test_metrics["accuracy"]
    wandb.run.summary["test_precision"] = test_metrics["precision"]
    wandb.run.summary["test_sensitivity"] = test_metrics["sensitivity"]
    wandb.run.summary["test_specificity"] = test_metrics["specificity"]
    wandb.run.summary["test_f1"] = test_metrics["f1"]
    wandb.run.summary["test_balanced_accuracy"] = test_metrics["balanced_accuracy"]
    wandb.run.summary["threshold"] = malignant_threshold

    wandb.finish()
