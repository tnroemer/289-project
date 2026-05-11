import os

import pandas as pd
import torch
import wandb

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from training.ham10000_training import (
    BINARY_LABELS,
    HAM_BINARY_LABELS,
    MALIGNANT_LABELS,
    OVERLAP_LABELS,
    TARGET_SENSITIVITY,
    TRAIN_AUGMENTATION_NOTE,
    SkinLesionDataset,
    batch_size,
    choose_threshold_for_sensitivity,
    collect_logits,
    compute_pos_weight,
    evaluate,
    load_checkpoint,
    make_transforms,
    num_epochs,
    num_workers,
    print_epoch,
    save_predictions,
    scores_from_logits,
    seed,
    train_one_epoch,
    wandb_metrics,
)
from models.model_architectures import build_model


DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")
SPLIT_DIR = os.path.join(RUN_DIR, "data", "splits")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")
METRICS_DIR = os.path.join(RUN_DIR, "metrics")

image_size = 224
learning_rate = 3e-4


def load_pad_split(split_name):
    split_path = os.path.join(SPLIT_DIR, f"pad_ufes20_{split_name}.csv")

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing PAD-UFES-20 split file: {split_path}. "
            "Run `sbatch submit/submit_create_data.sh` first."
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

    missing_images = (~df["image_path"].apply(os.path.exists)).sum()
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    print(f"{split_name} examples found: {len(df)}")
    print(f"{split_name} missing images: {missing_images}")

    if len(df) == 0:
        raise FileNotFoundError(f"No PAD-UFES-20 images found for split {split_name}.")

    return df


def load_pad_splits():
    train_df = load_pad_split("train")
    val_df = load_pad_split("val")
    test_df = load_pad_split("test")

    return train_df, val_df, test_df


def train_pad_ufes20_full_image_resnet():
    torch.manual_seed(seed)

    train_df, val_df, test_df = load_pad_splits()
    train_transform, val_transform = make_transforms(image_size)

    train_ds = SkinLesionDataset(train_df, train_transform)
    val_ds = SkinLesionDataset(val_df, val_transform)
    test_ds = SkinLesionDataset(test_df, val_transform)

    pos_weight = compute_pos_weight(train_df)

    print("PAD-UFES-20 train counts:")
    print(train_df["dx"].value_counts().sort_index())
    print("PAD-UFES-20 binary train counts:")
    print(train_df["binary_class"].value_counts().sort_index())
    print(f"BCE pos_weight (negatives / positives): {pos_weight:.4f}")

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

    model_name = "pad_ufes20_resnet"
    config = {
        "model": "SimpleResNet",
        "model_type": "resnet",
        "dataset": "PAD-UFES-20",
        "target": "binary_benign_malignant_overlap_labels",
        "overlap_labels": OVERLAP_LABELS,
        "labels": BINARY_LABELS,
        "malignant_labels": sorted(MALIGNANT_LABELS),
        "image_source": "full_image",
        "run_dir": RUN_DIR,
        "split_dir": SPLIT_DIR,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "image_size": image_size,
        "num_classes": 1,
        "num_workers": num_workers,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": pos_weight,
        "pos_weight_note": "number of benign training examples divided by number of malignant training examples",
        "scheduler": "cosine_annealing_lr",
        "target_sensitivity": TARGET_SENSITIVITY,
        "threshold_selection": "validation_threshold_for_target_sensitivity_after_training",
        "model_selection": "highest_validation_specificity_at_target_sensitivity",
        "train_augmentation": TRAIN_AUGMENTATION_NOTE,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

    model = build_model("resnet", num_classes=1, config=config).to(device)

    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(
        project="skin-cancer-cnn",
        name="pad-ufes20-resnet",
        config=config,
    )

    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Trainable parameters: {sum(parameter.numel() for parameter in trainable_parameters)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    predictions_path = os.path.join(PRED_DIR, f"{model_name}_test_predictions.csv")
    metrics_path = os.path.join(METRICS_DIR, f"{model_name}_test_metrics.csv")

    best_val_specificity = -1.0
    patience = 20
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
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

        print_epoch(epoch, train_metrics, val_metrics, val_threshold, val_threshold_metrics)

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
