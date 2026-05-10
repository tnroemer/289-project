import os

import pandas as pd
import torch
import wandb

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from training.ham10000_training import (
    COMMON_LABELS,
    MALIGNANT_LABELS,
    TRAIN_AUGMENTATION_NOTE,
    SkinLesionDataset,
    batch_size,
    compute_class_weights,
    evaluate,
    load_checkpoint,
    make_transforms,
    num_epochs,
    num_workers,
    print_epoch,
    save_predictions,
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
PAD_METADATA_PATH = os.path.join(RUN_DIR, "data", "pad-ufes-20-images", "metadata.csv")

image_size = 128
learning_rate = 3e-4


def split_group(group):
    group = group.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(group)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))

    test_df = group.iloc[:n_test]
    val_df = group.iloc[n_test:n_test + n_val]
    train_df = group.iloc[n_test + n_val:]

    return train_df, val_df, test_df


def prepare_pad_splits():
    if not os.path.exists(PAD_METADATA_PATH):
        raise FileNotFoundError(
            f"Missing prepared PAD-UFES-20 metadata: {PAD_METADATA_PATH}. "
            "Run `sbatch submit/submit_create_data.sh` first."
        )

    df = pd.read_csv(PAD_METADATA_PATH)
    df = df.dropna(subset=["image_path", "common_label"]).copy().reset_index(drop=True)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    df["dx"] = df["common_label"].str.lower()
    df = df[df["dx"].isin(COMMON_LABELS)].copy().reset_index(drop=True)

    label_to_id = {label: i for i, label in enumerate(COMMON_LABELS)}
    df["label"] = df["dx"].map(label_to_id)
    df["image_id"] = df["image_id"].astype(str)

    if len(df) == 0:
        raise FileNotFoundError("No prepared PAD-UFES-20 full images found.")

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby("label"):
        train_df, val_df, test_df = split_group(group)
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    os.makedirs(SPLIT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SPLIT_DIR, "pad_ufes20_train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, "pad_ufes20_val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "pad_ufes20_test.csv"), index=False)
    pd.concat([train_df, val_df, test_df]).to_csv(
        os.path.join(SPLIT_DIR, "pad_ufes20_all.csv"),
        index=False,
    )

    return train_df, val_df, test_df


def train_pad_ufes20_full_image_resnet():
    torch.manual_seed(seed)

    train_df, val_df, test_df = prepare_pad_splits()
    train_transform, val_transform = make_transforms(image_size)

    train_ds = SkinLesionDataset(train_df, train_transform)
    val_ds = SkinLesionDataset(val_df, val_transform)
    test_ds = SkinLesionDataset(test_df, val_transform)

    class_weights = compute_class_weights(train_df, COMMON_LABELS)

    print("PAD-UFES-20 train counts:")
    print(train_df["dx"].value_counts().sort_index())
    print("Class weights:")
    for label, weight in zip(COMMON_LABELS, class_weights):
        print(f"{label}: {weight:.4f}")

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
        "target": "multiclass_common_ham_pad_labels",
        "labels": COMMON_LABELS,
        "malignant_labels": sorted(MALIGNANT_LABELS),
        "image_source": "full_image",
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

    model = build_model("resnet", num_classes=len(COMMON_LABELS), config=config).to(device)

    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(
        project="skin-cancer-cnn",
        name="pad-ufes20-resnet",
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
                "val_malignant_precision": val_metrics["malignant_precision"],
                "val_malignant_recall": val_metrics["malignant_recall"],
                "val_malignant_specificity": val_metrics["malignant_specificity"],
                "val_malignant_f1": val_metrics["malignant_f1"],
                "config": config,
            }

            torch.save(checkpoint, best_model_path)

            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_val_macro_f1"] = val_metrics["macro_f1"]
            wandb.run.summary["best_val_accuracy"] = val_metrics["accuracy"]
            wandb.run.summary["best_val_balanced_accuracy"] = val_metrics["balanced_accuracy"]
            wandb.run.summary["best_val_malignant_recall"] = val_metrics["malignant_recall"]
            wandb.run.summary["best_val_malignant_specificity"] = val_metrics["malignant_specificity"]

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
    wandb.run.summary["test_malignant_recall"] = test_metrics["malignant_recall"]
    wandb.run.summary["test_malignant_specificity"] = test_metrics["malignant_specificity"]

    wandb.finish()
