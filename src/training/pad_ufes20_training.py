import os

import pandas as pd
import torch
import wandb

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from training.ham10000_training import (
    COMMON_LABELS,
    HAM_BINARY_LABELS,
    MALIGNANT_CLASS_WEIGHT_MULTIPLIER,
    MALIGNANT_LABELS,
    OVERLAP_LABELS,
    TARGET_SENSITIVITY,
    TRAIN_AUGMENTATION_NOTE,
    SkinLesionDataset,
    apply_temperature,
    batch_size,
    choose_threshold_for_sensitivity,
    collect_logits,
    compute_class_weights,
    evaluate,
    fit_temperature,
    load_checkpoint,
    malignant_scores_from_probs,
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
        df["dx"] = df["common_label"].str.lower()

    df["dx"] = df["dx"].astype(str).str.lower()
    df = df[df["dx"].isin(OVERLAP_LABELS)].copy().reset_index(drop=True)
    label_to_id = {label: i for i, label in enumerate(COMMON_LABELS)}
    df["common_label"] = df["dx"]
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
        "target": "binary_benign_malignant_overlap_labels",
        "overlap_labels": OVERLAP_LABELS,
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
        "class_weight_note": (
            "inverse frequency weights with malignant classes multiplied by "
            f"{MALIGNANT_CLASS_WEIGHT_MULTIPLIER}, normalized to mean 1"
        ),
        "malignant_class_weight_multiplier": MALIGNANT_CLASS_WEIGHT_MULTIPLIER,
        "scheduler": "cosine_annealing_lr",
        "target_sensitivity": TARGET_SENSITIVITY,
        "calibration": "temperature_scaling_on_validation_logits",
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
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Trainable parameters: {sum(parameter.numel() for parameter in trainable_parameters)}")

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

        log_row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
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
                "scheduler_state_dict": scheduler.state_dict(),
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

        scheduler.step()

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    checkpoint = load_checkpoint(best_model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_logits, val_targets = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_targets, device)
    val_probs = apply_temperature(val_logits, temperature)
    val_scores = malignant_scores_from_probs(val_probs, COMMON_LABELS, MALIGNANT_LABELS)
    malignant_threshold, val_operating_metrics = choose_threshold_for_sensitivity(
        val_targets,
        val_scores.tolist(),
        COMMON_LABELS,
        MALIGNANT_LABELS,
        TARGET_SENSITIVITY,
    )

    checkpoint["temperature"] = temperature
    checkpoint["malignant_threshold"] = malignant_threshold
    checkpoint["target_sensitivity"] = TARGET_SENSITIVITY
    checkpoint["val_operating_metrics"] = val_operating_metrics
    checkpoint["config"]["temperature"] = temperature
    checkpoint["config"]["malignant_threshold"] = malignant_threshold
    torch.save(checkpoint, best_model_path)

    print(f"Validation temperature: {temperature:.4f}")
    print(f"Validation operating threshold: {malignant_threshold:.4f}")
    print(f"Validation operating malignant recall: {val_operating_metrics['malignant_recall']:.4f}")
    print(f"Validation operating malignant specificity: {val_operating_metrics['malignant_specificity']:.4f}")

    val_operating_log = {
        f"val_operating/{name}": value
        for name, value in val_operating_metrics.items()
        if isinstance(value, (int, float))
    }
    val_operating_log["val_operating/temperature"] = temperature
    wandb.log(val_operating_log)

    test_metrics = save_predictions(
        model,
        test_loader,
        criterion,
        device,
        COMMON_LABELS,
        predictions_path,
        metrics_path,
        temperature=temperature,
        malignant_threshold=malignant_threshold,
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
    wandb.run.summary["operating_threshold"] = malignant_threshold
    wandb.run.summary["temperature"] = temperature
    wandb.run.summary["test_operating_malignant_recall"] = test_metrics["operating_malignant_recall"]
    wandb.run.summary["test_operating_malignant_specificity"] = test_metrics["operating_malignant_specificity"]

    wandb.finish()
