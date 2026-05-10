import os

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from training.ham10000_training import (
    BINARY_LABELS,
    HAM_BINARY_LABELS,
    MALIGNANT_LABELS,
    OVERLAP_LABELS,
    TARGET_SENSITIVITY,
    binary_metrics_from_scores,
    choose_threshold_for_sensitivity,
    operating_metrics_to_rows,
    scores_from_logits,
)
from models.model_architectures import build_model


DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")
MODEL_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")
SPLIT_DIR = os.path.join(RUN_DIR, "data", "splits")
PAD_IMAGE_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-images")
PAD_IMAGE_METADATA_PATH = os.path.join(PAD_IMAGE_DIR, "metadata.csv")
PAD_LESION_WHITE_IMAGE_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-lesion-white-images")
PAD_LESION_WHITE_METADATA_PATH = os.path.join(PAD_LESION_WHITE_IMAGE_DIR, "metadata.csv")

batch_size = 32
num_workers = 4


class PadImagesDataset(Dataset):
    def __init__(self, df, labels, transform):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.label_to_id = {label: i for i, label in enumerate(labels)}
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
            "labels": torch.tensor(self.label_to_id[row["binary_class"]], dtype=torch.long),
            "img_id": row["img_id"],
            "image_path": row["image_path"],
            "diagnostic": row["diagnostic"],
            "dx": row["dx"],
            "binary_class": row["binary_class"],
        }


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def compute_metrics(targets, preds, labels):
    num_classes = len(labels)
    total = len(targets)
    correct = sum(int(t == p) for t, p in zip(targets, preds))

    rows = [
        {"metric": "accuracy", "class": "overall", "value": correct / total if total > 0 else 0.0},
        {"metric": "num_examples", "class": "overall", "value": total},
    ]

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

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        rows.extend([
            {"metric": "support", "class": label, "value": support},
            {"metric": "precision", "class": label, "value": precision},
            {"metric": "recall", "class": label, "value": recall},
            {"metric": "specificity", "class": label, "value": specificity},
            {"metric": "f1", "class": label, "value": f1},
        ])

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

    rows.extend([
        {"metric": "macro_precision", "class": "overall", "value": sum(precisions) / num_classes},
        {"metric": "macro_recall", "class": "overall", "value": sum(recalls) / num_classes},
        {"metric": "macro_f1", "class": "overall", "value": sum(f1s) / num_classes},
        {"metric": "balanced_accuracy", "class": "overall", "value": sum(recalls) / num_classes},
        {"metric": "binary_accuracy", "class": "overall", "value": binary_accuracy},
        {"metric": "binary_macro_precision", "class": "overall", "value": binary_macro_precision},
        {"metric": "binary_macro_recall", "class": "overall", "value": binary_macro_recall},
        {"metric": "binary_macro_f1", "class": "overall", "value": binary_macro_f1},
        {"metric": "binary_balanced_accuracy", "class": "overall", "value": binary_macro_recall},
        {"metric": "sensitivity", "class": "malignant", "value": malignant_recall},
        {"metric": "specificity", "class": "benign", "value": malignant_specificity},
        {"metric": "malignant_precision", "class": "malignant", "value": malignant_precision},
        {"metric": "malignant_recall", "class": "malignant", "value": malignant_recall},
        {"metric": "malignant_specificity", "class": "benign", "value": malignant_specificity},
        {"metric": "malignant_f1", "class": "malignant", "value": malignant_f1},
        {"metric": "benign_precision", "class": "benign", "value": benign_precision},
        {"metric": "benign_recall", "class": "benign", "value": benign_recall},
        {"metric": "benign_specificity", "class": "malignant", "value": benign_specificity},
        {"metric": "benign_f1", "class": "benign", "value": benign_f1},
        {"metric": "binary_true_positive", "class": "malignant", "value": true_positive},
        {"metric": "binary_false_positive", "class": "malignant", "value": false_positive},
        {"metric": "binary_false_negative", "class": "malignant", "value": false_negative},
        {"metric": "binary_true_negative", "class": "benign", "value": true_negative},
        {"metric": "support", "class": "malignant", "value": true_positive + false_negative},
        {"metric": "support", "class": "benign", "value": true_negative + false_positive},
    ])

    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            count = sum(int(t == true_index and p == pred_index) for t, p in zip(targets, preds))
            rows.append({
                "metric": f"confusion_{true_label}_pred_{pred_label}",
                "class": "overall",
                "value": count,
            })

    return pd.DataFrame(rows)


def model_specs(image_source):
    if image_source == "full_image":
        return [
            {"name": "basic_cnn", "path": os.path.join(MODEL_DIR, "basic_cnn_best.pt"), "type": "cnn"},
            {"name": "vit", "path": os.path.join(MODEL_DIR, "vit_best.pt"), "type": "vit"},
            {"name": "resnet", "path": os.path.join(MODEL_DIR, "resnet_best.pt"), "type": "resnet"},
            {
                "name": "pretrained_resnet50",
                "path": os.path.join(MODEL_DIR, "pretrained_resnet50_best.pt"),
                "type": "pretrained_resnet50",
            },
        ]

    if image_source == "lesion_white":
        return [
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
            {
                "name": "resnet_lesion_white",
                "path": os.path.join(MODEL_DIR, "resnet_lesion_white_best.pt"),
                "type": "resnet",
            },
            {
                "name": "pretrained_resnet50_lesion_white",
                "path": os.path.join(MODEL_DIR, "pretrained_resnet50_lesion_white_best.pt"),
                "type": "pretrained_resnet50",
            },
        ]

    raise ValueError(f"Unknown image_source: {image_source}")


def prepare_pad_df(image_source):
    if image_source == "full_image":
        if not os.path.exists(PAD_IMAGE_METADATA_PATH):
            raise FileNotFoundError(
                f"Missing prepared PAD-UFES-20 metadata: {PAD_IMAGE_METADATA_PATH}. "
                "Run `sbatch submit/submit_create_data.sh` first."
            )
        df = pd.read_csv(PAD_IMAGE_METADATA_PATH)
        source_name = "full_image"
    elif image_source == "lesion_white":
        if not os.path.exists(PAD_LESION_WHITE_METADATA_PATH):
            raise FileNotFoundError(
                f"Missing prepared PAD-UFES-20 lesion-white metadata: {PAD_LESION_WHITE_METADATA_PATH}. "
                "Run `sbatch submit/submit_create_lesion_white_data.sh` first."
            )
        df = pd.read_csv(PAD_LESION_WHITE_METADATA_PATH)
        source_name = "lesion_white"
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    if "dx" not in df.columns:
        raise ValueError(
            "Prepared PAD-UFES-20 metadata is missing dx. "
            "Run `sbatch submit/submit_create_data.sh` to rebuild it."
        )

    missing_paths = df["image_path"].isna().sum()
    df = df.dropna(subset=["image_path", "dx"]).reset_index(drop=True)
    df["dx"] = df["dx"].astype(str).str.lower()
    df = df[df["dx"].isin(OVERLAP_LABELS)].copy().reset_index(drop=True)
    df["binary_class"] = df["dx"].map(HAM_BINARY_LABELS)
    df = df.dropna(subset=["binary_class"]).reset_index(drop=True)
    missing_files = (~df["image_path"].apply(os.path.exists)).sum()
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    print(f"Found PAD-UFES-20 {source_name}: {len(df)}")
    print(f"Missing PAD image paths before filtering: {missing_paths}")
    print(f"Missing PAD image files before filtering: {missing_files}")
    print("PAD dx counts:")
    print(df["dx"].value_counts().sort_index())
    print("PAD binary counts:")
    print(df["binary_class"].value_counts().sort_index())

    if len(df) == 0:
        raise FileNotFoundError(f"No PAD-UFES-20 images found for {source_name}.")

    return df, source_name


def prepare_ham_val_df(image_source):
    split_path = os.path.join(SPLIT_DIR, "ham10000_val.csv")

    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing HAM10000 validation split: {split_path}. "
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
    df["binary_class"] = df["dx"].map(HAM_BINARY_LABELS)
    df["img_id"] = df["image_id"]
    df["diagnostic"] = df["dx"]

    if image_source == "full_image":
        df["image_path"] = df["full_image_path"]
    elif image_source == "lesion_white":
        df["image_path"] = df["lesion_image_path"]
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    df = df.dropna(subset=["image_path", "binary_class"]).reset_index(drop=True)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    if len(df) == 0:
        raise FileNotFoundError(f"No HAM10000 validation images found for {image_source}.")

    return df


def predict_model(model_spec, df, device):
    if not os.path.exists(model_spec["path"]):
        print(f"Skipping missing model: {model_spec['path']}")
        return None

    checkpoint = load_checkpoint(model_spec["path"], device)
    labels = checkpoint.get("labels", BINARY_LABELS)
    if labels != BINARY_LABELS:
        raise ValueError(
            f"{model_spec['path']} was trained with labels {labels}. "
            "Retrain the model with the binary benign/malignant target before evaluation."
        )
    config = checkpoint.get("config", {})
    image_size = config.get("image_size", 224)
    if "malignant_threshold" not in checkpoint and "malignant_threshold" not in config:
        raise ValueError(
            f"{model_spec['path']} is missing the validation-selected malignant threshold. "
            "Retrain the model with the current binary training code."
        )
    malignant_threshold = checkpoint.get("malignant_threshold", config.get("malignant_threshold"))
    config["use_pretrained_backbone"] = False

    model = build_model(model_spec["type"], num_classes=1, config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    dataset = PadImagesDataset(df, labels, make_transform(image_size))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    targets = []
    scores_all = []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels_batch = batch["labels"].to(device, non_blocking=True)

            logits = model(images).view(-1)
            malignant_scores = scores_from_logits(logits)
            preds = (malignant_scores >= 0.5).long()

            targets.extend(labels_batch.cpu().tolist())
            scores_all.append(malignant_scores.cpu())

            preds_cpu = preds.cpu()
            labels_cpu = labels_batch.cpu()
            malignant_scores_cpu = malignant_scores.cpu()

            for i in range(images.size(0)):
                pred_index = int(preds_cpu[i])
                true_index = int(labels_cpu[i])
                malignant_score = float(malignant_scores_cpu[i])
                pred_binary_threshold = "malignant" if malignant_score >= malignant_threshold else "benign"

                row = {
                    "img_id": batch["img_id"][i],
                    "image_path": batch["image_path"][i],
                    "diagnostic": batch["diagnostic"][i],
                    "true_label": true_index,
                    "true_dx": batch["dx"][i],
                    "pred_label": pred_index,
                    "pred_dx": labels[pred_index],
                    "correct": int(pred_index == true_index),
                    "true_binary": "malignant" if labels[true_index] in MALIGNANT_LABELS else "benign",
                    "pred_binary": "malignant" if labels[pred_index] in MALIGNANT_LABELS else "benign",
                    "malignant_score": malignant_score,
                    "prob_benign": 1.0 - malignant_score,
                    "prob_malignant": malignant_score,
                    "operating_threshold": malignant_threshold,
                    "pred_binary_threshold": pred_binary_threshold,
                }
                row["binary_correct"] = int(row["true_binary"] == row["pred_binary"])
                row["binary_threshold_correct"] = int(row["true_binary"] == row["pred_binary_threshold"])

                rows.append(row)

    return {
        "labels": labels,
        "targets": targets,
        "scores": torch.cat(scores_all, dim=0),
        "rows": rows,
        "malignant_threshold": malignant_threshold,
    }


def save_model_predictions(
    rows,
    scores,
    labels,
    source_name,
    model_name,
    malignant_threshold=0.5,
):
    label_to_id = {label: i for i, label in enumerate(labels)}
    default_preds = (scores >= 0.5).long().tolist()
    threshold_preds = []

    output_rows = []
    for i, row in enumerate(rows):
        output_row = row.copy()
        default_index = default_preds[i]
        malignant_score = float(scores[i])
        pred_class = "malignant" if malignant_score >= malignant_threshold else "benign"
        pred_index = label_to_id[pred_class]
        threshold_preds.append(pred_index)

        output_row["default_label"] = default_index
        output_row["default_dx"] = labels[default_index]
        output_row["default_correct"] = int(default_index == output_row["true_label"])
        output_row["pred_label"] = pred_index
        output_row["pred_dx"] = pred_class
        output_row["correct"] = int(pred_index == output_row["true_label"])
        output_row["pred_binary"] = pred_class
        output_row["binary_correct"] = output_row["correct"]
        output_row["malignant_score"] = malignant_score
        output_row["prob_benign"] = 1.0 - malignant_score
        output_row["prob_malignant"] = malignant_score
        output_row["operating_threshold"] = malignant_threshold
        output_row["pred_binary_threshold"] = pred_class
        output_row["binary_threshold_correct"] = output_row["correct"]

        output_rows.append(output_row)

    targets = [row["true_label"] for row in output_rows]
    metrics_df = compute_metrics(targets, threshold_preds, labels)
    operating_metrics = binary_metrics_from_scores(
        targets,
        scores.tolist(),
        labels,
        MALIGNANT_LABELS,
        malignant_threshold,
    )
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame(operating_metrics_to_rows(operating_metrics)),
    ], ignore_index=True)

    os.makedirs(PRED_DIR, exist_ok=True)
    predictions_path = os.path.join(
        PRED_DIR,
        f"pad_ufes_20_{source_name}_{model_name}_predictions.csv",
    )
    metrics_path = os.path.join(
        PRED_DIR,
        f"pad_ufes_20_{source_name}_{model_name}_metrics.csv",
    )

    pd.DataFrame(output_rows).to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print(
        f"{model_name} sensitivity: {operating_metrics['malignant_recall']:.4f} | "
        f"specificity: {operating_metrics['malignant_specificity']:.4f} | "
        f"threshold: {malignant_threshold:.4f}"
    )
    if operating_metrics["malignant_recall"] < TARGET_SENSITIVITY:
        print(
            f"WARNING: {model_name} sensitivity is below target "
            f"{TARGET_SENSITIVITY:.2f} on PAD-UFES-20 {source_name}."
        )
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


def evaluate_model(model_spec, df, source_name, device):
    result = predict_model(model_spec, df, device)

    if result is None:
        return None

    save_model_predictions(
        result["rows"],
        result["scores"],
        result["labels"],
        source_name,
        model_spec["name"],
        malignant_threshold=result["malignant_threshold"],
    )

    return result


def choose_ensemble_threshold(model_specs_for_source, image_source, device):
    val_df = prepare_ham_val_df(image_source)
    val_results = []

    for model_spec in model_specs_for_source:
        result = predict_model(model_spec, val_df, device)
        if result is not None:
            val_results.append(result)

    if len(val_results) < 2:
        raise ValueError("Need at least two models to choose an ensemble threshold.")

    labels = val_results[0]["labels"]
    rows = val_results[0]["rows"]
    matching_results = [
        result for result in val_results
        if result["labels"] == labels and result["scores"].shape[0] == len(rows)
    ]

    if len(matching_results) < 2:
        raise ValueError("Need at least two compatible models to choose an ensemble threshold.")

    scores = torch.stack([result["scores"] for result in matching_results], dim=0).mean(dim=0)
    targets = matching_results[0]["targets"]
    threshold, metrics = choose_threshold_for_sensitivity(
        targets,
        scores.tolist(),
        labels,
        MALIGNANT_LABELS,
        TARGET_SENSITIVITY,
    )

    print(
        f"Ensemble HAM validation sensitivity: {metrics['malignant_recall']:.4f} | "
        f"specificity: {metrics['malignant_specificity']:.4f} | "
        f"threshold: {threshold:.4f}"
    )

    return threshold


def evaluate_ensemble(model_specs_for_source, model_results, image_source, source_name, device):
    model_results = [result for result in model_results if result is not None]

    if len(model_results) < 2:
        print(f"Skipping {source_name} ensemble because fewer than two models are available.")
        return

    labels = model_results[0]["labels"]
    rows = model_results[0]["rows"]
    matching_results = [
        result for result in model_results
        if result["labels"] == labels and result["scores"].shape[0] == len(rows)
    ]

    if len(matching_results) < 2:
        print(f"Skipping {source_name} ensemble because fewer than two compatible models are available.")
        return

    scores = torch.stack([result["scores"] for result in matching_results], dim=0).mean(dim=0)
    malignant_threshold = choose_ensemble_threshold(model_specs_for_source, image_source, device)

    save_model_predictions(
        rows,
        scores,
        labels,
        source_name,
        "ensemble",
        malignant_threshold=malignant_threshold,
    )


def evaluate_pad_ufes20(image_source):
    df, source_name = prepare_pad_df(image_source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    specs = model_specs(image_source)
    results = []
    for model_spec in specs:
        print(f"Evaluating {model_spec['name']}")
        results.append(evaluate_model(model_spec, df, source_name, device))

    print(f"Evaluating {source_name} ensemble")
    evaluate_ensemble(specs, results, image_source, source_name, device)
