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

from model_architectures import build_model


os.environ["KAGGLEHUB_CACHE"] = "/ocean/projects/mth250011p/troemer/"

DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")
DATASET_DIR = os.path.join(DATA_ROOT, "datasets", "pad-ufes-20")
MODEL_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "preds")
EXTRACTED_DIR = os.path.join(RUN_DIR, "data", "pad-ufes-20-extracted-lesions")
EXTRACTED_MANIFEST_PATH = os.path.join(EXTRACTED_DIR, "metadata.csv")

COMMON_LABELS = ["akiec", "bcc", "bkl", "mel", "nv"]
MALIGNANT_LABELS = {"akiec", "bcc", "mel"}
PAD_TO_COMMON_LABELS = {
    "ACK": "akiec",
    "BCC": "bcc",
    "SEK": "bkl",
    "MEL": "mel",
    "NEV": "nv",
}

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
            "labels": torch.tensor(self.label_to_id[row["common_label"]], dtype=torch.long),
            "img_id": row["img_id"],
            "image_path": row["image_path"],
            "diagnostic": row["diagnostic"],
            "common_label": row["common_label"],
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

    malignant_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    malignant_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    malignant_specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    malignant_f1 = (
        2 * malignant_precision * malignant_recall / (malignant_precision + malignant_recall)
        if malignant_precision + malignant_recall > 0
        else 0.0
    )

    rows.extend([
        {"metric": "macro_precision", "class": "overall", "value": sum(precisions) / num_classes},
        {"metric": "macro_recall", "class": "overall", "value": sum(recalls) / num_classes},
        {"metric": "macro_f1", "class": "overall", "value": sum(f1s) / num_classes},
        {"metric": "balanced_accuracy", "class": "overall", "value": sum(recalls) / num_classes},
        {"metric": "malignant_precision", "class": "malignant", "value": malignant_precision},
        {"metric": "malignant_recall", "class": "malignant", "value": malignant_recall},
        {"metric": "malignant_specificity", "class": "benign", "value": malignant_specificity},
        {"metric": "malignant_f1", "class": "malignant", "value": malignant_f1},
        {"metric": "true_positive", "class": "malignant", "value": true_positive},
        {"metric": "false_positive", "class": "malignant", "value": false_positive},
        {"metric": "false_negative", "class": "malignant", "value": false_negative},
        {"metric": "true_negative", "class": "benign", "value": true_negative},
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
        ]

    raise ValueError(f"Unknown image_source: {image_source}")


def prepare_full_image_df():
    dataset_dir = download_dataset()
    metadata_path = find_metadata_csv(dataset_dir)
    image_index = make_image_index(dataset_dir)

    df = pd.read_csv(metadata_path)
    df["image_path"] = df["img_id"].apply(
        lambda x: image_index.get(str(x)) or image_index.get(os.path.splitext(str(x))[0])
    )

    print(f"Metadata path: {metadata_path}")
    return df


def prepare_extracted_lesion_df():
    if not os.path.exists(EXTRACTED_MANIFEST_PATH):
        raise FileNotFoundError(
            "Could not find extracted PAD-UFES-20 lesion manifest. "
            "Run create_pad_ufes20_extracted_lesions.py first."
        )

    df = pd.read_csv(EXTRACTED_MANIFEST_PATH)
    df["image_path"] = df["extracted_image_path"]
    return df


def prepare_pad_df(image_source):
    if image_source == "full_image":
        df = prepare_full_image_df()
        source_name = "full_images"
    elif image_source == "lesion_white":
        df = prepare_extracted_lesion_df()
        source_name = "extracted_lesions"
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    df["diagnostic"] = df["diagnostic"].astype(str).str.upper()
    df["common_label"] = df["diagnostic"].map(PAD_TO_COMMON_LABELS)

    missing_images = df["image_path"].isna().sum()
    df = df.dropna(subset=["image_path", "common_label"]).reset_index(drop=True)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    print(f"Found PAD-UFES-20 {source_name}: {len(df)}")
    print(f"Missing PAD image paths before filtering: {missing_images}")
    print("PAD common-label counts:")
    print(df["common_label"].value_counts().sort_index())

    return df, source_name


def evaluate_model(model_spec, df, source_name, device):
    if not os.path.exists(model_spec["path"]):
        print(f"Skipping missing model: {model_spec['path']}")
        return

    checkpoint = load_checkpoint(model_spec["path"], device)
    labels = checkpoint.get("labels", COMMON_LABELS)
    config = checkpoint.get("config", {})
    image_size = config.get("image_size", 128 if model_spec["type"] != "vit" else 96)

    model = build_model(model_spec["type"], num_classes=len(labels), config=config)
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
    preds_all = []

    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels_batch = batch["labels"].to(device, non_blocking=True)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            targets.extend(labels_batch.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())

            probs_cpu = probs.cpu()
            preds_cpu = preds.cpu()
            labels_cpu = labels_batch.cpu()

            for i in range(images.size(0)):
                pred_index = int(preds_cpu[i])
                true_index = int(labels_cpu[i])

                row = {
                    "img_id": batch["img_id"][i],
                    "image_path": batch["image_path"][i],
                    "diagnostic": batch["diagnostic"][i],
                    "true_label": true_index,
                    "true_dx": batch["common_label"][i],
                    "pred_label": pred_index,
                    "pred_dx": labels[pred_index],
                    "correct": int(pred_index == true_index),
                }

                for j, label in enumerate(labels):
                    row[f"prob_{label}"] = float(probs_cpu[i, j])

                rows.append(row)

    pred_df = pd.DataFrame(rows)
    metrics_df = compute_metrics(targets, preds_all, labels)

    os.makedirs(PRED_DIR, exist_ok=True)
    predictions_path = os.path.join(
        PRED_DIR,
        f"pad_ufes_20_{source_name}_{model_spec['name']}_predictions.csv",
    )
    metrics_path = os.path.join(
        PRED_DIR,
        f"pad_ufes_20_{source_name}_{model_spec['name']}_metrics.csv",
    )

    pred_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


def evaluate_pad_ufes20(image_source):
    df, source_name = prepare_pad_df(image_source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for model_spec in model_specs(image_source):
        print(f"Evaluating {model_spec['name']}")
        evaluate_model(model_spec, df, source_name, device)
