import os

import numpy as np
import pandas as pd


DATA_ROOT = "/ocean/projects/mth250011p/troemer"
RUN_DIR = os.path.join(DATA_ROOT, "skin-lesions")
PRED_DIR = os.path.join(RUN_DIR, "preds")
METRICS_DIR = os.path.join(RUN_DIR, "metrics")

n_bootstrap = 5000
seed = 42


def clean_model_name(model_name):
    if model_name.endswith("_lesion_white"):
        model_name = model_name[: -len("_lesion_white")]

    if model_name == "basic_cnn":
        return "cnn"
    if model_name == "pretrained_resnet50":
        return "pretrained_resnet50"
    if model_name == "pad_ufes20_resnet":
        return "resnet"

    return model_name


def describe_predictions_file(file_name):
    stem = file_name.removesuffix("_predictions.csv")

    info = {
        "estimate": stem,
        "model": clean_model_name(stem),
        "training_dataset": "unknown",
        "training_image_source": "unknown",
        "evaluation_dataset": "unknown",
        "evaluation_image_source": "unknown",
        "data": "unknown",
    }

    if stem == "pad_ufes20_resnet_test":
        info.update({
            "model": "resnet",
            "training_dataset": "pad-ufes-20",
            "training_image_source": "full-image",
            "evaluation_dataset": "pad-ufes-20",
            "evaluation_image_source": "full-image",
            "data": "pad-ufes-20-test-full-image",
        })
        return info

    if stem.startswith("pad_ufes_20_full_image_"):
        model_name = stem.removeprefix("pad_ufes_20_full_image_")
        info.update({
            "model": clean_model_name(model_name),
            "training_dataset": "ham10000",
            "training_image_source": "full-image",
            "evaluation_dataset": "pad-ufes-20",
            "evaluation_image_source": "full-image",
            "data": "pad-ufes-20-full-image",
        })
        return info

    if stem.startswith("pad_ufes_20_lesion_white_"):
        model_name = stem.removeprefix("pad_ufes_20_lesion_white_")
        info.update({
            "model": clean_model_name(model_name),
            "training_dataset": "ham10000",
            "training_image_source": "lesion-white",
            "evaluation_dataset": "pad-ufes-20",
            "evaluation_image_source": "lesion-white",
            "data": "pad-ufes-20-lesion-white",
        })
        return info

    if stem.endswith("_lesion_white_test"):
        model_name = stem.removesuffix("_lesion_white_test")
        info.update({
            "model": clean_model_name(model_name),
            "training_dataset": "ham10000",
            "training_image_source": "lesion-white",
            "evaluation_dataset": "ham10000",
            "evaluation_image_source": "lesion-white",
            "data": "ham10000-test-lesion-white",
        })
        return info

    if stem.endswith("_test"):
        model_name = stem.removesuffix("_test")
        info.update({
            "model": clean_model_name(model_name),
            "training_dataset": "ham10000",
            "training_image_source": "full-image",
            "evaluation_dataset": "ham10000",
            "evaluation_image_source": "full-image",
            "data": "ham10000-test-full-image",
        })

    return info


def get_binary_labels(df, class_column, label_column):
    if class_column in df.columns:
        class_values = df[class_column].dropna().astype(str).str.lower().str.strip()
        unique_classes = set(class_values.unique())
        if unique_classes and unique_classes <= {"benign", "malignant"}:
            return (
                df[class_column].astype(str).str.lower().str.strip() == "malignant"
            ).astype(int).to_numpy()

    if label_column in df.columns:
        label_values = pd.to_numeric(df[label_column], errors="coerce")
        unique_labels = set(label_values.dropna().astype(int).unique())
        if unique_labels and unique_labels <= {0, 1}:
            return label_values.fillna(-1).astype(int).to_numpy()

    raise ValueError(
        f"Could not read binary labels from {class_column} or {label_column}. "
        "Expected benign/malignant classes or 0/1 labels."
    )


def sensitivity_specificity(y_true, y_pred):
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else np.nan
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else np.nan
    )

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "malignant_support": true_positive + false_negative,
        "benign_support": true_negative + false_positive,
        "num_examples": len(y_true),
    }


def bootstrap_sensitivity_specificity(y_true, y_pred, rng):
    n = len(y_true)
    sensitivity_values = []
    specificity_values = []

    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        metrics = sensitivity_specificity(y_true[sample_indices], y_pred[sample_indices])
        sensitivity_values.append(metrics["sensitivity"])
        specificity_values.append(metrics["specificity"])

    sensitivity_values = np.array(sensitivity_values, dtype=float)
    specificity_values = np.array(specificity_values, dtype=float)

    return {
        "sensitivity_bootstrap_mean": float(np.nanmean(sensitivity_values)),
        "sensitivity_bootstrap_se": float(np.nanstd(sensitivity_values, ddof=1)),
        "sensitivity_ci_lower": float(np.nanpercentile(sensitivity_values, 2.5)),
        "sensitivity_ci_upper": float(np.nanpercentile(sensitivity_values, 97.5)),
        "specificity_bootstrap_mean": float(np.nanmean(specificity_values)),
        "specificity_bootstrap_se": float(np.nanstd(specificity_values, ddof=1)),
        "specificity_ci_lower": float(np.nanpercentile(specificity_values, 2.5)),
        "specificity_ci_upper": float(np.nanpercentile(specificity_values, 97.5)),
    }


def threshold_from_predictions(df):
    if "threshold" not in df.columns:
        return np.nan

    thresholds = pd.to_numeric(df["threshold"], errors="coerce").dropna().unique()
    if len(thresholds) == 0:
        return np.nan

    return float(thresholds[0])


def main():
    if not os.path.exists(PRED_DIR):
        raise FileNotFoundError(f"Missing predictions folder: {PRED_DIR}")

    prediction_files = sorted([
        file_name
        for file_name in os.listdir(PRED_DIR)
        if file_name.endswith("_predictions.csv")
    ])

    if len(prediction_files) == 0:
        raise FileNotFoundError(f"No prediction CSV files found in {PRED_DIR}")

    rng = np.random.default_rng(seed)
    rows = []

    for file_name in prediction_files:
        predictions_path = os.path.join(PRED_DIR, file_name)
        df = pd.read_csv(predictions_path)
        y_true = get_binary_labels(df, "true_class", "true_label")
        y_pred = get_binary_labels(df, "pred_class", "pred_label")

        valid = np.isin(y_true, [0, 1]) & np.isin(y_pred, [0, 1])
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            print(f"Skipping empty predictions file after filtering: {predictions_path}")
            continue

        point_metrics = sensitivity_specificity(y_true, y_pred)
        bootstrap_metrics = bootstrap_sensitivity_specificity(y_true, y_pred, rng)
        description = describe_predictions_file(file_name)

        row = {
            **description,
            "threshold": threshold_from_predictions(df),
            "sensitivity_point": point_metrics["sensitivity"],
            "specificity_point": point_metrics["specificity"],
            **bootstrap_metrics,
            "true_positive": point_metrics["true_positive"],
            "false_positive": point_metrics["false_positive"],
            "false_negative": point_metrics["false_negative"],
            "true_negative": point_metrics["true_negative"],
            "malignant_support": point_metrics["malignant_support"],
            "benign_support": point_metrics["benign_support"],
            "num_examples": point_metrics["num_examples"],
            "n_bootstrap": n_bootstrap,
            "prediction_file": predictions_path,
        }
        rows.append(row)

        print(
            f"{row['estimate']} | "
            f"sens mean={row['sensitivity_bootstrap_mean']:.4f} "
            f"se={row['sensitivity_bootstrap_se']:.4f} | "
            f"spec mean={row['specificity_bootstrap_mean']:.4f} "
            f"se={row['specificity_bootstrap_se']:.4f}"
        )

    if len(rows) == 0:
        raise ValueError("No usable prediction files found for bootstrapping.")

    os.makedirs(METRICS_DIR, exist_ok=True)
    output_path = os.path.join(METRICS_DIR, "classification_bootstrap_metrics.csv")
    output_df = pd.DataFrame(rows).sort_values(["evaluation_dataset", "data", "model", "estimate"])
    output_df.to_csv(output_path, index=False)

    print(f"Saved bootstrap metrics to {output_path}")


if __name__ == "__main__":
    main()
