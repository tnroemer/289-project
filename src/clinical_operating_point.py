import torch


TARGET_SENSITIVITY = 0.90


def apply_temperature(logits, temperature):
    return torch.softmax(logits / temperature, dim=1)


def malignant_scores_from_probs(probs, labels, malignant_labels):
    malignant_indices = [i for i, label in enumerate(labels) if label in malignant_labels]
    return probs[:, malignant_indices].sum(dim=1)


def binary_metrics_from_scores(targets, scores, labels, malignant_labels, threshold):
    malignant_indices = {i for i, label in enumerate(labels) if label in malignant_labels}
    true_malignant = [target in malignant_indices for target in targets]
    pred_malignant = [float(score) >= threshold for score in scores]
    total = len(targets)

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
    benign_precision = true_negative / (true_negative + false_negative) if true_negative + false_negative > 0 else 0.0
    benign_recall = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    benign_specificity = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if benign_precision + benign_recall > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "target_sensitivity": TARGET_SENSITIVITY,
        "binary_accuracy": (true_positive + true_negative) / total if total > 0 else 0.0,
        "binary_macro_precision": (malignant_precision + benign_precision) / 2,
        "binary_macro_recall": (malignant_recall + benign_recall) / 2,
        "binary_macro_f1": (malignant_f1 + benign_f1) / 2,
        "binary_balanced_accuracy": (malignant_recall + benign_recall) / 2,
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
        if metrics["malignant_recall"] >= target_sensitivity:
            if best_metrics is None or metrics["malignant_specificity"] > best_metrics["malignant_specificity"]:
                best_metrics = metrics

    if best_metrics is None:
        best_metrics = binary_metrics_from_scores(targets, scores, labels, malignant_labels, 0.0)
        best_metrics["target_sensitivity"] = target_sensitivity

    return best_metrics["threshold"], best_metrics


def fit_temperature(logits, targets, device):
    logits = logits.detach().to(device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = torch.clamp(torch.exp(log_temperature), 0.05, 10.0)
        loss = criterion(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = torch.clamp(torch.exp(log_temperature), 0.05, 10.0).item()
    return float(temperature)


def operating_metrics_to_rows(metrics, temperature=None):
    rows = []

    for name, value in metrics.items():
        rows.append({
            "metric": f"operating_{name}",
            "class": "operating_point",
            "value": value,
        })

    if temperature is not None:
        rows.append({
            "metric": "operating_temperature",
            "class": "operating_point",
            "value": temperature,
        })

    return rows
