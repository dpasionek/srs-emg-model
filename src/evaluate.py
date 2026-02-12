from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Tuple, TypedDict

import torch
from torch.utils.data import DataLoader

from artifacts import load_artifacts
from data_pipeline import EMGWindowTorchDataset
from emg_parser import EMGParserConfig, parse_emg_csv

# Some more vocabulary:
# - "Confusion Matrix": A table that describes the performance of a classification 
#                       model by showing the counts of true positives, false positives, 
#                       true negatives, and false negatives for each class.

# - "Precision": The ratio of true positives to the total predicted positives.
# - "Accuracy": The ratio of correct predictions to total predictions.

# - "Recall": The ratio of true positives to the total actual positives.
# - "F1 Score": The harmonic mean of precision and recall, providing a single metric that balances both.

class ClassificationMetrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> ClassificationMetrics:
    # Compute accuracy, precision, recall, f1, and confusion matrix from predictions.
    if num_classes <= 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [],
        }

    preds = predictions.to(torch.long).cpu()
    targs = targets.to(torch.long).cpu()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for target, pred in zip(targs.tolist(), preds.tolist()):
        confusion[int(target), int(pred)] += 1

    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    accuracy = correct / total if total > 0 else 0.0

    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0

    for idx in range(num_classes):
        tp = float(confusion[idx, idx].item())
        fp = float(confusion[:, idx].sum().item() - tp)
        fn = float(confusion[idx, :].sum().item() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_total += precision
        recall_total += recall
        f1_total += f1

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_total / num_classes),
        "recall": float(recall_total / num_classes),
        "f1": float(f1_total / num_classes),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_metrics(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> ClassificationMetrics:
    # Run the model on a dataset and return aggregated classification metrics.
    model.eval()
    predictions: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    num_classes = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            num_classes = max(num_classes, int(logits.shape[1]))
            predictions.append(torch.argmax(logits, dim=1))
            targets_list.append(targets)

    if not targets_list:
        return compute_classification_metrics(torch.tensor([]), torch.tensor([]), 0)

    all_predictions = torch.cat(predictions)
    all_targets = torch.cat(targets_list)
    return compute_classification_metrics(all_predictions, all_targets, num_classes)


def evaluate_accuracy(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    # Convenience wrapper that returns only the accuracy score.
    metrics = evaluate_metrics(model, data_loader, device)
    return float(metrics["accuracy"])


def load_model_for_evaluation(artifact_dir: str) -> Tuple[torch.nn.Module, EMGParserConfig, Dict[str, Any]]:
    # Reconstruct the model and parser settings from saved artifacts.
    model, manifest = load_artifacts(artifact_dir)
    parser_data = manifest.get("config", {}).get("parser", {})
    parser_config = EMGParserConfig(
        window_size=int(parser_data.get("window_size", 0)),
        stride=int(parser_data.get("stride", 0)),
        sampling_rate=float(parser_data.get("sampling_rate", 0.0)),
        break_marker=str(parser_data.get("break_marker", "BREAK")),
    )
    return model, parser_config, manifest


def evaluate_artifact_on_csv(
    artifact_dir: str,
    csv_path: str,
    batch_size: int = 32,
    parser_overrides: Mapping[str, Any] | None = None,
) -> Tuple[ClassificationMetrics, Dict[str, Any], int, Mapping[str, Any]]:
    # Load a saved model and evaluate it on a new CSV file.
    model, parser_config, manifest = load_model_for_evaluation(artifact_dir)
    parser_config = _apply_parser_overrides(parser_config, parser_overrides)
    _validate_parser_model_compat(parser_config, manifest)
    parsed = parse_emg_csv(csv_path, parser_config)
    dataset = EMGWindowTorchDataset(parsed.windows, parsed.labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = evaluate_metrics(model, loader, torch.device("cpu"))
    return metrics, parsed.metadata, len(dataset), manifest


def _apply_parser_overrides(
    base_config: EMGParserConfig,
    overrides: Mapping[str, Any] | None,
) -> EMGParserConfig:
    # Apply CLI override values to the parser config.
    if not overrides:
        return base_config

    updated = base_config
    if "window_size" in overrides:
        updated = replace(updated, window_size=int(overrides["window_size"]))
    if "stride" in overrides:
        updated = replace(updated, stride=int(overrides["stride"]))
    if "sampling_rate" in overrides:
        updated = replace(updated, sampling_rate=float(overrides["sampling_rate"]))
    if "break_marker" in overrides:
        updated = replace(updated, break_marker=str(overrides["break_marker"]))
    return updated


def _validate_parser_model_compat(
    parser_config: EMGParserConfig,
    manifest: Mapping[str, Any],
) -> None:
    # Ensure the parser window size matches the model's training window size.
    model_window_size = int(manifest.get("config", {}).get("model", {}).get("window_size", 0))
    if model_window_size and parser_config.window_size != model_window_size:
        raise ValueError(
            "window_size must match the artifact model window_size for evaluation."
        )
