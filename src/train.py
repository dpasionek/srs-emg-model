from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from artifacts import save_artifacts
from data_pipeline import FEATURE_COUNT, NUM_CLASSES, prepare_datasets
from emg_parser import EMGParserConfig
from evaluate import evaluate_metrics
from models.baseline_model import BaselineEMGModel
from reporting import build_label_mapping, build_report_data, write_reports


# Some vocabulary (some of which might be review if you've read the other files)
# But is nonetheless important here too.
# - "Model": A function that takes in input data and produces predictions.
# - "Training": The process of adjusting the model's parameters to improve its predictions on a dataset.
# - "Evaluation": The process of measuring how well the model's predictions match the true labels on a dataset.
# - "Features": The input data that the model uses to make predictions.
# - "Labels": The true output that we want the model to predict.
# - "Classes": The distinct categories (target labels) that the model is trying to predict.
# - "Tensor": A multi-dimensional array used to store data in PyTorch.
# - "Epoch": One complete pass through the entire training dataset during training.
# - "Loss Function": A mathematical function that quantifies the difference between the model's 
#                    predictions and the true labels.
# - "Optimizer": An algorithm that adjusts the model's parameters based on the computed loss 
#                to improve predictions.
# - "Overfitting": When a model learns the training data too well, including its noise and outliers,
#                  leading to poor performance on new, unseen data.
# - "Underfitting": When a model is too simple to capture the underlying patterns in the data,
#                   leading to poor performance even on the training data.
# - "Seed": A value used to initialize the random number generator, 
#           ensuring reproducibility of results across runs.

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42


def run_training(
    csv_path: str,
    parser_config: EMGParserConfig,
    split_ratios: Sequence[float],
    training_config: TrainingConfig,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    # End-to-end training: prepare data, train the model, evaluate, and save artifacts.
    torch.manual_seed(training_config.seed)
    device = torch.device("cpu")

    datasets = prepare_datasets(csv_path, parser_config, split_ratios, training_config.seed)
    train_loader = DataLoader(datasets.train, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(datasets.val, batch_size=training_config.batch_size, shuffle=False)

    model = BaselineEMGModel(
        window_size=int(parser_config.window_size),
        feature_count=FEATURE_COUNT,
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    for _ in range(training_config.epochs):
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    val_metrics = evaluate_metrics(model, val_loader, device)
    val_accuracy = float(val_metrics.get("accuracy", 0.0))
    print(f"validation_accuracy: {val_accuracy:.4f}")

    split_sizes = {
        "train": len(datasets.train),
        "val": len(datasets.val),
        "test": len(datasets.test) if datasets.test is not None else 0,
    }
    report_data = build_report_data(
        metrics=val_metrics,
        split_sizes=split_sizes,
        metadata=datasets.metadata,
        seed=training_config.seed,
        label_mapping=build_label_mapping(),
    )
    report_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(csv_path)), "reports")
    report_paths = write_reports(report_data, report_dir, "training_report")

    config = {
        "model": {
            "window_size": int(parser_config.window_size),
            "feature_count": int(FEATURE_COUNT),
            "num_classes": int(NUM_CLASSES),
        },
        "parser": {
            "window_size": int(parser_config.window_size),
            "stride": int(parser_config.stride),
            "sampling_rate": float(parser_config.sampling_rate),
            "break_marker": str(parser_config.break_marker),
        },
        "training": {
            "epochs": int(training_config.epochs),
            "batch_size": int(training_config.batch_size),
            "learning_rate": float(training_config.learning_rate),
            "seed": int(training_config.seed),
        },
    }
    save_artifacts(
        report_dir,
        model,
        config=config,
        metrics=val_metrics,
        label_mapping=build_label_mapping(),
        report_paths={"report_json": report_paths["json"], "report_markdown": report_paths["markdown"]},
    )

    return {"val_accuracy": float(val_accuracy), **val_metrics}
