from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

# Some helpful vocabulary when talking about machine learning:
# - "Model": A function that takes in input data and produces predictions. 
#            In our case, it will take in EMG features and output predicted labels.
# - "Training": The process of adjusting the model's parameters to improve its predictions on a dataset.
# - "Evaluation": The process of measuring how well the model's predictions match the true labels on a dataset.
# - "Features": The input data that the model uses to make predictions. 
#               For EMG, this could be the raw sensor readings or some processed version of them.
# - "Labels": The true output that we want the model to predict.
#             For our EMG classification task, this would be the combination of finger and position.
# - "Classes": The distinct categories (target labels) that the model is trying to predict.
#              In our case, each unique combination of finger and position is a class.
# - "Tensor": A multi-dimensional array used to store data in PyTorch.
# - "Epoch": One complete pass through the entire training dataset during training.


from emg_parser import (
    EMGParserConfig,
    FEATURE_COLUMNS,
    FINGER_RANGE,
    POSITION_RANGE,
    parse_emg_csv,
)

# Our classes, the target predictions, are ALL combinations of fingers and positions. 
# So we calculate the total number of classes as: (number of fingers) * (number of positions).

# Our feature count is 2, despite having 5 columns in the raw data,
# because we are only using "emg_value" and "rms" as features for the model.

LABEL_FINGER_COUNT = int(FINGER_RANGE[1] - FINGER_RANGE[0] + 1)
LABEL_POSITION_COUNT = int(POSITION_RANGE[1] - POSITION_RANGE[0] + 1)
NUM_CLASSES = LABEL_FINGER_COUNT * LABEL_POSITION_COUNT
FEATURE_COUNT = len(FEATURE_COLUMNS)


@dataclass(frozen=True)
class DatasetSplits:
    train: "EMGWindowTorchDataset"
    val: "EMGWindowTorchDataset"
    test: Optional["EMGWindowTorchDataset"]
    metadata: Dict[str, object]


def encode_label(label: Dict[str, int]) -> int:
    # Map a (finger, position) label pair into a single class index.
    finger_index = int(label["finger"]) - int(FINGER_RANGE[0])
    position_index = int(label["position"]) - int(POSITION_RANGE[0])
    return finger_index * LABEL_POSITION_COUNT + position_index


class EMGWindowTorchDataset(Dataset):
    def __init__(self, windows: Sequence[Sequence[Sequence[float]]], labels: Sequence[Dict[str, int]]):
        # Store windows and labels as tensors for use by PyTorch DataLoader.
        self.features = torch.tensor(windows, dtype=torch.float32)
        self.targets = torch.tensor([encode_label(label) for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        # Return number of windows available in this dataset.
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fetch one window and its encoded label.
        return self.features[index], self.targets[index]


def prepare_datasets(
    csv_path: str,
    parser_config: EMGParserConfig,
    split_ratios: Sequence[float],
    seed: int,
) -> DatasetSplits:
    # Parse the CSV, split by session, and return train/val/test datasets.
    if len(split_ratios) not in (2, 3):
        raise ValueError("split_ratios must have length 2 (train/val) or 3 (train/val/test).")
    if any(ratio < 0 for ratio in split_ratios):
        raise ValueError("split_ratios must be non-negative.")

    parsed = parse_emg_csv(csv_path, parser_config)
    split_sessions = _split_session_ids(parsed.session_ids, split_ratios, seed)

    train_windows, train_labels = _filter_by_session(parsed.windows, parsed.labels, parsed.session_ids, split_sessions[0])
    val_windows, val_labels = _filter_by_session(parsed.windows, parsed.labels, parsed.session_ids, split_sessions[1])

    test_dataset = None
    if len(split_sessions) == 3:
        test_windows, test_labels = _filter_by_session(
            parsed.windows, parsed.labels, parsed.session_ids, split_sessions[2]
        )
        test_dataset = EMGWindowTorchDataset(test_windows, test_labels)

    return DatasetSplits(
        train=EMGWindowTorchDataset(train_windows, train_labels),
        val=EMGWindowTorchDataset(val_windows, val_labels),
        test=test_dataset,
        metadata=parsed.metadata,
    )


def _split_session_ids(
    session_ids: Iterable[int],
    split_ratios: Sequence[float],
    seed: int,
) -> List[List[int]]:
    # Randomly assign session IDs to splits while preserving split ratios.
    unique_sessions = list(sorted(set(session_ids)))
    if not unique_sessions:
        raise ValueError("No sessions available for splitting.")

    rng = random.Random(seed)
    rng.shuffle(unique_sessions)

    total_ratio = sum(split_ratios)
    if total_ratio <= 0:
        raise ValueError("split_ratios must sum to a positive value.")

    normalized = [ratio / total_ratio for ratio in split_ratios]
    counts = _allocate_counts(len(unique_sessions), normalized)

    splits: List[List[int]] = []
    cursor = 0
    for count in counts:
        splits.append(unique_sessions[cursor : cursor + count])
        cursor += count

    return splits


def _allocate_counts(total: int, normalized_ratios: Sequence[float]) -> List[int]:
    # Turn normalized ratios into integer counts, ensuring the train split is non-empty.
    counts = [int(total * ratio) for ratio in normalized_ratios[:-1]]
    remaining = total - sum(counts)
    counts.append(remaining)
    if total > 0 and counts[0] == 0:
        donor_index = 1 + max(range(len(counts) - 1), key=lambda idx: counts[idx + 1])
        if counts[donor_index] > 0:
            counts[donor_index] -= 1
            counts[0] += 1
    return counts


def _filter_by_session(
    windows: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[Dict[str, int]],
    session_ids: Sequence[int],
    allowed_sessions: Sequence[int],
) -> Tuple[List[List[List[float]]], List[Dict[str, int]]]:
    # Keep only windows that belong to the requested session IDs.
    allowed = set(allowed_sessions)
    filtered_windows: List[List[List[float]]] = []
    filtered_labels: List[Dict[str, int]] = []

    for window, label, session_id in zip(windows, labels, session_ids):
        if session_id in allowed:
            filtered_windows.append(list(list(row) for row in window))
            filtered_labels.append(dict(label))

    return filtered_windows, filtered_labels
