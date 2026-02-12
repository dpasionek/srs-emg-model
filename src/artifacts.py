from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import nn

from models.baseline_model import BaselineEMGModel

# Class to help save and load model artifacts.

def _to_relative_path(path: str, base_dir: str) -> str:
    # Keep paths portable by storing them relative to the artifacts directory when possible.
    if not path:
        return path
    base_abs = os.path.abspath(base_dir)
    path_abs = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(path_abs, base_abs)
    except ValueError:
        return path
    if rel_path.startswith(".."):
        return path
    return rel_path


def _resolve_path(base_dir: str, path: str) -> str:
    # Expand relative paths back to absolute paths when reading saved artifacts.
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def save_artifacts(
    output_dir: str,
    model: nn.Module,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    label_mapping: Sequence[Mapping[str, Any]],
    report_paths: Mapping[str, str],
) -> Dict[str, str]:
    # Save model weights plus a manifest that records config, metrics, and report paths.
    os.makedirs(output_dir, exist_ok=True)
    weights_filename = "model.pt"
    weights_path = os.path.join(output_dir, weights_filename)
    torch.save(model.state_dict(), weights_path)

    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        "config": dict(config),
        "metrics": dict(metrics),
        "label_mapping": list(label_mapping),
        "paths": {
            "weights": weights_filename,
            "manifest": "manifest.json",
            **{key: _to_relative_path(value, output_dir) for key, value in report_paths.items()},
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    return {
        "weights": weights_path,
        "manifest": manifest_path,
        **{key: _resolve_path(output_dir, value) for key, value in manifest["paths"].items()},
    }


def load_artifacts(artifact_dir: str) -> Tuple[nn.Module, Dict[str, Any]]:
    # Load the saved model and its manifest metadata from a training run.
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    model_config = manifest.get("config", {}).get("model", {})
    model = BaselineEMGModel(
        window_size=int(model_config.get("window_size", 0)),
        feature_count=int(model_config.get("feature_count", 0)),
        num_classes=int(model_config.get("num_classes", 0)),
    )

    weights_path = _resolve_path(
        artifact_dir, manifest.get("paths", {}).get("weights", "model.pt")
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    return model, manifest
