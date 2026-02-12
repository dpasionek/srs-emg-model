from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping

from data_pipeline import encode_label
from emg_parser import FINGER_RANGE, POSITION_RANGE


def build_label_mapping() -> List[Dict[str, int]]:
    # Create a stable list that maps each class index to (finger, position).
    mapping: List[Dict[str, int]] = []
    for finger in range(int(FINGER_RANGE[0]), int(FINGER_RANGE[1]) + 1):
        for position in range(int(POSITION_RANGE[0]), int(POSITION_RANGE[1]) + 1):
            label = {"finger": int(finger), "position": int(position)}
            mapping.append({"index": int(encode_label(label)), **label})
    mapping.sort(key=lambda item: item["index"])
    return mapping


def build_report_data(
    metrics: Mapping[str, Any],
    split_sizes: Dict[str, int],
    metadata: Dict[str, Any],
    seed: int,
    label_mapping: Iterable[Dict[str, int]],
) -> Dict[str, object]:
    # Bundle metrics, metadata, and label info into a single report payload.
    mapping_list = list(label_mapping)
    return {
        "metadata": {
            "window_size": int(metadata.get("window_size", 0)),
            "stride": int(metadata.get("stride", 0)),
            "sampling_rate": float(metadata.get("sampling_rate", 0.0)),
            "feature_order": list(metadata.get("feature_order", [])),
        },
        "splits": {
            "train": int(split_sizes.get("train", 0)),
            "val": int(split_sizes.get("val", 0)),
            "test": int(split_sizes.get("test", 0)),
        },
        "seed": int(seed),
        "labels": {
            "count": int(len(mapping_list)),
            "mapping": mapping_list,
        },
        "metrics": metrics,
    }


def format_markdown_report(report_data: Dict[str, Any]) -> str:
    # Convert report data into a readable Markdown summary.
    metadata = report_data.get("metadata", {})
    splits = report_data.get("splits", {})
    metrics = report_data.get("metrics", {})
    labels = report_data.get("labels", {})
    mapping = labels.get("mapping", [])

    lines = ["# EMG Evaluation Report", "", "## Metadata"]
    lines.append(f"- window_size: {metadata.get('window_size', 0)}")
    lines.append(f"- stride: {metadata.get('stride', 0)}")
    lines.append(f"- sampling_rate: {metadata.get('sampling_rate', 0.0)}")
    lines.append(f"- seed: {report_data.get('seed', 0)}")
    lines.append("")

    lines.append("## Splits")
    lines.append(f"- train: {splits.get('train', 0)}")
    lines.append(f"- val: {splits.get('val', 0)}")
    lines.append(f"- test: {splits.get('test', 0)}")
    lines.append("")

    lines.append("## Metrics")
    lines.append(f"- accuracy: {metrics.get('accuracy', 0.0):.4f}")
    lines.append(f"- precision: {metrics.get('precision', 0.0):.4f}")
    lines.append(f"- recall: {metrics.get('recall', 0.0):.4f}")
    lines.append(f"- f1: {metrics.get('f1', 0.0):.4f}")
    lines.append("")

    lines.append("## Confusion Matrix")
    confusion = metrics.get("confusion_matrix", [])
    if confusion:
        lines.append("```")
        for row in confusion:
            lines.append(" ".join(str(value) for value in row))
        lines.append("```")
    else:
        lines.append("- empty")
    lines.append("")

    lines.append("## Label Mapping")
    lines.append("| index | finger | position |")
    lines.append("| --- | --- | --- |")
    for item in mapping:
        lines.append(f"| {item.get('index', 0)} | {item.get('finger', 0)} | {item.get('position', 0)} |")

    return "\n".join(lines) + "\n"


def write_reports(report_data: Dict[str, Any], output_dir: str, report_name: str) -> Dict[str, str]:
    # Write JSON and Markdown versions of the report to disk.
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{report_name}.json")
    md_path = os.path.join(output_dir, f"{report_name}.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2)
        handle.write("\n")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(format_markdown_report(report_data))

    return {"json": json_path, "markdown": md_path}
