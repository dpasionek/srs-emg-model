from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

from emg_parser import EMGParserConfig
from evaluate import evaluate_artifact_on_csv
from reporting import build_report_data, write_reports
from train import TrainingConfig, run_training

# Command-line interface construction.

def build_parser() -> argparse.ArgumentParser:
    # Define CLI commands and their arguments for training and evaluation.
    parser = argparse.ArgumentParser(description="EMG baseline training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train baseline model")
    train_parser.add_argument("csv_path", help="Path to the EMG CSV file.")
    train_parser.add_argument("--window-size", type=int, default=10, help="Window length in rows.")
    train_parser.add_argument("--stride", type=int, default=5, help="Stride between windows in rows.")
    train_parser.add_argument("--sampling-rate", type=float, default=1.0, help="Sampling rate in Hz.")
    train_parser.add_argument(
        "--break-marker",
        type=str,
        default="BREAK",
        help="Sentinel token in the TIMESTAMP column that indicates a session break.",
    )
    train_parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    train_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    train_parser.add_argument("--test-ratio", type=float, default=0.0, help="Test split ratio.")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and training.")
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports and artifacts (defaults next to CSV).",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model artifact")
    eval_parser.add_argument(
        "artifact_dir",
        help="Directory with manifest.json and model weights from training.",
    )
    eval_parser.add_argument("csv_path", help="Path to the EMG CSV file.")
    eval_parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override window size for parsing (must match model window_size).",
    )
    eval_parser.add_argument("--stride", type=int, default=None, help="Override stride for parsing.")
    eval_parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Override sampling rate for parsing.",
    )
    eval_parser.add_argument(
        "--break-marker",
        type=str,
        default=None,
        help="Override break marker for parsing.",
    )
    eval_parser.add_argument("--train-ratio", type=float, default=0.0, help="Train split ratio.")
    eval_parser.add_argument("--val-ratio", type=float, default=0.0, help="Validation split ratio.")
    eval_parser.add_argument("--test-ratio", type=float, default=1.0, help="Test split ratio.")
    eval_parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation reports (defaults to artifact dir).",
    )

    return parser


def _allocate_split_sizes(total_count: int, split_ratios: Sequence[float]) -> Dict[str, int]:
    # Convert ratio inputs into integer split counts that add up to the total.
    total_ratio = float(sum(split_ratios))
    if total_ratio <= 0:
        return {"train": 0, "val": 0, "test": int(total_count)}

    normalized = [ratio / total_ratio for ratio in split_ratios]
    counts = [int(total_count * ratio) for ratio in normalized[:-1]]
    remaining = int(total_count) - sum(counts)
    counts.append(remaining)
    return {"train": counts[0], "val": counts[1], "test": counts[2]}


def main(argv: Optional[List[str]] = None) -> None:
    # Entry point that wires CLI arguments to training or evaluation workflows.
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        parser_config = EMGParserConfig(
            window_size=args.window_size,
            stride=args.stride,
            sampling_rate=args.sampling_rate,
            break_marker=args.break_marker,
        )
        split_ratios = (args.train_ratio, args.val_ratio)
        if args.test_ratio > 0:
            split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        run_training(args.csv_path, parser_config, split_ratios, training_config, output_dir=args.output_dir)
    elif args.command == "evaluate":
        parser_overrides = {
            "window_size": args.window_size,
            "stride": args.stride,
            "sampling_rate": args.sampling_rate,
            "break_marker": args.break_marker,
        }
        parser_overrides = {key: value for key, value in parser_overrides.items() if value is not None}

        metrics, metadata, total_count, manifest = evaluate_artifact_on_csv(
            args.artifact_dir,
            args.csv_path,
            batch_size=args.batch_size,
            parser_overrides=parser_overrides or None,
        )
        label_mapping = manifest.get("label_mapping", [])
        seed = int(manifest.get("config", {}).get("training", {}).get("seed", 0))
        split_sizes = _allocate_split_sizes(
            int(total_count),
            (args.train_ratio, args.val_ratio, args.test_ratio),
        )
        report_data = build_report_data(
            metrics=metrics,
            split_sizes=split_sizes,
            metadata=metadata,
            seed=seed,
            label_mapping=label_mapping,
        )
        report_dir = args.output_dir or args.artifact_dir
        write_reports(report_data, report_dir, "evaluation_report")


if __name__ == "__main__":
    main()
