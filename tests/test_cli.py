import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from artifacts import save_artifacts
from cli import build_parser, main
from data_pipeline import FEATURE_COUNT, NUM_CLASSES
from models.baseline_model import BaselineEMGModel
from reporting import build_label_mapping


class TestCLIParsing(unittest.TestCase):
    def test_train_parsing(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "train",
                "data.csv",
                "--window-size",
                "12",
                "--stride",
                "3",
                "--sampling-rate",
                "200.0",
                "--break-marker",
                "PAUSE",
                "--train-ratio",
                "0.7",
                "--val-ratio",
                "0.2",
                "--test-ratio",
                "0.1",
                "--epochs",
                "2",
                "--batch-size",
                "16",
                "--learning-rate",
                "0.01",
                "--seed",
                "9",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.command, "train")
        self.assertEqual(args.csv_path, "data.csv")
        self.assertEqual(args.window_size, 12)
        self.assertEqual(args.stride, 3)
        self.assertAlmostEqual(args.sampling_rate, 200.0)
        self.assertEqual(args.break_marker, "PAUSE")
        self.assertAlmostEqual(args.train_ratio, 0.7)
        self.assertAlmostEqual(args.val_ratio, 0.2)
        self.assertAlmostEqual(args.test_ratio, 0.1)
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.batch_size, 16)
        self.assertAlmostEqual(args.learning_rate, 0.01)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.output_dir, "out")

    def test_evaluate_parsing(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "evaluate",
                "artifacts",
                "data.csv",
                "--window-size",
                "2",
                "--stride",
                "1",
                "--sampling-rate",
                "100.0",
                "--break-marker",
                "BREAK",
                "--train-ratio",
                "0.6",
                "--val-ratio",
                "0.2",
                "--test-ratio",
                "0.2",
                "--batch-size",
                "8",
                "--output-dir",
                "reports",
            ]
        )

        self.assertEqual(args.command, "evaluate")
        self.assertEqual(args.artifact_dir, "artifacts")
        self.assertEqual(args.csv_path, "data.csv")
        self.assertEqual(args.window_size, 2)
        self.assertEqual(args.stride, 1)
        self.assertAlmostEqual(args.sampling_rate, 100.0)
        self.assertEqual(args.break_marker, "BREAK")
        self.assertAlmostEqual(args.train_ratio, 0.6)
        self.assertAlmostEqual(args.val_ratio, 0.2)
        self.assertAlmostEqual(args.test_ratio, 0.2)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.output_dir, "reports")


class TestCLIEvaluateSmoke(unittest.TestCase):
    def test_evaluate_command_writes_reports(self) -> None:
        csv_text = (
            "timestamp,emg_value,rms,finger,position\n"
            "1,0.1,0.01,1,1\n"
            "2,0.2,0.02,1,1\n"
            "3,0.3,0.03,1,1\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifact_dir = temp_path / "artifacts"
            output_dir = temp_path / "evaluation"
            artifact_dir.mkdir()
            output_dir.mkdir()

            csv_path = temp_path / "emg.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            torch.manual_seed(7)
            model = BaselineEMGModel(
                window_size=2,
                feature_count=FEATURE_COUNT,
                num_classes=NUM_CLASSES,
            )

            report_json = artifact_dir / "training_report.json"
            report_md = artifact_dir / "training_report.md"
            report_json.write_text("{}", encoding="utf-8")
            report_md.write_text("# Report", encoding="utf-8")

            config = {
                "model": {
                    "window_size": 2,
                    "feature_count": FEATURE_COUNT,
                    "num_classes": NUM_CLASSES,
                },
                "parser": {
                    "window_size": 2,
                    "stride": 1,
                    "sampling_rate": 1.0,
                    "break_marker": "BREAK",
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "seed": 7,
                },
            }
            metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": [],
            }

            save_artifacts(
                str(artifact_dir),
                model,
                config=config,
                metrics=metrics,
                label_mapping=build_label_mapping(),
                report_paths={
                    "report_json": str(report_json),
                    "report_markdown": str(report_md),
                },
            )

            main(
                [
                    "evaluate",
                    str(artifact_dir),
                    str(csv_path),
                    "--window-size",
                    "2",
                    "--stride",
                    "1",
                    "--sampling-rate",
                    "1.0",
                    "--break-marker",
                    "BREAK",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertTrue((output_dir / "evaluation_report.json").exists())
            self.assertTrue((output_dir / "evaluation_report.md").exists())


if __name__ == "__main__":
    unittest.main()
