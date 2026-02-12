import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from evaluate import compute_classification_metrics
from reporting import build_report_data, format_markdown_report, write_reports


class TestReporting(unittest.TestCase):
    def test_compute_classification_metrics(self) -> None:
        targets = torch.tensor([0, 1, 2, 1])
        predictions = torch.tensor([0, 2, 2, 1])

        metrics = compute_classification_metrics(predictions, targets, num_classes=3)

        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=6)
        self.assertAlmostEqual(metrics["precision"], 0.833333, places=5)
        self.assertAlmostEqual(metrics["recall"], 0.833333, places=5)
        self.assertAlmostEqual(metrics["f1"], 0.777777, places=5)

    def test_report_formatting_and_writing(self) -> None:
        metrics = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "confusion_matrix": [[2, 0], [0, 3]],
        }
        report_data = build_report_data(
            metrics=metrics,
            split_sizes={"train": 5, "val": 2, "test": 0},
            metadata={"window_size": 2, "stride": 1, "sampling_rate": 100.0},
            seed=7,
            label_mapping=[
                {"index": 0, "finger": 1, "position": 1},
                {"index": 1, "finger": 1, "position": 2},
            ],
        )
        markdown = format_markdown_report(report_data)

        self.assertIn("# EMG Evaluation Report", markdown)
        self.assertIn("## Metrics", markdown)
        self.assertIn("accuracy", markdown)
        self.assertIn("## Confusion Matrix", markdown)
        self.assertIn("| index | finger | position |", markdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_reports(report_data, temp_dir, "unit_test_report")
            json_path = Path(paths["json"])
            md_path = Path(paths["markdown"])

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["splits"]["train"], 5)


if __name__ == "__main__":
    unittest.main()
