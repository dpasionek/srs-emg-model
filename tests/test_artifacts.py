import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from artifacts import load_artifacts, save_artifacts
from models.baseline_model import BaselineEMGModel


class TestArtifacts(unittest.TestCase):
    def test_save_and_load_artifacts(self) -> None:
        torch.manual_seed(7)
        model = BaselineEMGModel(window_size=2, feature_count=3, num_classes=4)

        config = {
            "model": {"window_size": 2, "feature_count": 3, "num_classes": 4},
            "parser": {"window_size": 2, "stride": 1, "sampling_rate": 1.0, "break_marker": "BREAK"},
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01, "seed": 7},
        }
        metrics = {
            "accuracy": 0.5,
            "precision": 0.4,
            "recall": 0.3,
            "f1": 0.35,
            "confusion_matrix": [[1, 0], [1, 0]],
        }
        label_mapping = [{"index": 0, "finger": 1, "position": 1}]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            report_json = temp_path / "training_report.json"
            report_md = temp_path / "training_report.md"
            report_json.write_text("{}", encoding="utf-8")
            report_md.write_text("# Report", encoding="utf-8")

            paths = save_artifacts(
                str(temp_path),
                model,
                config=config,
                metrics=metrics,
                label_mapping=label_mapping,
                report_paths={"report_json": str(report_json), "report_markdown": str(report_md)},
            )

            self.assertTrue(Path(paths["weights"]).exists())
            self.assertTrue(Path(paths["manifest"]).exists())

            loaded_model, manifest = load_artifacts(str(temp_path))
            self.assertEqual(manifest["config"]["model"]["num_classes"], 4)

            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded_model.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
