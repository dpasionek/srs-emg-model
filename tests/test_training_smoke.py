import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from emg_parser import EMGParserConfig
from train import TrainingConfig, run_training


class TestTrainingSmoke(unittest.TestCase):
    def test_training_runs_and_reports_accuracy(self) -> None:
        csv_text = (
            "timestamp,emg_value,rms,finger,position\n"
            "1,0.1,0.01,1,1\n"
            "2,0.2,0.02,1,1\n"
            "3,0.3,0.03,1,1\n"
            "BREAK,,,,\n"
            "4,0.4,0.04,2,2\n"
            "5,0.5,0.05,2,2\n"
            "6,0.6,0.06,2,2\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emg.csv"
            path.write_text(csv_text, encoding="utf-8")

            parser_config = EMGParserConfig(window_size=2, stride=1)
            training_config = TrainingConfig(epochs=1, batch_size=2, learning_rate=1e-2, seed=123)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                metrics = run_training(
                    str(path),
                    parser_config,
                    split_ratios=(0.5, 0.5),
                    training_config=training_config,
                )

            output = buffer.getvalue()
            self.assertIn("validation_accuracy", output)
            self.assertGreaterEqual(metrics["val_accuracy"], 0.0)
            self.assertLessEqual(metrics["val_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
