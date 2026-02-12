import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from emg_parser import EMGParserConfig, parse_emg_csv


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestEMGParser(unittest.TestCase):
    def test_parse_valid_csv_orders_and_windows(self) -> None:
        config = EMGParserConfig(window_size=2, stride=1)
        dataset = parse_emg_csv(str(FIXTURES / "emg_valid.csv"), config)

        self.assertEqual(len(dataset.windows), 2)
        self.assertEqual(len(dataset.labels), 2)
        self.assertEqual(dataset.session_ids, [0, 0])
        self.assertEqual(
            dataset.windows,
            [
                [[1.0, 0.1, 0.01], [2.0, 0.2, 0.02]],
                [[2.0, 0.2, 0.02], [3.0, 0.3, 0.03]],
            ],
        )
        self.assertEqual(
            dataset.labels,
            [{"finger": 1, "position": 1}, {"finger": 1, "position": 1}],
        )

    def test_missing_required_columns_raises(self) -> None:
        config = EMGParserConfig(window_size=2, stride=1)
        with self.assertRaisesRegex(ValueError, "Missing required columns: rms"):
            parse_emg_csv(str(FIXTURES / "emg_missing_columns.csv"), config)

    def test_non_numeric_values_raise(self) -> None:
        config = EMGParserConfig(window_size=2, stride=1)
        with self.assertRaisesRegex(ValueError, "Non-numeric values found in column: emg_value"):
            parse_emg_csv(str(FIXTURES / "emg_bad_types.csv"), config)

    def test_label_range_errors_raise(self) -> None:
        config = EMGParserConfig(window_size=2, stride=1)
        csv_text = (
            "timestamp,emg_value,rms,finger,position\n"
            "1,0.1,0.01,6,1\n"
            "2,0.2,0.02,6,1\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emg.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Values in column 'finger' must be in range 1-5"
            ):
                parse_emg_csv(str(path), config)

    def test_mixed_labels_in_window_raise(self) -> None:
        config = EMGParserConfig(window_size=2, stride=1)
        csv_text = (
            "timestamp,emg_value,rms,finger,position\n"
            "1,0.1,0.01,1,1\n"
            "2,0.2,0.02,2,1\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emg.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Mixed labels in window"):
                parse_emg_csv(str(path), config)

    def test_mixed_position_in_window_raises(self) -> None:
        config = EMGParserConfig(window_size=2, stride=1)
        csv_text = (
            "timestamp,emg_value,rms,finger,position\n"
            "1,0.1,0.01,1,1\n"
            "2,0.2,0.02,1,2\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emg.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Mixed labels in window"):
                parse_emg_csv(str(path), config)

    def test_session_break_marker_handling(self) -> None:
        config = EMGParserConfig(window_size=2, stride=1)
        csv_text = (
            "timestamp,emg_value,rms,finger,position\n"
            "1,0.1,0.01,1,1\n"
            "2,0.2,0.02,1,1\n"
            "3,0.3,0.03,1,1\n"
            "BREAK\n"
            "4,0.4,0.04,2,2\n"
            "5,0.5,0.05,2,2\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emg.csv"
            path.write_text(csv_text, encoding="utf-8")
            dataset = parse_emg_csv(str(path), config)

            self.assertEqual(len(dataset.windows), 3)
            self.assertEqual(dataset.session_ids, [0, 0, 1])
            self.assertEqual(
                dataset.labels,
                [
                    {"finger": 1, "position": 1},
                    {"finger": 1, "position": 1},
                    {"finger": 2, "position": 2},
                ],
            )


if __name__ == "__main__":
    unittest.main()
