# srs-emg-model
A small model to ingest data collected from EMG sensors

## Setup
Create a virtual environment, then install runtime and test dependencies:

```bash
python -m venv .venv
```

```bash
pip install torch pandas pytest
```

## CLI
Run the CLI directly from the repo root.

Example dataset and snapshots:
- data/examples/emg_example.csv
- data/examples/README.md
- data/examples/reports/

Train example:
```bash
python src/cli.py train data/examples/emg_example.csv \
	--window-size 10 \
	--stride 5 \
	--sampling-rate 100.0 \
	--break-marker BREAK \
	--train-ratio 0.8 \
	--val-ratio 0.2 \
	--epochs 5 \
	--batch-size 32 \
	--learning-rate 0.001 \
	--seed 42 \
	--output-dir artifacts/example-run
```

Evaluate example:
```bash
python src/cli.py evaluate artifacts/example-run data/examples/emg_example.csv \
	--window-size 10 \
	--stride 5 \
	--sampling-rate 100.0 \
	--break-marker BREAK \
	--train-ratio 0.0 \
	--val-ratio 0.0 \
	--test-ratio 1.0 \
	--batch-size 32 \
	--output-dir artifacts/example-run/eval
```

Expected output artifacts:
- Training run output dir: model.pt, manifest.json, training_report.json, training_report.md
- Evaluation output dir: evaluation_report.json, evaluation_report.md

Static snapshot reports from the example dataset live in data/examples/reports.
