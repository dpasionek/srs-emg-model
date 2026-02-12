# Example EMG Dataset

This directory contains a fully synthetic EMG sample for end-to-end CLI demos.

## Contents
- emg_example.csv: Two sessions separated by a BREAK line in the timestamp column.
- Each session keeps a single label so windowing produces consistent labels per window.
- Total rows: 100 (50 per session), which keeps CLI runs fast.

## Label mapping (combined labels)
The combined label index follows the project mapping:
- index = (finger - 1) * 3 + (position - 1)
- finger 1, position 1 -> index 0
- finger 2, position 2 -> index 4

## Windowing example
Using window_size 10 and stride 5:
- Session 1 (finger 1, position 1) yields 9 windows.
- Session 2 (finger 2, position 2) yields 9 windows.
- Total windows: 18.

## Report snapshots
Static snapshot reports are stored in data/examples/reports using:
- window_size 10
- stride 5
- sampling_rate 100.0
- seed 42

Files:
- data/examples/reports/training_report.json
- data/examples/reports/training_report.md
- data/examples/reports/evaluation_report.json
- data/examples/reports/evaluation_report.md
