from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

REQUIRED_COLUMNS: Sequence[str] = (
    "timestamp",
    "emg_value",
    "rms",
    "finger",
    "position",
)
FEATURE_COLUMNS: Sequence[str] = ("timestamp", "emg_value", "rms")
FINGER_RANGE = (1, 5)
POSITION_RANGE = (1, 3)


@dataclass(frozen=True)
class EMGParserConfig:
    window_size: int
    stride: int
    sampling_rate: float = 1.0
    break_marker: str = "BREAK"


@dataclass
class EMGWindowedDataset:
    windows: List[List[List[float]]]
    labels: List[Dict[str, int]]
    session_ids: List[int]
    metadata: Dict[str, Any]


def parse_emg_csv(path: str, config: EMGParserConfig) -> EMGWindowedDataset:
    # Read the EMG CSV and produce fixed-size windows plus labels per window.
    _validate_config(config)
    sessions = _split_sessions(path, config.break_marker)
    if not sessions:
        raise ValueError("No CSV data found after applying break markers.")

    header_line = _find_header_line(sessions)
    normalized_sessions = _normalize_sessions(sessions, header_line)

    windows: List[List[List[float]]] = []
    labels: List[Dict[str, int]] = []
    session_ids: List[int] = []

    for session_id, session_lines in enumerate(normalized_sessions):
        if not _has_data_rows(session_lines):
            continue
        df = _read_session(session_lines)
        _validate_dataframe(df)
        df = _coerce_and_sort(df)
        _append_windows(
            df,
            config.window_size,
            config.stride,
            session_id,
            windows,
            labels,
            session_ids,
        )

    metadata = {
        "sampling_rate": float(config.sampling_rate),
        "window_size": int(config.window_size),
        "stride": int(config.stride),
        "feature_order": list(FEATURE_COLUMNS),
    }
    return EMGWindowedDataset(
        windows=windows,
        labels=labels,
        session_ids=session_ids,
        metadata=metadata,
    )


def _validate_config(config: EMGParserConfig) -> None:
    # Guard against invalid parser settings before reading any data.
    if config.window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if config.stride <= 0:
        raise ValueError("stride must be a positive integer.")
    if config.sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if not config.break_marker:
        raise ValueError("break_marker must be a non-empty string.")


def _split_sessions(path: str, break_marker: str) -> List[List[str]]:
    # Split the raw CSV into sessions using a break marker line.
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    sessions: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if _is_break_line(line, break_marker):
            if current:
                sessions.append(current)
                current = []
            continue
        current.append(line)

    if current:
        sessions.append(current)

    return sessions


def _is_break_line(line: str, break_marker: str) -> bool:
    # Check whether a line marks a session boundary.
    stripped = line.strip()
    if not stripped:
        return False
    if stripped == break_marker:
        return True
    first_field = stripped.split(",", 1)[0].strip()
    return first_field == break_marker


def _find_header_line(sessions: Iterable[Sequence[str]]) -> str:
    # Find the first non-empty line to use as a CSV header template.
    for session in sessions:
        for line in session:
            if line.strip():
                return line
    raise ValueError("CSV header line not found.")


def _normalize_sessions(sessions: Iterable[Sequence[str]], header_line: str) -> List[List[str]]:
    # Ensure every session has a header line for consistent parsing.
    normalized: List[List[str]] = []
    header_stripped = header_line.strip()
    for session in sessions:
        lines = list(session)
        first_data_line = next((line for line in lines if line.strip()), "")
        if not first_data_line:
            continue
        if first_data_line.strip() != header_stripped:
            lines.insert(0, header_line)
        normalized.append(lines)
    return normalized


def _has_data_rows(lines: Sequence[str]) -> bool:
    # Confirm there is at least one data row beyond the header.
    if not lines:
        return False
    non_empty = [line for line in lines if line.strip()]
    return len(non_empty) > 1


def _read_session(lines: Sequence[str]) -> pd.DataFrame:
    # Load a single session's CSV text into a DataFrame.
    csv_text = "".join(lines)
    return pd.read_csv(StringIO(csv_text))


def _validate_dataframe(df: pd.DataFrame) -> None:
    # Check that the DataFrame includes all required EMG columns.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _coerce_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    # Convert values to numeric types and sort by timestamp.
    df = df.copy()

    for column in ("timestamp", "emg_value", "rms"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Non-numeric values found in column: {column}")

    for label_column in ("finger", "position"):
        series = pd.to_numeric(df[label_column], errors="coerce")
        if series.isna().any():
            raise ValueError(f"Missing or non-numeric values found in column: {label_column}")
        if (series % 1 != 0).any():
            raise ValueError(f"Non-integer values found in column: {label_column}")
        df[label_column] = series.astype(int)

    _enforce_label_range(df, "finger", FINGER_RANGE)
    _enforce_label_range(df, "position", POSITION_RANGE)

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def _enforce_label_range(df: pd.DataFrame, column: str, valid_range: Sequence[int]) -> None:
    # Validate that label values fall within the allowed range.
    lower, upper = int(valid_range[0]), int(valid_range[1])
    out_of_range = ~df[column].between(lower, upper)
    if out_of_range.any():
        raise ValueError(
            f"Values in column '{column}' must be in range {lower}-{upper} (inclusive)."
        )


def _append_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    session_id: int,
    windows: List[List[List[float]]],
    labels: List[Dict[str, int]],
    session_ids: List[int],
) -> None:
    # Slice a session into overlapping windows and attach a single label per window.
    features = df[list(FEATURE_COLUMNS)].to_numpy()
    finger = df["finger"].to_numpy()
    position = df["position"].to_numpy()

    max_start = len(df) - window_size
    if max_start < 0:
        return

    for start in range(0, max_start + 1, stride):
        end = start + window_size
        window_finger = finger[start:end]
        window_position = position[start:end]

        if len(set(window_finger)) != 1 or len(set(window_position)) != 1:
            raise ValueError(
                "Mixed labels in window; ensure finger and position are constant within a window."
            )

        windows.append(features[start:end].tolist())
        labels.append(
            {"finger": int(window_finger[0]), "position": int(window_position[0])}
        )
        session_ids.append(session_id)
