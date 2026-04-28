"""
loader.py — Load and validate accelerometer CSV files.

Supports common column name variants from phone sensor apps
(Physics Toolbox, phyphox, Android sensor logger, etc.)
"""

import os
import numpy as np
import pandas as pd


# Maps common column name variants to canonical names
_TIMESTAMP_ALIASES = {"timestamp", "time", "t", "seconds", "elapsed", "time (s)", "time(s)"}
_X_ALIASES = {"x", "ax", "accel_x", "acceleration_x", "x_axis", "x (m/s^2)", "x(m/s^2)", "ax (g)"}
_Y_ALIASES = {"y", "ay", "accel_y", "acceleration_y", "y_axis", "y (m/s^2)", "y(m/s^2)", "ay (g)"}
_Z_ALIASES = {"z", "az", "accel_z", "acceleration_z", "z_axis", "z (m/s^2)", "z(m/s^2)", "az (g)"}


def _map_columns(columns: list) -> dict:
    """Return a rename mapping from raw column names to canonical names."""
    mapping = {}
    for col in columns:
        lower = col.strip().lower()
        if lower in _TIMESTAMP_ALIASES:
            mapping[col] = "timestamp"
        elif lower in _X_ALIASES:
            mapping[col] = "x"
        elif lower in _Y_ALIASES:
            mapping[col] = "y"
        elif lower in _Z_ALIASES:
            mapping[col] = "z"
    return mapping


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load accelerometer data from a CSV file.

    Expected columns (case-insensitive, common variants accepted):
        timestamp (or time, t, seconds, elapsed)
        x  (or ax, accel_x, acceleration_x, x_axis)
        y  (or ay, accel_y, acceleration_y, y_axis)
        z  (or az, accel_z, acceleration_z, z_axis)

    Returns a DataFrame with columns: timestamp, x, y, z, magnitude
    All acceleration values are kept in their original units (g or m/s²).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required acceleration columns cannot be found.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(f"File is empty: {filepath}")

    col_map = _map_columns(list(df.columns))
    df = df.rename(columns=col_map)

    required = {"x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Could not find required columns {missing} in '{filepath}'. "
            f"Columns found after mapping: {list(df.columns)}"
        )

    # Add a synthetic timestamp if none exists
    if "timestamp" not in df.columns:
        df.insert(0, "timestamp", np.arange(len(df), dtype=float))

    # Coerce everything to numeric and drop unreadable rows
    for col in ("timestamp", "x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    original_len = len(df)
    df = df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    dropped = original_len - len(df)
    if dropped > 0:
        print(f"  Warning: dropped {dropped} row(s) with non-numeric values.")

    df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

    return df[["timestamp", "x", "y", "z", "magnitude"]]


def infer_sample_rate(df: pd.DataFrame) -> float:
    """
    Infer the sample rate in Hz from the timestamp column.

    Uses the median inter-sample interval to be robust against gaps.
    Falls back to 100 Hz if timestamps are ambiguous or unavailable.
    """
    if len(df) < 2:
        return 100.0

    diffs = df["timestamp"].diff().dropna()
    median_diff = diffs.median()

    if median_diff <= 0:
        return 100.0

    return round(1.0 / median_diff, 4)
