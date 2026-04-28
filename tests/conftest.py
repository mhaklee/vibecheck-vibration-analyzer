"""
conftest.py — Shared pytest fixtures for VibeCheck tests.

All fixtures generate synthetic accelerometer data so tests run
without needing real CSV files.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os


# ── Constants ──────────────────────────────────────────────────────────────────
FS = 100.0          # sample rate in Hz
DURATION = 5.0      # seconds
N = int(FS * DURATION)  # 500 samples


# ── Raw signal builders ────────────────────────────────────────────────────────

def _make_raw_df(
    freq_x: float = 10.0,
    freq_y: float = 20.0,
    freq_z: float = 5.0,
    amp_x: float = 0.1,
    amp_y: float = 0.05,
    amp_z: float = 0.02,
    gravity_z: float = 1.0,
    noise: float = 0.005,
    fs: float = FS,
    duration: float = DURATION,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic raw accelerometer DataFrame.

    x — sine wave at freq_x Hz
    y — sine wave at freq_y Hz
    z — sine wave at freq_z Hz + static gravity offset
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    x = amp_x * np.sin(2 * np.pi * freq_x * t) + rng.normal(0, noise, n)
    y = amp_y * np.sin(2 * np.pi * freq_y * t) + rng.normal(0, noise, n)
    z = gravity_z + amp_z * np.sin(2 * np.pi * freq_z * t) + rng.normal(0, noise, n)
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    return pd.DataFrame({"timestamp": t, "x": x, "y": y, "z": z, "magnitude": magnitude})


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Raw accelerometer DataFrame with gravity on z-axis."""
    return _make_raw_df()


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """
    Cleaned accelerometer DataFrame — gravity removed, detrended.
    Returned directly without going through the cleaner module
    so cleaner tests are independent of this fixture.
    """
    df = _make_raw_df()
    df = df.copy()
    for axis in ("x", "y", "z"):
        df[axis] = df[axis] - df[axis].mean()
    df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    return df


@pytest.fixture
def sample_csv(raw_df: pd.DataFrame) -> str:
    """
    Write a raw DataFrame to a temporary CSV file.
    Yields the file path; deletes the file after the test.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    )
    raw_df.to_csv(tmp.name, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_csv_no_timestamp(raw_df: pd.DataFrame) -> str:
    """CSV without a timestamp column — loader should synthesise one."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    )
    raw_df.drop(columns=["timestamp"]).to_csv(tmp.name, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_csv_alt_columns(raw_df: pd.DataFrame) -> str:
    """CSV using alternative column names (ax, ay, az, time)."""
    df = raw_df.rename(columns={
        "timestamp": "time",
        "x": "ax",
        "y": "ay",
        "z": "az",
    }).drop(columns=["magnitude"])
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def fs() -> float:
    """Sample rate used across all fixtures."""
    return FS
