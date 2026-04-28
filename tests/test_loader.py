"""
test_loader.py — Tests for vibecheck.loader
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from vibecheck.loader import load_csv, infer_sample_rate


# ── load_csv ───────────────────────────────────────────────────────────────────

class TestLoadCsv:

    def test_returns_dataframe(self, sample_csv):
        df = load_csv(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, sample_csv):
        df = load_csv(sample_csv)
        assert set(df.columns) == {"timestamp", "x", "y", "z", "magnitude"}

    def test_correct_row_count(self, sample_csv, raw_df):
        df = load_csv(sample_csv)
        assert len(df) == len(raw_df)

    def test_magnitude_is_positive(self, sample_csv):
        df = load_csv(sample_csv)
        assert (df["magnitude"] >= 0).all()

    def test_magnitude_computed_correctly(self, sample_csv):
        df = load_csv(sample_csv)
        expected = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        np.testing.assert_allclose(df["magnitude"].values, expected.values, rtol=1e-6)

    def test_alt_column_names(self, sample_csv_alt_columns):
        """Loader should accept ax/ay/az/time column names."""
        df = load_csv(sample_csv_alt_columns)
        assert {"x", "y", "z", "timestamp"}.issubset(df.columns)

    def test_synthesises_timestamp_when_missing(self, sample_csv_no_timestamp):
        df = load_csv(sample_csv_no_timestamp)
        assert "timestamp" in df.columns
        assert len(df["timestamp"]) == len(df)

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv")

    def test_raises_value_error_missing_columns(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        pd.DataFrame({"timestamp": [0, 1], "x": [0.1, 0.2]}).to_csv(tmp.name, index=False)
        tmp.close()
        try:
            with pytest.raises(ValueError, match="Could not find required columns"):
                load_csv(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_drops_non_numeric_rows(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        tmp.write("timestamp,x,y,z\n0,0.1,0.2,0.3\n1,bad,0.2,0.3\n2,0.3,0.4,0.5\n")
        tmp.close()
        try:
            df = load_csv(tmp.name)
            assert len(df) == 2
        finally:
            os.unlink(tmp.name)

    def test_all_columns_numeric(self, sample_csv):
        df = load_csv(sample_csv)
        for col in ("timestamp", "x", "y", "z", "magnitude"):
            assert pd.api.types.is_float_dtype(df[col]), f"Column {col} is not float"


# ── infer_sample_rate ──────────────────────────────────────────────────────────

class TestInferSampleRate:

    def test_infers_100hz(self, raw_df):
        fs = infer_sample_rate(raw_df)
        assert abs(fs - 100.0) < 1.0

    def test_infers_50hz(self):
        t = np.arange(0, 5, 1 / 50.0)
        df = pd.DataFrame({
            "timestamp": t,
            "x": np.zeros(len(t)),
            "y": np.zeros(len(t)),
            "z": np.ones(len(t)),
            "magnitude": np.ones(len(t)),
        })
        fs = infer_sample_rate(df)
        assert abs(fs - 50.0) < 1.0

    def test_fallback_single_row(self):
        df = pd.DataFrame({"timestamp": [0.0], "x": [0.1], "y": [0.1], "z": [1.0], "magnitude": [1.0]})
        fs = infer_sample_rate(df)
        assert fs == 100.0

    def test_returns_float(self, raw_df):
        fs = infer_sample_rate(raw_df)
        assert isinstance(fs, float)
