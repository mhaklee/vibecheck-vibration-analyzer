"""
test_cleaner.py — Tests for vibecheck.cleaner
"""

import numpy as np
import pandas as pd
import pytest

from vibecheck.cleaner import (
    remove_gravity,
    detrend_signal,
    apply_lowpass_filter,
    clean,
)


# ── remove_gravity ─────────────────────────────────────────────────────────────

class TestRemoveGravity:

    def test_mean_method_centres_signal(self, raw_df):
        df = remove_gravity(raw_df, method="mean")
        for axis in ("x", "y", "z"):
            assert abs(df[axis].mean()) < 1e-9, f"{axis} mean not zeroed"

    def test_highpass_method_runs(self, raw_df, fs):
        df = remove_gravity(raw_df, method="highpass", fs=fs)
        assert len(df) == len(raw_df)

    def test_highpass_reduces_dc(self, raw_df, fs):
        df = remove_gravity(raw_df, method="highpass", fs=fs)
        # z had ~1g gravity; after high-pass the mean should be much smaller
        assert abs(df["z"].mean()) < 0.1

    def test_does_not_modify_original(self, raw_df):
        original_z_mean = raw_df["z"].mean()
        remove_gravity(raw_df, method="mean")
        assert abs(raw_df["z"].mean() - original_z_mean) < 1e-9

    def test_magnitude_recalculated(self, raw_df):
        df = remove_gravity(raw_df, method="mean")
        expected = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        np.testing.assert_allclose(df["magnitude"].values, expected.values, rtol=1e-6)

    def test_invalid_method_raises(self, raw_df):
        with pytest.raises(ValueError, match="Unknown gravity removal method"):
            remove_gravity(raw_df, method="invalid")


# ── detrend_signal ─────────────────────────────────────────────────────────────

class TestDetrendSignal:

    def test_removes_linear_trend(self):
        n = 500
        t = np.arange(n) / 100.0
        # Strong linear trend on x
        df = pd.DataFrame({
            "timestamp": t,
            "x": 5.0 * t,        # pure ramp
            "y": np.zeros(n),
            "z": np.zeros(n),
            "magnitude": np.zeros(n),
        })
        df_out = detrend_signal(df)
        assert abs(df_out["x"].mean()) < 0.01
        # Slope should be gone: std of detrended ramp is near 0
        assert df_out["x"].std() < 0.1

    def test_does_not_modify_original(self, raw_df):
        x_orig = raw_df["x"].copy()
        detrend_signal(raw_df)
        pd.testing.assert_series_equal(raw_df["x"], x_orig)

    def test_preserves_length(self, raw_df):
        df = detrend_signal(raw_df)
        assert len(df) == len(raw_df)

    def test_magnitude_recalculated(self, raw_df):
        df = detrend_signal(raw_df)
        expected = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        np.testing.assert_allclose(df["magnitude"].values, expected.values, rtol=1e-6)


# ── apply_lowpass_filter ───────────────────────────────────────────────────────

class TestApplyLowpassFilter:

    def test_attenuates_high_frequency(self, fs):
        """A 40 Hz signal should be attenuated by a 10 Hz low-pass filter."""
        n = 500
        t = np.arange(n) / fs
        df = pd.DataFrame({
            "timestamp": t,
            "x": np.sin(2 * np.pi * 40 * t),
            "y": np.zeros(n),
            "z": np.zeros(n),
            "magnitude": np.zeros(n),
        })
        df_filtered = apply_lowpass_filter(df, cutoff=10.0, fs=fs)
        assert df_filtered["x"].std() < df["x"].std() * 0.5

    def test_passes_low_frequency(self, fs):
        """A 5 Hz signal should pass through a 20 Hz low-pass filter largely intact."""
        n = 500
        t = np.arange(n) / fs
        signal = np.sin(2 * np.pi * 5 * t)
        df = pd.DataFrame({
            "timestamp": t,
            "x": signal,
            "y": np.zeros(n),
            "z": np.zeros(n),
            "magnitude": np.zeros(n),
        })
        df_filtered = apply_lowpass_filter(df, cutoff=20.0, fs=fs)
        np.testing.assert_allclose(
            df_filtered["x"].values, signal, atol=0.05
        )

    def test_preserves_length(self, raw_df, fs):
        df = apply_lowpass_filter(raw_df, cutoff=30.0, fs=fs)
        assert len(df) == len(raw_df)

    def test_cutoff_above_nyquist_raises(self, raw_df, fs):
        with pytest.raises(ValueError):
            apply_lowpass_filter(raw_df, cutoff=fs, fs=fs)

    def test_does_not_modify_original(self, raw_df, fs):
        x_orig = raw_df["x"].copy()
        apply_lowpass_filter(raw_df, cutoff=30.0, fs=fs)
        pd.testing.assert_series_equal(raw_df["x"], x_orig)


# ── clean (pipeline) ──────────────────────────────────────────────────────────

class TestClean:

    def test_returns_dataframe(self, raw_df, fs):
        df = clean(raw_df, fs=fs)
        assert isinstance(df, pd.DataFrame)

    def test_gravity_removed_by_default(self, raw_df, fs):
        df = clean(raw_df, gravity=True, fs=fs)
        assert abs(df["z"].mean()) < 0.01

    def test_no_gravity_flag_skips_removal(self, raw_df, fs):
        original_z_mean = raw_df["z"].mean()
        df = clean(raw_df, gravity=False, detrend=False, fs=fs)
        assert abs(df["z"].mean() - original_z_mean) < 0.01

    def test_full_pipeline_preserves_length(self, raw_df, fs):
        df = clean(raw_df, gravity=True, detrend=True, lowpass=True,
                   lowpass_cutoff=30.0, fs=fs)
        assert len(df) == len(raw_df)

    def test_output_has_magnitude_column(self, raw_df, fs):
        df = clean(raw_df, fs=fs)
        assert "magnitude" in df.columns
