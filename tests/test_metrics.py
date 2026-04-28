"""
test_metrics.py — Tests for vibecheck.metrics
"""

import numpy as np
import pandas as pd
import pytest

from vibecheck.metrics import (
    calculate_rms,
    calculate_peak,
    calculate_crest_factor,
    get_all_metrics,
)


# ── calculate_rms ──────────────────────────────────────────────────────────────

class TestCalculateRms:

    def test_sine_wave_rms(self):
        """RMS of a unit sine wave is 1/sqrt(2)."""
        t = np.linspace(0, 1, 10000, endpoint=False)
        signal = np.sin(2 * np.pi * 10 * t)
        rms = calculate_rms(signal)
        assert abs(rms - 1.0 / np.sqrt(2)) < 0.001

    def test_dc_signal(self):
        """RMS of a constant signal equals the constant."""
        signal = np.full(100, 3.0)
        assert abs(calculate_rms(signal) - 3.0) < 1e-9

    def test_zero_signal(self):
        signal = np.zeros(100)
        assert calculate_rms(signal) == 0.0

    def test_single_value(self):
        assert abs(calculate_rms(np.array([4.0])) - 4.0) < 1e-9

    def test_empty_signal(self):
        assert calculate_rms(np.array([])) == 0.0

    def test_returns_float(self):
        result = calculate_rms(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, float)

    def test_symmetric_signal(self):
        """RMS of [1, -1, 1, -1] should be 1.0."""
        signal = np.array([1.0, -1.0, 1.0, -1.0])
        assert abs(calculate_rms(signal) - 1.0) < 1e-9


# ── calculate_peak ─────────────────────────────────────────────────────────────

class TestCalculatePeak:

    def test_positive_peak(self):
        signal = np.array([0.1, 0.5, 0.3, -0.2])
        assert calculate_peak(signal) == 0.5

    def test_negative_peak(self):
        """Peak is the maximum absolute value, so -0.8 beats 0.5."""
        signal = np.array([0.1, 0.5, -0.8])
        assert abs(calculate_peak(signal) - 0.8) < 1e-9

    def test_all_negative(self):
        signal = np.array([-1.0, -2.0, -3.0])
        assert calculate_peak(signal) == 3.0

    def test_zero_signal(self):
        assert calculate_peak(np.zeros(50)) == 0.0

    def test_empty_signal(self):
        assert calculate_peak(np.array([])) == 0.0

    def test_returns_float(self):
        result = calculate_peak(np.array([1.0, 2.0]))
        assert isinstance(result, float)


# ── calculate_crest_factor ────────────────────────────────────────────────────

class TestCalculateCrestFactor:

    def test_sine_wave_crest_factor(self):
        """Crest factor of a sine wave is sqrt(2) ≈ 1.414."""
        t = np.linspace(0, 1, 10000, endpoint=False)
        signal = np.sin(2 * np.pi * 10 * t)
        cf = calculate_crest_factor(signal)
        assert abs(cf - np.sqrt(2)) < 0.01

    def test_dc_signal_crest_factor(self):
        """Crest factor of a constant signal is 1.0."""
        signal = np.full(100, 2.5)
        assert abs(calculate_crest_factor(signal) - 1.0) < 1e-6

    def test_zero_rms_returns_zero(self):
        """Avoid division by zero — return 0.0 when RMS is 0."""
        signal = np.zeros(100)
        assert calculate_crest_factor(signal) == 0.0

    def test_impulse_has_high_crest_factor(self):
        """An impulse signal has a very high crest factor."""
        signal = np.zeros(500)
        signal[250] = 10.0
        cf = calculate_crest_factor(signal)
        assert cf > 10.0

    def test_returns_float(self):
        result = calculate_crest_factor(np.array([1.0, -1.0, 1.0]))
        assert isinstance(result, float)


# ── get_all_metrics ────────────────────────────────────────────────────────────

class TestGetAllMetrics:

    def test_returns_dict(self, clean_df):
        result = get_all_metrics(clean_df)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, clean_df):
        result = get_all_metrics(clean_df)
        expected_keys = [
            "rms_x", "rms_y", "rms_z", "rms_magnitude",
            "peak_x", "peak_y", "peak_z", "peak_magnitude",
            "crest_factor_x", "crest_factor_y", "crest_factor_z", "crest_factor_magnitude",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_all_values_are_floats(self, clean_df):
        result = get_all_metrics(clean_df)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float"

    def test_all_values_non_negative(self, clean_df):
        result = get_all_metrics(clean_df)
        for key, val in result.items():
            assert val >= 0.0, f"{key} is negative: {val}"

    def test_peak_greater_or_equal_rms(self, clean_df):
        result = get_all_metrics(clean_df)
        for axis in ("x", "y", "z", "magnitude"):
            assert result[f"peak_{axis}"] >= result[f"rms_{axis}"], \
                f"peak_{axis} < rms_{axis}"

    def test_crest_factor_equals_peak_over_rms(self, clean_df):
        result = get_all_metrics(clean_df)
        for axis in ("x", "y", "z", "magnitude"):
            rms = result[f"rms_{axis}"]
            peak = result[f"peak_{axis}"]
            cf = result[f"crest_factor_{axis}"]
            if rms > 0:
                assert abs(cf - peak / rms) < 1e-6

    def test_missing_column_skipped_gracefully(self, clean_df):
        df_no_mag = clean_df.drop(columns=["magnitude"])
        result = get_all_metrics(df_no_mag)
        assert "rms_magnitude" not in result
        assert "rms_x" in result
