"""
test_fft_analysis.py — Tests for vibecheck.fft_analysis
"""

import numpy as np
import pandas as pd
import pytest

from vibecheck.fft_analysis import (
    compute_fft,
    dominant_frequency,
    top_peaks,
    run_fft_analysis,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pure_sine(freq: float, fs: float = 100.0, duration: float = 5.0) -> np.ndarray:
    """Generate a pure sine wave with unit amplitude."""
    t = np.arange(int(fs * duration)) / fs
    return np.sin(2 * np.pi * freq * t)


# ── compute_fft ────────────────────────────────────────────────────────────────

class TestComputeFft:

    def test_returns_two_arrays(self):
        signal = _pure_sine(10.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(mags, np.ndarray)

    def test_arrays_same_length(self):
        signal = _pure_sine(10.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        assert len(freqs) == len(mags)

    def test_freqs_start_at_zero(self):
        signal = _pure_sine(10.0)
        freqs, _ = compute_fft(signal, fs=100.0)
        assert freqs[0] == 0.0

    def test_max_freq_is_nyquist(self):
        fs = 100.0
        signal = _pure_sine(10.0, fs=fs)
        freqs, _ = compute_fft(signal, fs=fs)
        assert abs(freqs[-1] - fs / 2) < 1.0

    def test_peak_at_correct_frequency(self):
        """FFT peak should align with the injected sine frequency."""
        fs = 100.0
        freq = 10.0
        signal = _pure_sine(freq, fs=fs)
        freqs, mags = compute_fft(signal, fs=fs)
        peak_freq = freqs[np.argmax(mags[1:]) + 1]
        assert abs(peak_freq - freq) < 0.5

    def test_all_magnitudes_non_negative(self):
        signal = _pure_sine(15.0)
        _, mags = compute_fft(signal, fs=100.0)
        assert (mags >= 0).all()

    def test_empty_signal(self):
        freqs, mags = compute_fft(np.array([]), fs=100.0)
        assert len(freqs) == 0
        assert len(mags) == 0


# ── dominant_frequency ────────────────────────────────────────────────────────

class TestDominantFrequency:

    def test_detects_known_frequency(self):
        fs = 100.0
        freq = 15.0
        signal = _pure_sine(freq, fs=fs)
        freqs, mags = compute_fft(signal, fs=fs)
        dom = dominant_frequency(freqs, mags)
        assert abs(dom - freq) < 1.0

    def test_skips_dc_bin(self):
        """DC-heavy signal should not return 0 Hz as dominant."""
        signal = np.ones(500) + 0.01 * _pure_sine(10.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        dom = dominant_frequency(freqs, mags)
        assert dom > 0.0

    def test_returns_float(self):
        signal = _pure_sine(10.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        result = dominant_frequency(freqs, mags)
        assert isinstance(result, float)

    def test_short_array_returns_zero(self):
        result = dominant_frequency(np.array([0.0]), np.array([1.0]))
        assert result == 0.0


# ── top_peaks ─────────────────────────────────────────────────────────────────

class TestTopPeaks:

    def test_returns_list(self):
        signal = _pure_sine(10.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        peaks = top_peaks(freqs, mags, n=5)
        assert isinstance(peaks, list)

    def test_returns_tuples_of_two(self):
        signal = _pure_sine(10.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        peaks = top_peaks(freqs, mags, n=5)
        for item in peaks:
            assert len(item) == 2

    def test_does_not_exceed_n(self):
        signal = _pure_sine(10.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        for n in (1, 3, 5):
            peaks = top_peaks(freqs, mags, n=n)
            assert len(peaks) <= n

    def test_sorted_by_magnitude_descending(self):
        signal = _pure_sine(10.0) + 0.5 * _pure_sine(20.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        peaks = top_peaks(freqs, mags, n=5)
        magnitudes = [m for _, m in peaks]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_first_peak_near_dominant_frequency(self):
        fs = 100.0
        freq = 10.0
        signal = _pure_sine(freq, fs=fs)
        freqs, mags = compute_fft(signal, fs=fs)
        peaks = top_peaks(freqs, mags, n=5)
        top_freq = peaks[0][0]
        assert abs(top_freq - freq) < 1.0

    def test_frequencies_are_positive(self):
        signal = _pure_sine(10.0)
        freqs, mags = compute_fft(signal, fs=100.0)
        peaks = top_peaks(freqs, mags, n=5)
        for freq, _ in peaks:
            assert freq > 0.0

    def test_empty_spectrum(self):
        peaks = top_peaks(np.array([]), np.array([]), n=5)
        assert isinstance(peaks, list)


# ── run_fft_analysis ──────────────────────────────────────────────────────────

class TestRunFftAnalysis:

    def test_returns_dict_with_expected_keys(self, clean_df, fs):
        result = run_fft_analysis(clean_df, fs=fs)
        expected_keys = {"freqs", "magnitudes", "dominant_freq", "top_peaks", "fs", "axis"}
        assert expected_keys.issubset(result.keys())

    def test_dominant_freq_on_x_axis(self, fs):
        """x-axis has a 10 Hz sine — dominant frequency should be near 10 Hz."""
        n = int(fs * 5)
        t = np.arange(n) / fs
        df = pd.DataFrame({
            "timestamp": t,
            "x": np.sin(2 * np.pi * 10 * t),
            "y": np.zeros(n),
            "z": np.zeros(n),
            "magnitude": np.abs(np.sin(2 * np.pi * 10 * t)),
        })
        result = run_fft_analysis(df, fs=fs, axis="x")
        assert abs(result["dominant_freq"] - 10.0) < 1.0

    def test_axis_parameter_stored_in_result(self, clean_df, fs):
        result = run_fft_analysis(clean_df, fs=fs, axis="z")
        assert result["axis"] == "z"

    def test_fs_stored_in_result(self, clean_df, fs):
        result = run_fft_analysis(clean_df, fs=fs)
        assert result["fs"] == fs

    def test_top_peaks_count_respects_n_peaks(self, clean_df, fs):
        for n in (1, 3, 5):
            result = run_fft_analysis(clean_df, fs=fs, n_peaks=n)
            assert len(result["top_peaks"]) <= n

    def test_invalid_axis_raises(self, clean_df, fs):
        with pytest.raises(ValueError, match="not found"):
            run_fft_analysis(clean_df, fs=fs, axis="w")

    def test_freqs_and_magnitudes_same_length(self, clean_df, fs):
        result = run_fft_analysis(clean_df, fs=fs)
        assert len(result["freqs"]) == len(result["magnitudes"])

    def test_dominant_freq_is_float(self, clean_df, fs):
        result = run_fft_analysis(clean_df, fs=fs)
        assert isinstance(result["dominant_freq"], float)
