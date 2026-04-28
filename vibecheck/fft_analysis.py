"""
fft_analysis.py — FFT-based frequency analysis of accelerometer signals.

Computes single-sided amplitude spectra, detects dominant frequencies,
and identifies the top N spectral peaks.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, List, Tuple


def compute_fft(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided FFT amplitude spectrum of a real signal.

    Uses a Hann window to reduce spectral leakage.

    Parameters
    ----------
    signal : 1-D array of acceleration values.
    fs     : sample rate in Hz.

    Returns
    -------
    freqs      : frequency axis in Hz (length n//2 + 1).
    magnitudes : amplitude spectrum in the same units as signal.
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])

    window = np.hanning(n)
    # Correction factor so windowed amplitude matches original signal amplitude
    window_correction = n / np.sum(window)

    windowed = signal * window
    fft_vals = np.fft.rfft(windowed)
    magnitudes = (2.0 / n) * np.abs(fft_vals) * window_correction
    # DC component is not doubled
    magnitudes[0] /= 2.0

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, magnitudes


def dominant_frequency(freqs: np.ndarray, magnitudes: np.ndarray) -> float:
    """
    Return the frequency bin with the highest amplitude.

    Skips the DC component (0 Hz) by starting the search at index 1.
    """
    if len(freqs) < 2:
        return 0.0
    idx = np.argmax(magnitudes[1:]) + 1
    return float(freqs[idx])


def top_peaks(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    n: int = 5,
    min_distance_hz: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Detect the top N local maxima in the FFT spectrum.

    Parameters
    ----------
    freqs           : frequency axis in Hz.
    magnitudes      : amplitude spectrum.
    n               : maximum number of peaks to return.
    min_distance_hz : minimum separation between peaks in Hz.
                      Prevents closely spaced bins from all being returned.

    Returns
    -------
    List of (frequency_hz, magnitude) tuples, sorted by magnitude descending.
    """
    if len(freqs) < 2:
        return []

    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    min_distance_bins = max(1, int(min_distance_hz / freq_resolution))

    peak_indices, _ = find_peaks(magnitudes, distance=min_distance_bins)

    # Exclude DC bin
    peak_indices = peak_indices[peak_indices > 0]

    if len(peak_indices) == 0:
        # Fall back to the single highest bin if find_peaks returns nothing
        idx = int(np.argmax(magnitudes[1:])) + 1
        return [(float(freqs[idx]), float(magnitudes[idx]))]

    sorted_peaks = sorted(peak_indices, key=lambda i: magnitudes[i], reverse=True)
    top_n = sorted_peaks[:n]
    return [(float(freqs[i]), float(magnitudes[i])) for i in top_n]


def run_fft_analysis(
    df: pd.DataFrame,
    fs: float = 100.0,
    axis: str = "magnitude",
    n_peaks: int = 5,
) -> Dict:
    """
    Run a complete FFT analysis on a single DataFrame column.

    Parameters
    ----------
    df      : cleaned accelerometer DataFrame.
    fs      : sample rate in Hz.
    axis    : column to analyse ('x', 'y', 'z', or 'magnitude').
    n_peaks : number of top spectral peaks to detect.

    Returns
    -------
    dict with keys:
        freqs         — frequency axis (np.ndarray)
        magnitudes    — amplitude spectrum (np.ndarray)
        dominant_freq — frequency with highest amplitude (float, Hz)
        top_peaks     — list of (freq, magnitude) tuples
        fs            — sample rate used
        axis          — axis analysed
    """
    if axis not in df.columns:
        raise ValueError(
            f"Axis '{axis}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    signal = df[axis].values.astype(float)
    freqs, magnitudes = compute_fft(signal, fs)
    dom_freq = dominant_frequency(freqs, magnitudes)
    peaks = top_peaks(freqs, magnitudes, n=n_peaks)

    return {
        "freqs": freqs,
        "magnitudes": magnitudes,
        "dominant_freq": dom_freq,
        "top_peaks": peaks,
        "fs": fs,
        "axis": axis,
    }
