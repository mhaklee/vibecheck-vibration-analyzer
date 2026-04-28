"""
metrics.py — Core vibration metrics calculations.

Provides RMS, peak acceleration, and crest factor for each axis and magnitude.
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_rms(signal: np.ndarray) -> float:
    """
    Root Mean Square (RMS) of a signal.

    RMS is the most common single-number summary of vibration intensity.
    It reflects the energy content of the signal averaged over time.
    """
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def calculate_peak(signal: np.ndarray) -> float:
    """
    Peak acceleration — the maximum absolute value in the signal.

    Useful for identifying worst-case shock loads.
    """
    if len(signal) == 0:
        return 0.0
    return float(np.max(np.abs(signal)))


def calculate_crest_factor(signal: np.ndarray) -> float:
    """
    Crest factor — ratio of peak to RMS.

    A low crest factor (~1.4) indicates smooth, sinusoidal vibration.
    A high crest factor (>4) suggests impulsive events such as bearing
    faults, mechanical looseness, or impacts.

    Returns 0.0 if RMS is zero (signal is flat).
    """
    rms = calculate_rms(signal)
    if rms == 0.0:
        return 0.0
    return calculate_peak(signal) / rms


def get_all_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate RMS, peak, and crest factor for each axis (x, y, z) and magnitude.

    Parameters
    ----------
    df : cleaned DataFrame containing at minimum columns x, y, z, magnitude.

    Returns
    -------
    dict with keys like 'rms_x', 'peak_magnitude', 'crest_factor_z', etc.
    """
    results: Dict[str, float] = {}
    for axis in ("x", "y", "z", "magnitude"):
        if axis not in df.columns:
            continue
        sig = df[axis].values
        results[f"rms_{axis}"] = calculate_rms(sig)
        results[f"peak_{axis}"] = calculate_peak(sig)
        results[f"crest_factor_{axis}"] = calculate_crest_factor(sig)
    return results
