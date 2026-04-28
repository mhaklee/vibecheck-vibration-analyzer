"""
cleaner.py — Signal cleaning and preprocessing for accelerometer data.

Provides gravity removal, detrending, and optional low-pass filtering.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal


def remove_gravity(
    df: pd.DataFrame,
    method: str = "mean",
    fs: float = 100.0,
) -> pd.DataFrame:
    """
    Remove the static gravity component from each acceleration axis.

    Parameters
    ----------
    df : DataFrame with columns x, y, z.
    method : 'mean'     — subtract the column mean (good for stationary devices).
             'highpass' — apply a high-pass Butterworth filter to remove DC offset
                          (better for devices that change orientation during recording).
    fs : sample rate in Hz (only used when method='highpass').

    Returns a new DataFrame with magnitude recalculated.
    """
    df = df.copy()
    for axis in ("x", "y", "z"):
        if method == "mean":
            df[axis] = df[axis] - df[axis].mean()
        elif method == "highpass":
            df[axis] = _highpass_filter(df[axis].values, cutoff=0.5, fs=fs)
        else:
            raise ValueError(f"Unknown gravity removal method: '{method}'. Use 'mean' or 'highpass'.")

    df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    return df


def detrend_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linear trend from each axis using scipy's detrend.

    Useful for correcting slow sensor drift over a recording.
    Returns a new DataFrame with magnitude recalculated.
    """
    df = df.copy()
    for axis in ("x", "y", "z"):
        df[axis] = scipy_signal.detrend(df[axis].values, type="linear")
    df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    return df


def apply_lowpass_filter(
    df: pd.DataFrame,
    cutoff: float = 50.0,
    fs: float = 100.0,
    order: int = 4,
) -> pd.DataFrame:
    """
    Apply a zero-phase Butterworth low-pass filter to all axes.

    Parameters
    ----------
    cutoff : filter cutoff frequency in Hz.
    fs     : sample rate in Hz.
    order  : filter order (higher = steeper roll-off).

    Returns a new DataFrame with magnitude recalculated.
    """
    if cutoff >= fs / 2:
        raise ValueError(
            f"Cutoff ({cutoff} Hz) must be less than the Nyquist frequency ({fs / 2} Hz)."
        )
    df = df.copy()
    for axis in ("x", "y", "z"):
        df[axis] = _lowpass_filter(df[axis].values, cutoff=cutoff, fs=fs, order=order)
    df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    return df


def clean(
    df: pd.DataFrame,
    gravity: bool = True,
    gravity_method: str = "mean",
    detrend: bool = True,
    lowpass: bool = False,
    lowpass_cutoff: float = 50.0,
    fs: float = 100.0,
) -> pd.DataFrame:
    """
    Full signal cleaning pipeline.

    Steps applied in order:
      1. Gravity removal (optional)
      2. Detrending (optional)
      3. Low-pass filtering (optional)

    Parameters
    ----------
    gravity        : remove the DC/gravity component.
    gravity_method : 'mean' or 'highpass'.
    detrend        : remove linear trend.
    lowpass        : apply a low-pass filter.
    lowpass_cutoff : low-pass cutoff frequency in Hz.
    fs             : sample rate in Hz.

    Returns a cleaned copy of the DataFrame.
    """
    if gravity:
        df = remove_gravity(df, method=gravity_method, fs=fs)
    if detrend:
        df = detrend_signal(df)
    if lowpass:
        df = apply_lowpass_filter(df, cutoff=lowpass_cutoff, fs=fs)
    return df


# ── Private helpers ────────────────────────────────────────────────────────────

def _butter_filter(data: np.ndarray, cutoff: float, fs: float, btype: str, order: int) -> np.ndarray:
    """Design and apply a zero-phase Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = np.clip(cutoff / nyq, 1e-6, 1.0 - 1e-6)
    b, a = scipy_signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return scipy_signal.filtfilt(b, a, data)


def _highpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    return _butter_filter(data, cutoff, fs, btype="high", order=order)


def _lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    return _butter_filter(data, cutoff, fs, btype="low", order=order)
