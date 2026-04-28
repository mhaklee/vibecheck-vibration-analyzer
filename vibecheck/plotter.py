"""
plotter.py — Matplotlib-based plots for vibration analysis results.

All functions return a matplotlib Figure and optionally save a PNG.
Close figures after use with plt.close(fig) to avoid memory leaks,
especially inside Streamlit or loop-based workflows.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional

# ── Shared style constants ────────────────────────────────────────────────────
_DPI = 120
_COLOR_X = "#1f77b4"
_COLOR_Y = "#ff7f0e"
_COLOR_Z = "#2ca02c"
_COLOR_MAG = "#d62728"
_COLOR_FFT = "#9467bd"
_LINEWIDTH = 0.85


def plot_time_domain(
    df: pd.DataFrame,
    title: str = "Time-Domain Acceleration",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot x, y, z axes and magnitude over time in a two-panel figure.

    Top panel  : x, y, z individual axes.
    Bottom panel: resultant magnitude.

    Parameters
    ----------
    df          : cleaned DataFrame with columns timestamp, x, y, z, magnitude.
    title       : figure title.
    output_path : if provided, save figure as PNG to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=_DPI, sharex=True)

    t = df["timestamp"]

    ax0 = axes[0]
    ax0.plot(t, df["x"], label="X", color=_COLOR_X, linewidth=_LINEWIDTH)
    ax0.plot(t, df["y"], label="Y", color=_COLOR_Y, linewidth=_LINEWIDTH)
    ax0.plot(t, df["z"], label="Z", color=_COLOR_Z, linewidth=_LINEWIDTH)
    ax0.set_ylabel("Acceleration (g)")
    ax0.set_title(title, fontsize=12, fontweight="bold")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(t, df["magnitude"], label="Magnitude", color=_COLOR_MAG, linewidth=_LINEWIDTH)
    ax1.set_ylabel("Magnitude (g)")
    ax1.set_xlabel("Time (s)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")

    return fig


def plot_fft_spectrum(
    fft_results: Dict,
    title: str = "FFT Frequency Spectrum",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the FFT amplitude spectrum with dominant and top peaks annotated.

    Parameters
    ----------
    fft_results : dict returned by fft_analysis.run_fft_analysis().
    title       : figure title.
    output_path : if provided, save figure as PNG to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    freqs = fft_results["freqs"]
    magnitudes = fft_results["magnitudes"]
    dom_freq = fft_results["dominant_freq"]
    peaks = fft_results["top_peaks"]

    fig, ax = plt.subplots(figsize=(11, 4), dpi=_DPI)

    ax.plot(freqs, magnitudes, color=_COLOR_FFT, linewidth=_LINEWIDTH, label="Spectrum")

    # Annotate top peaks — first peak in red, rest in orange
    for i, (freq, mag) in enumerate(peaks):
        color = "red" if i == 0 else "darkorange"
        ax.axvline(x=freq, color=color, linestyle="--", alpha=0.55, linewidth=0.8)
        ax.annotate(
            f"{freq:.1f} Hz",
            xy=(freq, mag),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
            clip_on=True,
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (g)")
    ax.set_title(
        f"{title}   |   Dominant: {dom_freq:.2f} Hz",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")

    return fig


def plot_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    fft_before: Dict,
    fft_after: Dict,
    label_before: str = "Before",
    label_after: str = "After",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    2×2 side-by-side comparison of two vibration tests.

    Layout:
        [0,0] Before — time domain   [0,1] After — time domain
        [1,0] Before — FFT spectrum  [1,1] After — FFT spectrum

    Parameters
    ----------
    df_before / df_after   : cleaned DataFrames for each condition.
    fft_before / fft_after : dicts returned by run_fft_analysis().
    label_before / label_after : display labels.
    output_path            : if provided, save figure as PNG.

    Returns
    -------
    matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=_DPI)

    _plot_magnitude(axes[0, 0], df_before, label_before, color=_COLOR_MAG)
    _plot_magnitude(axes[0, 1], df_after, label_after, color=_COLOR_Z)
    _plot_spectrum(axes[1, 0], fft_before, label_before, color=_COLOR_FFT)
    _plot_spectrum(axes[1, 1], fft_after, label_after, color=_COLOR_Y)

    fig.suptitle(
        f"Vibration Comparison: {label_before} vs {label_after}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")

    return fig


# ── Private helpers ────────────────────────────────────────────────────────────

def _plot_magnitude(ax: plt.Axes, df: pd.DataFrame, label: str, color: str) -> None:
    ax.plot(df["timestamp"], df["magnitude"], color=color, linewidth=_LINEWIDTH)
    ax.set_title(f"{label} — Time Domain", fontsize=10, fontweight="bold")
    ax.set_ylabel("Magnitude (g)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)


def _plot_spectrum(ax: plt.Axes, fft_results: Dict, label: str, color: str) -> None:
    dom = fft_results["dominant_freq"]
    ax.plot(fft_results["freqs"], fft_results["magnitudes"], color=color, linewidth=_LINEWIDTH)
    ax.axvline(x=dom, color="red", linestyle="--", alpha=0.6, linewidth=0.8,
               label=f"Peak: {dom:.1f} Hz")
    ax.set_title(f"{label} — FFT Spectrum", fontsize=10, fontweight="bold")
    ax.set_ylabel("Magnitude (g)")
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
