"""
comparator.py — Before/after vibration test comparison.

Computes metric deltas and FFT results for two test conditions
and formats a plain-text comparison summary.
"""

from typing import Dict
import pandas as pd

from vibecheck.metrics import get_all_metrics
from vibecheck.fft_analysis import run_fft_analysis


def compare(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    fs: float = 100.0,
    label_before: str = "Before",
    label_after: str = "After",
) -> Dict:
    """
    Compare vibration metrics and FFT results between two test conditions.

    Parameters
    ----------
    df_before / df_after : cleaned DataFrames for each condition.
    fs                   : sample rate in Hz.
    label_before / label_after : display labels for the report.

    Returns
    -------
    dict with keys:
        label_before, label_after
        metrics_before, metrics_after  — raw metric dicts
        fft_before, fft_after          — raw FFT result dicts
        deltas                         — per-metric dict with before, after,
                                         delta, and pct_change values
    """
    metrics_before = get_all_metrics(df_before)
    metrics_after = get_all_metrics(df_after)

    fft_before = run_fft_analysis(df_before, fs=fs)
    fft_after = run_fft_analysis(df_after, fs=fs)

    deltas: Dict[str, Dict] = {}
    for key in metrics_before:
        val_before = metrics_before[key]
        val_after = metrics_after.get(key, 0.0)
        delta = val_after - val_before
        pct_change = (delta / val_before * 100.0) if val_before != 0.0 else 0.0
        deltas[key] = {
            "before": val_before,
            "after": val_after,
            "delta": delta,
            "pct_change": pct_change,
        }

    return {
        "label_before": label_before,
        "label_after": label_after,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fft_before": fft_before,
        "fft_after": fft_after,
        "deltas": deltas,
    }


def format_comparison_summary(comparison: Dict) -> str:
    """
    Format a plain-text before/after comparison summary table.

    Parameters
    ----------
    comparison : dict returned by compare().

    Returns
    -------
    Formatted string ready for printing or saving to a file.
    """
    lb = comparison["label_before"]
    la = comparison["label_after"]
    deltas = comparison["deltas"]
    fft_b = comparison["fft_before"]
    fft_a = comparison["fft_after"]

    sep = "=" * 64
    thin = "-" * 58

    _key_metrics = [
        ("rms_magnitude",          "RMS Magnitude (g)"),
        ("peak_magnitude",         "Peak Magnitude (g)"),
        ("crest_factor_magnitude", "Crest Factor"),
        ("rms_x",                  "RMS X (g)"),
        ("rms_y",                  "RMS Y (g)"),
        ("rms_z",                  "RMS Z (g)"),
        ("peak_x",                 "Peak X (g)"),
        ("peak_y",                 "Peak Y (g)"),
        ("peak_z",                 "Peak Z (g)"),
    ]

    lines = [
        sep,
        "  VibeCheck — Before / After Comparison",
        f"  {lb}  →  {la}",
        sep,
        "",
        f"  {'Metric':<30} {'Before':>10} {'After':>10} {'Change':>10}",
        "  " + thin,
    ]

    for key, display_label in _key_metrics:
        if key not in deltas:
            continue
        d = deltas[key]
        direction = "▲" if d["delta"] > 0 else "▼"
        lines.append(
            f"  {display_label:<30} {d['before']:>10.4f} {d['after']:>10.4f} "
            f"  {direction}{abs(d['pct_change']):>6.1f}%"
        )

    # Dominant frequency row
    dom_before = fft_b["dominant_freq"]
    dom_after = fft_a["dominant_freq"]
    dom_delta = dom_after - dom_before
    direction = "▲" if dom_delta > 0 else "▼"
    lines += [
        "  " + thin,
        f"  {'Dominant Frequency (Hz)':<30} {dom_before:>10.2f} {dom_after:>10.2f} "
        f"  {direction}{abs(dom_delta):>5.2f} Hz",
        "",
        sep,
    ]

    return "\n".join(lines)
