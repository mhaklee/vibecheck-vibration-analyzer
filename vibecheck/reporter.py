"""
reporter.py — Plain-English interpretation and summary report generation.

Produces human-readable diagnostics from vibration metrics and FFT results.
"""

import os
from datetime import datetime
from typing import Dict, Optional


def interpret_metrics(metrics: Dict[str, float]) -> str:
    """
    Return a plain-English interpretation of vibration metrics.

    Reads rms_magnitude, peak_magnitude, and crest_factor_magnitude.
    Falls back to z-axis values if magnitude keys are absent.
    """
    lines = []

    rms = metrics.get("rms_magnitude", metrics.get("rms_z", 0.0))
    crest = metrics.get("crest_factor_magnitude", metrics.get("crest_factor_z", 0.0))

    # RMS interpretation
    if rms < 0.01:
        lines.append(
            "RMS vibration is very low. The system appears essentially stationary, "
            "or the sensor may not be capturing meaningful motion."
        )
    elif rms < 0.05:
        lines.append(
            "RMS vibration is low, consistent with a well-balanced or lightly loaded system."
        )
    elif rms < 0.2:
        lines.append(
            "RMS vibration is moderate. Typical of a running motor, fan, or "
            "lightly loaded mechanical system."
        )
    elif rms < 0.5:
        lines.append(
            "RMS vibration is elevated. This may indicate imbalance, loose components, "
            "or a resonance condition worth investigating."
        )
    else:
        lines.append(
            "RMS vibration is high. Significant imbalance, mechanical looseness, "
            "or structural resonance is likely."
        )

    # Crest factor interpretation
    if crest < 2.0:
        lines.append(
            "Crest factor is low, suggesting smooth and continuous vibration "
            "with no notable impulsive events."
        )
    elif crest < 4.0:
        lines.append(
            "Crest factor is moderate. Some periodic impacts or slight irregularities "
            "may be present."
        )
    elif crest < 6.0:
        lines.append(
            "Crest factor is elevated. Impulsive vibration detected — possible bearing wear, "
            "mechanical looseness, or intermittent knocking."
        )
    else:
        lines.append(
            "Crest factor is high. Strong impulsive events are present. "
            "Inspect for loose fasteners, bearing damage, or structural faults."
        )

    return "\n".join(lines)


def interpret_fft(fft_results: Dict) -> str:
    """
    Return a plain-English interpretation of FFT analysis results.
    """
    lines = []
    dom_freq = fft_results.get("dominant_freq", 0.0)
    peaks = fft_results.get("top_peaks", [])

    lines.append(f"Dominant vibration frequency: {dom_freq:.2f} Hz")

    if dom_freq < 1.0:
        lines.append(
            "The dominant frequency is very low, likely capturing slow structural "
            "oscillations or sensor drift rather than mechanical vibration."
        )
    elif dom_freq < 10.0:
        lines.append(
            f"Low-frequency vibration at {dom_freq:.1f} Hz may correspond to a slowly "
            f"rotating component (~{dom_freq * 60:.0f} RPM) or large structural motion."
        )
    elif dom_freq < 100.0:
        lines.append(
            f"The dominant frequency of {dom_freq:.1f} Hz falls in the typical range "
            f"for rotating machinery (~{dom_freq * 60:.0f} RPM). This could be a "
            f"motor shaft, fan blade pass, or drive component."
        )
    elif dom_freq < 500.0:
        lines.append(
            f"Mid-to-high frequency vibration at {dom_freq:.1f} Hz. May correspond to "
            f"gear mesh frequency, blade harmonics, or a structural resonance mode."
        )
    else:
        lines.append(
            f"High-frequency vibration at {dom_freq:.1f} Hz. Could indicate bearing "
            f"defects, high-speed rotating components, or structural resonance."
        )

    if len(peaks) > 1:
        other_peaks = ", ".join(f"{f:.1f} Hz" for f, _ in peaks[1:5])
        lines.append(f"Additional spectral peaks detected at: {other_peaks}.")

    return "\n".join(lines)


def generate_report(
    metrics: Dict[str, float],
    fft_results: Dict,
    label: str = "Vibration Test",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a formatted plain-text summary report.

    Combines vibration metrics, FFT results, and plain-English interpretations.

    Parameters
    ----------
    metrics     : dict from metrics.get_all_metrics().
    fft_results : dict from fft_analysis.run_fft_analysis().
    label       : descriptive name for this test run.
    output_path : if provided, write the report to this file path.

    Returns
    -------
    The full report as a string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 62
    thin = "-" * 40

    _metric_labels = [
        ("rms_x",               "RMS X (g)"),
        ("rms_y",               "RMS Y (g)"),
        ("rms_z",               "RMS Z (g)"),
        ("rms_magnitude",       "RMS Magnitude (g)"),
        ("peak_x",              "Peak X (g)"),
        ("peak_y",              "Peak Y (g)"),
        ("peak_z",              "Peak Z (g)"),
        ("peak_magnitude",      "Peak Magnitude (g)"),
        ("crest_factor_x",      "Crest Factor X"),
        ("crest_factor_y",      "Crest Factor Y"),
        ("crest_factor_z",      "Crest Factor Z"),
        ("crest_factor_magnitude", "Crest Factor Magnitude"),
    ]

    lines = [
        sep,
        "  VibeCheck — Vibration Analysis Report",
        f"  Test label : {label}",
        f"  Generated  : {timestamp}",
        sep,
        "",
        "VIBRATION METRICS",
        thin,
    ]

    for key, display_label in _metric_labels:
        if key in metrics:
            lines.append(f"  {display_label:<30}  {metrics[key]:.5f}")

    lines += [
        "",
        "FFT ANALYSIS",
        thin,
        f"  Dominant frequency   : {fft_results.get('dominant_freq', 0.0):.2f} Hz",
        f"  Analysis axis        : {fft_results.get('axis', 'magnitude')}",
        f"  Sample rate          : {fft_results.get('fs', 0.0):.2f} Hz",
        "",
        "  Top spectral peaks:",
    ]

    for rank, (freq, mag) in enumerate(fft_results.get("top_peaks", []), start=1):
        lines.append(f"    {rank}.  {freq:>8.2f} Hz   magnitude: {mag:.5f} g")

    lines += [
        "",
        "INTERPRETATION",
        thin,
        interpret_metrics(metrics),
        "",
        interpret_fft(fft_results),
        "",
        sep,
    ]

    report = "\n".join(lines)

    if output_path:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report
