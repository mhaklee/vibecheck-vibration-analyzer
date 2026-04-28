"""
app.py — Streamlit dashboard for VibeCheck.

Run with:
    streamlit run vibecheck/app.py
"""

import io
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from vibecheck.loader import load_csv, infer_sample_rate
from vibecheck.cleaner import clean
from vibecheck.metrics import get_all_metrics
from vibecheck.fft_analysis import run_fft_analysis
from vibecheck.plotter import plot_time_domain, plot_fft_spectrum, plot_comparison
from vibecheck.reporter import generate_report, interpret_metrics, interpret_fft
from vibecheck.comparator import compare, format_comparison_summary


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeCheck",
    page_icon="📳",
    layout="wide",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig_to_bytes(fig: plt.Figure) -> bytes:
    """Convert a matplotlib figure to PNG bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _load_uploaded(uploaded_file) -> pd.DataFrame:
    """Write an uploaded file to a temp path and load it with load_csv."""
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        df = load_csv(tmp_path)
    finally:
        os.unlink(tmp_path)
    return df


def _metrics_columns(metrics: dict) -> None:
    """Render RMS / Peak / Crest Factor in three columns."""
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("**RMS (g)**")
        for axis in ("x", "y", "z", "magnitude"):
            key = f"rms_{axis}"
            if key in metrics:
                st.metric(axis.upper(), f"{metrics[key]:.4f}")
    with m2:
        st.markdown("**Peak (g)**")
        for axis in ("x", "y", "z", "magnitude"):
            key = f"peak_{axis}"
            if key in metrics:
                st.metric(axis.upper(), f"{metrics[key]:.4f}")
    with m3:
        st.markdown("**Crest Factor**")
        for axis in ("x", "y", "z", "magnitude"):
            key = f"crest_factor_{axis}"
            if key in metrics:
                st.metric(axis.upper(), f"{metrics[key]:.2f}")


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("VibeCheck ⚙️")
st.sidebar.markdown(
    "Lightweight vibration analysis from accelerometer CSV data.  \n"
    "[GitHub](https://github.com/mhaklee/vibecheck-vibration-analyzer)"
)
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["Single File Analysis", "Before / After Comparison"])

st.sidebar.markdown("---")
st.sidebar.subheader("Signal Cleaning")
do_gravity = st.sidebar.checkbox("Remove gravity", value=True)
gravity_method = st.sidebar.selectbox(
    "Gravity method", ["mean", "highpass"], disabled=not do_gravity
)
do_detrend = st.sidebar.checkbox("Detrend signal", value=True)
do_lowpass = st.sidebar.checkbox("Apply low-pass filter", value=False)
lowpass_cutoff = st.sidebar.slider(
    "Low-pass cutoff (Hz)", min_value=1.0, max_value=200.0, value=50.0, step=1.0,
    disabled=not do_lowpass,
)

st.sidebar.markdown("---")
st.sidebar.subheader("FFT Settings")
fft_axis = st.sidebar.selectbox("Analysis axis", ["magnitude", "x", "y", "z"])
n_peaks = st.sidebar.slider("Top peaks to detect", min_value=1, max_value=10, value=5)
sample_rate_override = st.sidebar.number_input(
    "Sample rate override (Hz)", min_value=0.0, value=0.0, step=1.0,
    help="Leave at 0 to infer from timestamps automatically.",
)


# ── Main ───────────────────────────────────────────────────────────────────────
st.title("📳 VibeCheck — Vibration Analyzer")
st.markdown(
    "Upload accelerometer CSV data to calculate vibration metrics, "
    "run FFT frequency analysis, and generate a plain-English report."
)


# ════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE FILE
# ════════════════════════════════════════════════════════════════
if mode == "Single File Analysis":
    st.header("Single File Analysis")
    uploaded = st.file_uploader(
        "Upload accelerometer CSV  (columns: timestamp, x, y, z)",
        type=["csv"],
        key="single",
    )

    if not uploaded:
        st.info("Upload a CSV file to get started.")
        st.stop()

    # Load & clean
    try:
        df_raw = _load_uploaded(uploaded)
    except Exception as exc:
        st.error(f"Failed to load file: {exc}")
        st.stop()

    fs = float(sample_rate_override) if sample_rate_override > 0 else infer_sample_rate(df_raw)

    df_clean = clean(
        df_raw,
        gravity=do_gravity,
        gravity_method=gravity_method,
        detrend=do_detrend,
        lowpass=do_lowpass,
        lowpass_cutoff=lowpass_cutoff,
        fs=fs,
    )

    metrics = get_all_metrics(df_clean)
    fft_results = run_fft_analysis(df_clean, fs=fs, axis=fft_axis, n_peaks=n_peaks)

    # Info bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{len(df_clean):,}")
    c2.metric("Sample Rate", f"{fs:.1f} Hz")
    c3.metric("Duration", f"{len(df_clean) / fs:.2f} s")
    c4.metric("Dominant Freq", f"{fft_results['dominant_freq']:.2f} Hz")

    st.markdown("---")

    # Metrics
    st.subheader("Vibration Metrics")
    _metrics_columns(metrics)

    st.markdown("---")

    # Time-domain plot
    st.subheader("Time-Domain Signal")
    fig_time = plot_time_domain(df_clean, title=f"{uploaded.name} — Time Domain")
    st.pyplot(fig_time)
    st.download_button(
        "⬇ Download time-domain plot",
        data=_fig_to_bytes(fig_time),
        file_name="time_domain.png",
        mime="image/png",
    )
    plt.close(fig_time)

    # FFT spectrum plot
    st.subheader("FFT Frequency Spectrum")
    fig_fft = plot_fft_spectrum(fft_results, title=f"{uploaded.name} — FFT Spectrum")
    st.pyplot(fig_fft)
    st.download_button(
        "⬇ Download FFT plot",
        data=_fig_to_bytes(fig_fft),
        file_name="fft_spectrum.png",
        mime="image/png",
    )
    plt.close(fig_fft)

    # Top peaks table
    st.subheader("Top Frequency Peaks")
    peaks_df = pd.DataFrame(fft_results["top_peaks"], columns=["Frequency (Hz)", "Magnitude (g)"])
    peaks_df.index = peaks_df.index + 1
    st.dataframe(
        peaks_df.style.format({"Frequency (Hz)": "{:.2f}", "Magnitude (g)": "{:.5f}"}),
        use_container_width=True,
    )

    st.markdown("---")

    # Interpretation
    st.subheader("Interpretation")
    st.info(interpret_metrics(metrics))
    st.info(interpret_fft(fft_results))

    # Full report
    st.subheader("Full Report")
    label = os.path.splitext(uploaded.name)[0]
    report = generate_report(metrics, fft_results, label=label)
    st.text(report)
    st.download_button(
        "⬇ Download report (.txt)",
        data=report.encode("utf-8"),
        file_name=f"report_{label}.txt",
        mime="text/plain",
    )


# ════════════════════════════════════════════════════════════════
# MODE 2 — BEFORE / AFTER COMPARISON
# ════════════════════════════════════════════════════════════════
else:
    st.header("Before / After Comparison")

    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_before = st.file_uploader("Upload BEFORE CSV", type=["csv"], key="before")
        label_before = st.text_input("Label (before)", value="Before")
    with col_b:
        uploaded_after = st.file_uploader("Upload AFTER CSV", type=["csv"], key="after")
        label_after = st.text_input("Label (after)", value="After")

    if not uploaded_before or not uploaded_after:
        st.info("Upload both CSV files to run the comparison.")
        st.stop()

    try:
        df_before_raw = _load_uploaded(uploaded_before)
        df_after_raw = _load_uploaded(uploaded_after)
    except Exception as exc:
        st.error(f"Failed to load files: {exc}")
        st.stop()

    fs = float(sample_rate_override) if sample_rate_override > 0 else infer_sample_rate(df_before_raw)

    clean_kwargs = dict(
        gravity=do_gravity,
        gravity_method=gravity_method,
        detrend=do_detrend,
        lowpass=do_lowpass,
        lowpass_cutoff=lowpass_cutoff,
        fs=fs,
    )
    df_before_clean = clean(df_before_raw, **clean_kwargs)
    df_after_clean = clean(df_after_raw, **clean_kwargs)

    result = compare(
        df_before_clean,
        df_after_clean,
        fs=fs,
        label_before=label_before,
        label_after=label_after,
    )

    # Key metric deltas
    st.subheader("Key Metric Changes")
    key_metrics = [
        ("rms_magnitude",          "RMS Magnitude (g)"),
        ("peak_magnitude",         "Peak Magnitude (g)"),
        ("crest_factor_magnitude", "Crest Factor"),
    ]
    delta_cols = st.columns(len(key_metrics))
    for col, (key, display_label) in zip(delta_cols, key_metrics):
        if key in result["deltas"]:
            d = result["deltas"][key]
            col.metric(
                label=display_label,
                value=f"{d['after']:.4f}",
                delta=f"{d['pct_change']:+.1f}%",
                delta_color="inverse",
            )

    dom_b = result["fft_before"]["dominant_freq"]
    dom_a = result["fft_after"]["dominant_freq"]
    st.metric(
        "Dominant Frequency Shift",
        value=f"{dom_a:.2f} Hz",
        delta=f"{dom_a - dom_b:+.2f} Hz  (was {dom_b:.2f} Hz)",
    )

    st.markdown("---")

    # Comparison plots
    st.subheader("Comparison Plots")
    fig_comp = plot_comparison(
        df_before_clean,
        df_after_clean,
        result["fft_before"],
        result["fft_after"],
        label_before=label_before,
        label_after=label_after,
    )
    st.pyplot(fig_comp)
    st.download_button(
        "⬇ Download comparison plot",
        data=_fig_to_bytes(fig_comp),
        file_name="comparison.png",
        mime="image/png",
    )
    plt.close(fig_comp)

    st.markdown("---")

    # Detailed metrics side by side
    st.subheader("Detailed Metrics")
    tab_before, tab_after = st.tabs([label_before, label_after])
    with tab_before:
        _metrics_columns(result["metrics_before"])
    with tab_after:
        _metrics_columns(result["metrics_after"])

    st.markdown("---")

    # Summary report
    st.subheader("Comparison Summary")
    summary = format_comparison_summary(result)
    st.text(summary)
    st.download_button(
        "⬇ Download comparison report (.txt)",
        data=summary.encode("utf-8"),
        file_name=f"report_{label_before}_vs_{label_after}.txt",
        mime="text/plain",
    )
