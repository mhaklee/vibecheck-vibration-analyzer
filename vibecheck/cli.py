"""
cli.py — Command-line interface for VibeCheck.

Usage examples:

  # Analyze a single file
  vibecheck analyze data/examples/fan_on_desk.csv --save-report

  # Compare two files
  vibecheck compare data/examples/fan_on_desk.csv data/examples/fan_on_foam.csv \\
      --label-before "Fan on desk" --label-after "Fan on foam" --save-report
"""

import argparse
import os
import sys

from vibecheck.loader import load_csv, infer_sample_rate
from vibecheck.cleaner import clean
from vibecheck.metrics import get_all_metrics
from vibecheck.fft_analysis import run_fft_analysis
from vibecheck.plotter import plot_time_domain, plot_fft_spectrum, plot_comparison
from vibecheck.reporter import generate_report
from vibecheck.comparator import compare, format_comparison_summary


# ── Sub-command handlers ───────────────────────────────────────────────────────

def _run_analyze(args: argparse.Namespace) -> None:
    """Handler for the 'analyze' sub-command."""
    print(f"\nVibeCheck — loading '{args.file}' ...")

    try:
        df = load_csv(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    fs = args.sample_rate if args.sample_rate else infer_sample_rate(df)
    print(f"  Samples      : {len(df)}")
    print(f"  Sample rate  : {fs} Hz")
    print(f"  Duration     : {len(df) / fs:.2f} s")

    print("\nCleaning signal ...")
    df_clean = clean(
        df,
        gravity=not args.no_gravity,
        gravity_method=args.gravity_method,
        detrend=not args.no_detrend,
        lowpass=args.lowpass,
        lowpass_cutoff=args.lowpass_cutoff,
        fs=fs,
    )

    print("Calculating metrics ...")
    metrics = get_all_metrics(df_clean)

    print("Running FFT analysis ...")
    fft_results = run_fft_analysis(df_clean, fs=fs, axis=args.axis, n_peaks=args.n_peaks)

    label = args.label or os.path.splitext(os.path.basename(args.file))[0]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.no_plots:
        print("Generating plots ...")
        time_path = os.path.join(output_dir, f"{label}_time.png")
        fft_path = os.path.join(output_dir, f"{label}_fft.png")
        plot_time_domain(df_clean, title=f"{label} — Time Domain", output_path=time_path)
        plot_fft_spectrum(fft_results, title=f"{label} — FFT Spectrum", output_path=fft_path)
        print(f"  Saved: {time_path}")
        print(f"  Saved: {fft_path}")

    report_path = os.path.join(output_dir, f"report_{label}.txt") if args.save_report else None
    report = generate_report(metrics, fft_results, label=label, output_path=report_path)

    print()
    print(report)

    if report_path:
        print(f"\nReport saved to: {report_path}")


def _run_compare(args: argparse.Namespace) -> None:
    """Handler for the 'compare' sub-command."""
    print(f"\nVibeCheck — comparing '{args.before}' vs '{args.after}' ...")

    try:
        df_before = load_csv(args.before)
        df_after = load_csv(args.after)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    fs = args.sample_rate if args.sample_rate else infer_sample_rate(df_before)

    label_before = args.label_before or os.path.splitext(os.path.basename(args.before))[0]
    label_after = args.label_after or os.path.splitext(os.path.basename(args.after))[0]

    print("Cleaning signals ...")
    df_before_clean = clean(df_before, fs=fs)
    df_after_clean = clean(df_after, fs=fs)

    print("Running comparison ...")
    result = compare(
        df_before_clean,
        df_after_clean,
        fs=fs,
        label_before=label_before,
        label_after=label_after,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.no_plots:
        print("Generating comparison plot ...")
        plot_name = f"comparison_{label_before}_vs_{label_after}.png"
        plot_path = os.path.join(output_dir, plot_name)
        plot_comparison(
            df_before_clean,
            df_after_clean,
            result["fft_before"],
            result["fft_after"],
            label_before=label_before,
            label_after=label_after,
            output_path=plot_path,
        )
        print(f"  Saved: {plot_path}")

    summary = format_comparison_summary(result)
    print()
    print(summary)

    if args.save_report:
        report_name = f"report_comparison_{label_before}_vs_{label_after}.txt"
        report_path = os.path.join(output_dir, report_name)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\nReport saved to: {report_path}")


# ── Argument parser ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vibecheck",
        description="VibeCheck — Lightweight vibration analysis from accelerometer CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  vibecheck analyze fan_on_desk.csv --save-report\n"
            "  vibecheck compare before.csv after.csv --label-before 'No pad' "
            "--label-after 'Foam pad' --save-report\n"
        ),
    )
    parser.add_argument("--version", action="version", version="vibecheck 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── analyze ──────────────────────────────────────────────────────────────
    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a single accelerometer CSV file.",
        description="Load, clean, and analyse a single accelerometer recording.",
    )
    analyze.add_argument("file", help="Path to the accelerometer CSV file.")
    analyze.add_argument(
        "--label",
        default=None,
        help="Short label for this test (default: derived from filename).",
    )
    analyze.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        dest="sample_rate",
        metavar="HZ",
        help="Override sample rate in Hz (default: inferred from timestamps).",
    )
    analyze.add_argument(
        "--axis",
        default="magnitude",
        choices=["x", "y", "z", "magnitude"],
        help="Axis to use for FFT analysis (default: magnitude).",
    )
    analyze.add_argument(
        "--n-peaks",
        type=int,
        default=5,
        dest="n_peaks",
        metavar="N",
        help="Number of FFT peaks to detect (default: 5).",
    )
    analyze.add_argument(
        "--output-dir",
        default="outputs",
        dest="output_dir",
        metavar="DIR",
        help="Directory for plots and reports (default: outputs).",
    )
    analyze.add_argument(
        "--no-gravity",
        action="store_true",
        dest="no_gravity",
        help="Skip gravity removal.",
    )
    analyze.add_argument(
        "--gravity-method",
        default="mean",
        choices=["mean", "highpass"],
        dest="gravity_method",
        help="Gravity removal method (default: mean).",
    )
    analyze.add_argument(
        "--no-detrend",
        action="store_true",
        dest="no_detrend",
        help="Skip linear detrending.",
    )
    analyze.add_argument(
        "--lowpass",
        action="store_true",
        help="Apply a low-pass Butterworth filter.",
    )
    analyze.add_argument(
        "--lowpass-cutoff",
        type=float,
        default=50.0,
        dest="lowpass_cutoff",
        metavar="HZ",
        help="Low-pass filter cutoff frequency in Hz (default: 50).",
    )
    analyze.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Skip plot generation.",
    )
    analyze.add_argument(
        "--save-report",
        action="store_true",
        dest="save_report",
        help="Save the text report to the output directory.",
    )
    analyze.set_defaults(func=_run_analyze)

    # ── compare ───────────────────────────────────────────────────────────────
    compare_cmd = subparsers.add_parser(
        "compare",
        help="Compare two accelerometer CSV files (before/after).",
        description="Load and compare two accelerometer recordings side by side.",
    )
    compare_cmd.add_argument("before", help="Path to the 'before' CSV file.")
    compare_cmd.add_argument("after", help="Path to the 'after' CSV file.")
    compare_cmd.add_argument(
        "--label-before",
        default=None,
        dest="label_before",
        help="Display label for the 'before' condition (default: filename).",
    )
    compare_cmd.add_argument(
        "--label-after",
        default=None,
        dest="label_after",
        help="Display label for the 'after' condition (default: filename).",
    )
    compare_cmd.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        dest="sample_rate",
        metavar="HZ",
        help="Override sample rate in Hz (default: inferred from before file).",
    )
    compare_cmd.add_argument(
        "--output-dir",
        default="outputs",
        dest="output_dir",
        metavar="DIR",
        help="Directory for plots and reports (default: outputs).",
    )
    compare_cmd.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Skip plot generation.",
    )
    compare_cmd.add_argument(
        "--save-report",
        action="store_true",
        dest="save_report",
        help="Save the comparison report to the output directory.",
    )
    compare_cmd.set_defaults(func=_run_compare)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
