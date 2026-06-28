"""
Compute distribution statistics from data/results/localizations.csv.

Prints:
  - Standard describe() table (count, mean, std, min, 25%, 50%, 75%, max)
  - P90, P95, P99 percentiles
  - Two columns ready to paste into Table 4.4 (Median, P95, P99 rows)

Run from the project root:
    uv run python scripts/distribution_stats.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = ["crack_area_pct", "crack_length_px", "mean_width_px", "max_width_px"]
PRETTY_NAMES = {
    "crack_area_pct":  "Area Percent",
    "crack_length_px": "Length (px)",
    "mean_width_px":   "Mean Width (px)",
    "max_width_px":    "Max Width (px)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute distribution statistics for localisation measurements.")
    parser.add_argument(
        "--localizations",
        type=Path,
        default=Path("data/results/localizations.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading: {args.localizations}")
    df = pd.read_csv(args.localizations)
    print(f"Loaded {len(df):,} rows")

    # Drop any rows that don't have valid measurements (e.g. error rows)
    sub = df[COLUMNS].dropna()
    print(f"Valid rows for statistics: {len(sub):,}\n")

    # Standard describe table
    print("=" * 70)
    print("describe() table")
    print("=" * 70)
    described = sub.describe()
    # Add P90, P95, P99 manually since describe() default percentiles are 25/50/75
    p90 = sub.quantile(0.90)
    p95 = sub.quantile(0.95)
    p99 = sub.quantile(0.99)
    described.loc["90%"] = p90
    described.loc["95%"] = p95
    described.loc["99%"] = p99
    # Re-order so percentiles appear in ascending order
    order = ["count", "mean", "std", "min", "25%", "50%", "75%", "90%", "95%", "99%", "max"]
    described = described.reindex(order)
    # Print with sensible formatting for area_pct (which is 0..1 in the CSV)
    print(described.to_string(float_format=lambda v: f"{v:10.4f}"))

    # Ready-to-paste rows for Table 4.4 of the thesis
    print()
    print("=" * 70)
    print("Ready-to-paste values for Table 4.4 (with area expressed as %)")
    print("=" * 70)
    print()
    print("  Statistic | Area Percent | Length (px) | Mean Width (px) | Max Width (px)")
    print("  --------- | ------------ | ----------- | --------------- | ---------------")

    def fmt_row(label: str, series: pd.Series) -> None:
        area_pct = series["crack_area_pct"] * 100
        print(
            f"  {label:9s} |    {area_pct:5.2f} %    |   {series['crack_length_px']:6.1f}    |"
            f"       {series['mean_width_px']:5.1f}     |       {series['max_width_px']:5.1f}"
        )

    fmt_row("Mean",   sub.mean())
    fmt_row("Median", sub.median())
    fmt_row("P95",    p95)
    fmt_row("P99",    p99)

    # Save a JSON copy of the full describe table for the record
    out_json = args.localizations.parent / "distribution_stats.json"
    described.to_json(out_json, indent=2)
    print(f"\nFull distribution table written to: {out_json}")


if __name__ == "__main__":
    main()