"""
Compute per-surface precision, recall and F1 from predictions.csv.

This loads the predictions file produced by `sdnet-infer`, groups by
the `surface` column, and computes the standard binary-classification
metrics on the labelled subset of each surface. It prints a neat table
to stdout and also writes a small JSON report.

Run from the project root:
    uv run python scripts/per_surface_metrics.py

Optional flags:
    --predictions PATH         Path to predictions.csv (default: data/results/predictions.csv)
    --split test               Restrict to a specific split (default: test only)
                               Use 'all' to evaluate across train+val+test combined.
    --report-path PATH         Where to write the JSON report
                               (default: data/results/per_surface_metrics.json)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int | list[list[int]]]:
    """Compute the full metric set on a labelled subset."""
    y_true = df["target"].astype(int).to_numpy()
    y_pred = df["predicted_target"].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "rows": int(len(df)),
        "cracked_rows": int((y_true == 1).sum()),
        "non_cracked_rows": int((y_true == 0).sum()),
        "predicted_cracked": int((y_pred == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def format_row(label: str, m: dict) -> str:
    return (
        f"  {label:18s} | n={m['rows']:>5d} | "
        f"acc={m['accuracy']:.3f}  bal={m['balanced_accuracy']:.3f}  "
        f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-surface metrics from predictions.csv")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/results/predictions.csv"),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to evaluate. Use 'all' to combine train+validation+test.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/results/per_surface_metrics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading predictions from: {args.predictions}")
    df = pd.read_csv(args.predictions)

    # Filter to rows that actually have ground truth labels.
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    # Filter to the requested split.
    if args.split.lower() != "all":
        before = len(df)
        df = df[df["split"] == args.split].copy()
        print(f"Filtered to split='{args.split}': {len(df):,} rows (was {before:,})")
    else:
        print(f"Using ALL splits (train + validation + test): {len(df):,} rows")

    if df.empty:
        print("No labelled rows found. Aborting.")
        return

    # Overall metrics (anchor row, same as what's in Table 4.2 already).
    overall = compute_metrics(df)

    # Per-surface metrics.
    per_surface: dict[str, dict] = {}
    for surface, group in df.groupby("surface"):
        per_surface[str(surface)] = compute_metrics(group)

    # Pretty print.
    print()
    print("=" * 100)
    print(f"PER-SURFACE METRICS  (split = {args.split})")
    print("=" * 100)
    print(format_row("Overall", overall))
    print("  " + "-" * 90)
    surface_labels = {
        "bridge_deck": "Bridge Deck (D)",
        "wall":        "Wall (W)",
        "pavement":    "Pavement (P)",
        "unknown":     "Unknown",
    }
    # Print in the canonical D, W, P order if available
    ordered_keys = ["bridge_deck", "wall", "pavement"]
    for key in ordered_keys:
        if key in per_surface:
            print(format_row(surface_labels.get(key, key), per_surface[key]))
    # Print any remaining surfaces not in the canonical order
    for key, m in per_surface.items():
        if key not in ordered_keys:
            print(format_row(surface_labels.get(key, key), m))
    print("=" * 100)

    # --- Print the values formatted for Table 4.3 of the thesis ---
    print("\nReady-to-paste values for Table 4.3 of the thesis:")
    print()
    print("  Surface          | Precision | Recall | F1")
    print("  ---------------- | --------- | ------ | -----")
    for key in ordered_keys:
        if key in per_surface:
            m = per_surface[key]
            print(
                f"  {surface_labels[key]:16s} |   {m['precision']:.2f}    |  {m['recall']:.2f}  | {m['f1']:.2f}"
            )

    # --- Write the JSON report ---
    report = {
        "created_at": utc_now_iso(),
        "predictions_path": str(args.predictions.resolve()),
        "split": args.split,
        "overall": overall,
        "per_surface": per_surface,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nWrote detailed report to: {args.report_path}")


if __name__ == "__main__":
    main()
