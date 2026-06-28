"""
Re-tune the decision threshold of an already-trained crack classifier
WITHOUT retraining the model.

What this script does:
  1. Loads the saved Extra Trees model from data/models/crack_classifier.joblib
  2. Loads the manifest and extracts features for VALIDATION and TEST splits
     (one-time ~14 minute cost, then everything is in memory)
  3. Sweeps thresholds from 0.05 to 0.95 in 0.01 steps
  4. Prints a comparison table for SIX optimisation strategies:
       - accuracy (current default — re-confirm)
       - balanced_accuracy
       - f1
       - precision (recall floor 0.5)
       - recall (precision floor 0.5)
       - balanced_accuracy with recall floor 0.7  <-- inspection-triage default
  5. Writes detailed per-strategy metrics to
       data/results/threshold_retune_report.json
  6. Updates the model bundle's decision_threshold with the recommended choice
       (saves a backup of the old bundle first)

Run from the project root:
    uv run python scripts/retune_threshold.py

Optional flags:
    --strategy balanced_recall_floor_0.7    pick which strategy becomes the new default
    --no-update-bundle                       just report, do not modify the model file
    --val-only                               skip test extraction (faster, less complete)
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from sdnet_pipeline.config import (
    DEFAULT_MANIFEST,
    DEFAULT_METRICS,
    DEFAULT_MODEL,
    RESULTS_DIR,
    ensure_data_dirs,
)
from sdnet_pipeline.features import extract_features


# ---------------------------------------------------------------------------
# Strategies: name -> (target_metric, min_recall, min_precision)
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, dict[str, float | str]] = {
    "accuracy": {
        "target": "accuracy",
        "min_recall": 0.0,
        "min_precision": 0.0,
        "note": "Current default. Tends to lean toward majority class on imbalanced data.",
    },
    "balanced_accuracy": {
        "target": "balanced_accuracy",
        "min_recall": 0.0,
        "min_precision": 0.0,
        "note": "Averages per-class recall. Better than accuracy on imbalanced data.",
    },
    "f1": {
        "target": "f1",
        "min_recall": 0.0,
        "min_precision": 0.0,
        "note": "Harmonic mean of precision and recall. Good general-purpose choice.",
    },
    "precision_recall_floor_0.5": {
        "target": "precision",
        "min_recall": 0.5,
        "min_precision": 0.0,
        "note": "Prioritise precision but require at least 50% recall.",
    },
    "recall_precision_floor_0.5": {
        "target": "recall",
        "min_recall": 0.0,
        "min_precision": 0.5,
        "note": "Prioritise recall but require at least 50% precision.",
    },
    "balanced_recall_floor_0.7": {
        "target": "balanced_accuracy",
        "min_recall": 0.7,
        "min_precision": 0.0,
        "note": "Inspection-triage operating point: balanced accuracy with recall >= 0.70.",
    },
}


def extract_split_features(df: pd.DataFrame, image_size: int, label: str) -> np.ndarray:
    """Extract features for a split with a progress bar."""
    features = []
    for path in tqdm(df["path"].tolist(), desc=f"Extracting {label} features"):
        features.append(extract_features(path, image_size=image_size))
    return np.vstack(features)


def metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float | list[list[int]]]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def sweep_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, float | list[list[int]]]]:
    candidates = np.round(np.linspace(0.05, 0.95, 91), 2)
    return [metrics_at_threshold(y_true, y_score, float(t)) for t in candidates]


def best_threshold_for_strategy(
    sweep: list[dict[str, float | list[list[int]]]],
    target: str,
    min_recall: float,
    min_precision: float,
) -> dict[str, float | list[list[int]]]:
    feasible = [
        row for row in sweep
        if row["recall"] >= min_recall and row["precision"] >= min_precision
    ]
    if not feasible:
        # Relax constraints rather than crashing — still return the best we can.
        feasible = sweep
    return max(feasible, key=lambda row: (row[target], row["recall"], row["threshold"]))


def format_row(name: str, row: dict[str, float | list[list[int]]]) -> str:
    return (
        f"  {name:34s} | t={row['threshold']:.2f} | "
        f"acc={row['accuracy']:.3f}  bal={row['balanced_accuracy']:.3f}  "
        f"P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}"
    )


def print_comparison(strategies_results: dict[str, dict]) -> None:
    print()
    print("=" * 100)
    print("THRESHOLD COMPARISON ON VALIDATION SPLIT")
    print("=" * 100)
    for name, payload in strategies_results.items():
        row = payload["validation_metrics"]
        print(format_row(name, row))
    print("=" * 100)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-tune decision threshold of an already-trained model.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=RESULTS_DIR / "threshold_retune_report.json",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="balanced_recall_floor_0.7",
        help="Strategy used to choose the recommended new default threshold.",
    )
    parser.add_argument(
        "--no-update-bundle",
        action="store_true",
        help="Print report only; do not modify the saved model bundle.",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Sweep only the validation split (skip test). Faster, less complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_dirs()

    print(f"Loading model bundle from: {args.model_path}")
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    image_size = int(bundle.get("feature_config", {}).get("image_size", 224))
    current_threshold = float(bundle.get("decision_threshold", 0.5))
    print(f"  Image size: {image_size}")
    print(f"  Current decision threshold: {current_threshold:.2f}")

    print(f"\nLoading manifest from: {args.manifest}")
    df = pd.read_csv(args.manifest)
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)
    validation_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()
    print(f"  Validation rows: {len(validation_df):,}")
    print(f"  Test rows:       {len(test_df):,}")

    print("\nExtracting validation features (~7 minutes on a laptop)...")
    x_val = extract_split_features(validation_df, image_size=image_size, label="validation")
    y_val = validation_df["target"].to_numpy()
    val_scores = model.predict_proba(x_val)[:, 1]

    val_sweep = sweep_thresholds(y_val, val_scores)
    try:
        val_roc_auc = float(roc_auc_score(y_val, val_scores))
    except ValueError:
        val_roc_auc = None

    test_sweep: list[dict] | None = None
    test_roc_auc: float | None = None
    if not args.val_only:
        print("\nExtracting test features (~7 minutes on a laptop)...")
        x_test = extract_split_features(test_df, image_size=image_size, label="test")
        y_test = test_df["target"].to_numpy()
        test_scores = model.predict_proba(x_test)[:, 1]
        test_sweep = sweep_thresholds(y_test, test_scores)
        try:
            test_roc_auc = float(roc_auc_score(y_test, test_scores))
        except ValueError:
            test_roc_auc = None

    strategies_results: dict[str, dict] = {}
    for name, cfg in STRATEGIES.items():
        best_val = best_threshold_for_strategy(
            val_sweep,
            target=str(cfg["target"]),
            min_recall=float(cfg["min_recall"]),
            min_precision=float(cfg["min_precision"]),
        )
        # Apply that same threshold to the test set, for honest reporting.
        test_at_chosen = None
        if test_sweep is not None:
            test_at_chosen = next(
                (row for row in test_sweep if abs(row["threshold"] - best_val["threshold"]) < 0.005),
                None,
            )
        strategies_results[name] = {
            "target": cfg["target"],
            "min_recall": cfg["min_recall"],
            "min_precision": cfg["min_precision"],
            "note": cfg["note"],
            "validation_metrics": best_val,
            "test_metrics_at_chosen_threshold": test_at_chosen,
        }

    print_comparison(strategies_results)

    if test_sweep is not None:
        print("\nSAME THRESHOLDS APPLIED TO HELD-OUT TEST SPLIT")
        print("=" * 100)
        for name, payload in strategies_results.items():
            row = payload["test_metrics_at_chosen_threshold"]
            if row is not None:
                print(format_row(name, row))
        print("=" * 100)
        print(f"\nValidation ROC AUC: {val_roc_auc:.4f}" if val_roc_auc else "")
        print(f"Test ROC AUC:       {test_roc_auc:.4f}" if test_roc_auc else "")

    chosen = strategies_results[args.strategy]
    chosen_threshold = float(chosen["validation_metrics"]["threshold"])
    print(f"\nRecommended strategy: {args.strategy}")
    print(f"  Note: {chosen['note']}")
    print(f"  New decision threshold: {chosen_threshold:.2f}")
    print(f"  Validation: {format_row('', chosen['validation_metrics']).split('|', 1)[1]}")
    if chosen["test_metrics_at_chosen_threshold"]:
        print(f"  Test:       {format_row('', chosen['test_metrics_at_chosen_threshold']).split('|', 1)[1]}")

    # ---- Write the JSON report ----
    report = {
        "created_at": utc_now_iso(),
        "model_path": str(args.model_path.resolve()),
        "manifest_path": str(args.manifest.resolve()),
        "image_size": image_size,
        "previous_decision_threshold": current_threshold,
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "validation_roc_auc": val_roc_auc,
        "test_roc_auc": test_roc_auc,
        "strategies": strategies_results,
        "recommended_strategy": args.strategy,
        "recommended_threshold": chosen_threshold,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote detailed report to: {args.report_path}")

    # ---- Optionally update the saved model bundle ----
    if not args.no_update_bundle:
        backup_path = args.model_path.with_suffix(args.model_path.suffix + ".pre_retune.bak")
        if not backup_path.exists():
            shutil.copy2(args.model_path, backup_path)
            print(f"Backed up old model bundle to: {backup_path}")
        bundle["decision_threshold"] = chosen_threshold
        bundle["threshold_retune"] = {
            "performed_at": utc_now_iso(),
            "strategy": args.strategy,
            "previous_threshold": current_threshold,
            "new_threshold": chosen_threshold,
        }
        joblib.dump(bundle, args.model_path)
        print(f"Updated model bundle with new decision threshold: {chosen_threshold:.2f}")
        print("\nNext step: re-run inference to regenerate predictions.csv with the new threshold:")
        print("  uv run sdnet-infer --limit 0")
        print("Then re-run localisation to regenerate severity distributions:")
        print("  uv run sdnet-localize --limit 0")
    else:
        print("\n(--no-update-bundle was set; model file was NOT modified.)")


if __name__ == "__main__":
    main()
