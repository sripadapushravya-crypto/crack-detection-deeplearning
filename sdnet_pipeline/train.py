from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sdnet_pipeline.config import DEFAULT_MANIFEST, DEFAULT_METRICS, DEFAULT_MODEL, ensure_data_dirs
from sdnet_pipeline.features import extract_features
from sdnet_pipeline.utils import utc_now_iso, write_json


THRESHOLD_METRICS = {"accuracy", "balanced_accuracy", "f1", "precision", "recall"}


def sample_frame(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df
    stratify_cols = (
        df["split"].astype(str)
        + "_"
        + df["surface"].astype(str)
        + "_"
        + df["target"].astype(int).astype(str)
    )
    if stratify_cols.value_counts().min() < 2 or sample_size < stratify_cols.nunique():
        return df.sample(n=sample_size, random_state=seed)
    _, sampled = train_test_split(
        df,
        test_size=sample_size,
        random_state=seed,
        stratify=stratify_cols,
    )
    return sampled.sample(frac=1, random_state=seed)


def extract_matrix(df: pd.DataFrame, image_size: int, label: str) -> np.ndarray:
    features = []
    for path in tqdm(df["path"].tolist(), desc=f"Extracting {label} features"):
        features.append(extract_features(path, image_size=image_size))
    return np.vstack(features)


def create_model(
    model_type: str,
    seed: int,
    n_estimators: int,
    max_depth: int | None,
) -> object:
    """
    Build a scikit-learn classifier.

    ExtraTrees is the recommended choice. Key settings:
    - class_weight="balanced"   — penalises missed cracks proportional to class imbalance
    - min_samples_split=5       — prevents overfitting on noisy crack-like texture
    - max_features="sqrt"       — de-correlates trees, improves generalisation
    - min_samples_leaf=2        — minimum leaf size for stability

    SGDClassifier always uses class_weight="balanced" via log-loss SVM.
    RandomForest uses balanced_subsample to handle within-bag imbalance.
    """
    if model_type == "sgd":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        max_iter=1200,
                        tol=1e-3,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            min_samples_split=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            min_samples_leaf=2,
            min_samples_split=5,      # prevents overfit on texture noise
            max_features="sqrt",      # de-correlates trees, improves generalisation
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model_type: {model_type!r}")


def threshold_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def threshold_metric_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"Unsupported threshold metric: {metric!r}")


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str,
    min_recall: float,
) -> dict[str, object]:
    """
    Grid search over [0.05, 0.95] to find the decision threshold that maximises
    the chosen metric subject to a minimum recall constraint.

    Tiebreaker: among equal-metric candidates prefer LOWER threshold (higher recall).
    This is the correct engineering priority for structural inspection — missing a
    crack (false negative) is more costly than a false alarm.

    For structural inspection the recommended settings are:
        metric      = "balanced_accuracy"
        min_recall  = 0.70
    """
    candidates = np.round(np.linspace(0.05, 0.95, 91), 2)
    rows: list[dict[str, float]] = []

    for threshold in candidates:
        y_pred = threshold_predictions(scores, threshold)
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": recall,
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    feasible = [row for row in rows if row["recall"] >= min_recall]
    if not feasible:
        # Relax to best-recall candidates if the floor is unachievable
        max_recall = max(row["recall"] for row in rows)
        feasible = [row for row in rows if row["recall"] >= max_recall * 0.95]

    # Tiebreaker prefers LOWER threshold (higher recall) — correct for inspection
    best = max(feasible, key=lambda row: (row[metric], row["recall"], -row["threshold"]))

    return {
        "metric": metric,
        "min_recall": float(min_recall),
        "selected_threshold": float(best["threshold"]),
        "selected_metrics": best,
        "candidates": rows,
    }


def split_metrics(
    model: object,
    df: pd.DataFrame,
    x: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    y_true = df["target"].astype(int).to_numpy()
    y_score = model.predict_proba(x)[:, 1]
    y_pred = threshold_predictions(y_score, threshold)

    metrics: dict[str, object] = {
        "rows": int(len(df)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["non_cracked", "cracked"],
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def train_model(
    manifest_path: Path,
    model_path: Path,
    metrics_path: Path,
    sample_size: int,
    image_size: int,
    seed: int,
    model_type: str,
    threshold_metric: str,
    min_recall: float,
    n_estimators: int,
    max_depth: int | None,
) -> dict[str, object]:
    ensure_data_dirs()

    if threshold_metric not in THRESHOLD_METRICS:
        raise ValueError(f"--threshold-metric must be one of: {sorted(THRESHOLD_METRICS)}")

    df = pd.read_csv(manifest_path)
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    if df["target"].nunique() < 2:
        raise RuntimeError("Training requires both cracked and non-cracked images.")

    # Log class distribution so imbalance is visible in the run output
    counts = df["target"].value_counts()
    total = len(df)
    crack_pct = 100.0 * counts.get(1, 0) / max(total, 1)
    print(
        f"Dataset: {total:,} images | "
        f"cracked={counts.get(1, 0):,} ({crack_pct:.1f}%) | "
        f"non_cracked={counts.get(0, 0):,} ({100 - crack_pct:.1f}%)"
    )
    print(
        f"Threshold tuning: metric={threshold_metric}, min_recall={min_recall} "
        f"(recommended for inspection: balanced_accuracy + min_recall=0.70)"
    )

    df = sample_frame(df, sample_size=sample_size, seed=seed)
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        train_df = df.sample(frac=0.70, random_state=seed)

    eval_frames = {
        split: split_df.copy()
        for split, split_df in df.groupby("split")
        if split in {"train", "validation", "test"} and not split_df.empty
    }

    x_train = extract_matrix(train_df, image_size=image_size, label="train")
    y_train = train_df["target"].astype(int).to_numpy()

    model = create_model(
        model_type=model_type,
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    model.fit(x_train, y_train)

    validation_df = eval_frames.get("validation")
    if validation_df is not None and not validation_df.empty:
        x_validation = extract_matrix(validation_df, image_size=image_size, label="validation")
        threshold_source = "validation"
        threshold_y = validation_df["target"].astype(int).to_numpy()
        threshold_scores = model.predict_proba(x_validation)[:, 1]
    else:
        x_validation = None
        threshold_source = "train"
        threshold_y = y_train
        threshold_scores = model.predict_proba(x_train)[:, 1]

    threshold_report = tune_threshold(
        threshold_y,
        threshold_scores,
        metric=threshold_metric,
        min_recall=min_recall,
    )
    decision_threshold = float(threshold_report["selected_threshold"])

    metrics: dict[str, object] = {
        "created_at": utc_now_iso(),
        "manifest": str(manifest_path.resolve()),
        "model_path": str(model_path.resolve()),
        "model_type": model_type,
        "image_size": image_size,
        "sample_size": int(sample_size),
        "trained_rows": int(len(train_df)),
        "total_rows_used": int(len(df)),
        "threshold_source": threshold_source,
        "threshold_tuning": threshold_report,
        "decision_threshold": decision_threshold,
        "splits": {},
    }

    for split, split_df in eval_frames.items():
        if split == "train" and split_df.index.equals(train_df.index):
            x_split = x_train
        elif (
            split == "validation"
            and x_validation is not None
            and split_df.index.equals(validation_df.index)
        ):
            x_split = x_validation
        else:
            x_split = extract_matrix(split_df, image_size=image_size, label=split)
        metrics["splits"][split] = split_metrics(model, split_df, x_split, decision_threshold)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_config": {"image_size": image_size, "version": "hog_lbp_frangi_geo_v3"},
            "labels": {0: "non_cracked", 1: "cracked"},
            "decision_threshold": decision_threshold,
            "threshold_tuning": threshold_report,
            "model_type": model_type,
            "created_at": utc_now_iso(),
        },
        model_path,
    )
    write_json(metrics_path, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local SDNET crack classifier.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="0 means use all labeled images.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--model-type",
        choices=["sgd", "extra_trees", "random_forest"],
        default="extra_trees",
        help="extra_trees is the recommended laptop-friendly improved baseline.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=sorted(THRESHOLD_METRICS),
        default="balanced_accuracy",
        help=(
            "Metric optimised on validation scores to select the decision threshold. "
            "balanced_accuracy is recommended for structural inspection because the "
            "dataset is imbalanced (~21%% cracked) and accuracy alone suppresses recall."
        ),
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.70,
        help=(
            "Minimum cracked-class recall enforced during threshold tuning. "
            "0.70 is the recommended floor for structural inspection — "
            "missing cracks has higher engineering cost than false alarms."
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means unlimited depth.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(
        args.manifest,
        args.model_path,
        args.metrics_path,
        sample_size=args.sample_size,
        image_size=args.image_size,
        seed=args.seed,
        model_type=args.model_type,
        threshold_metric=args.threshold_metric,
        min_recall=args.min_recall,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth or None,
    )
    test_metrics = (
        metrics.get("splits", {}).get("test")
        or metrics.get("splits", {}).get("validation")
    )
    print(f"Saved model  : {args.model_path}")
    print(
        f"Threshold    : {metrics['decision_threshold']:.2f} "
        f"(metric={args.threshold_metric}, min_recall={args.min_recall}, "
        f"source={metrics['threshold_source']})"
    )
    if test_metrics:
        cm = test_metrics.get("confusion_matrix", [])
        tn = cm[0][0] if cm else "?"
        fp = cm[0][1] if cm else "?"
        fn = cm[1][0] if cm else "?"
        tp = cm[1][1] if cm else "?"
        print(
            f"Evaluation   : accuracy={test_metrics['accuracy']:.3f}, "
            f"balanced_accuracy={test_metrics['balanced_accuracy']:.3f}, "
            f"precision={test_metrics['precision']:.3f}, "
            f"recall={test_metrics['recall']:.3f}, "
            f"f1={test_metrics['f1']:.3f}"
        )
        print(f"Confusion    : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(
            "              (FN = missed cracks — keep this low for safe inspection)"
        )


if __name__ == "__main__":
    main()