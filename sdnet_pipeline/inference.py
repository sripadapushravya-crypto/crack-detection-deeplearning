from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from sdnet_pipeline.config import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_PREDICTIONS,
    DEFAULT_SUMMARY,
    ensure_data_dirs,
)
from sdnet_pipeline.features import extract_features
from sdnet_pipeline.utils import utc_now_iso, write_json


def run_inference(
    manifest_path: Path,
    model_path: Path,
    predictions_path: Path,
    summary_path: Path,
    limit: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_data_dirs()

    bundle = joblib.load(model_path)
    model = bundle["model"]
    image_size = int(bundle.get("feature_config", {}).get("image_size", 224))
    labels = bundle.get("labels", {0: "non_cracked", 1: "cracked"})
    decision_threshold = float(bundle.get("decision_threshold", 0.5))
    feature_version = bundle.get("feature_config", {}).get("version", "unknown")

    print(f"Model          : {bundle.get('model_type', 'unknown')}")
    print(f"Feature version: {feature_version}")
    print(f"Decision threshold: {decision_threshold:.3f}")

    df = pd.read_csv(manifest_path)
    if limit > 0:
        df = df.head(limit).copy()

    features = []
    for path in tqdm(df["path"].tolist(), desc="Extracting inference features"):
        features.append(extract_features(path, image_size=image_size))

    probs = model.predict_proba(features)[:, 1]
    predicted = (probs >= decision_threshold).astype(int)

    result = df.copy()
    result["predicted_target"] = predicted
    result["predicted_label"] = [labels[int(v)] for v in predicted]
    result["crack_probability"] = probs
    result["confidence"] = [
        prob if pred == 1 else 1.0 - prob
        for pred, prob in zip(predicted, probs)
    ]

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(predictions_path, index=False)

    labeled = result[result["target"].notna()].copy()
    metrics = None
    if not labeled.empty and labeled["target"].nunique() == 2:
        y_true = labeled["target"].astype(int)
        y_pred = labeled["predicted_target"].astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    summary: dict[str, object] = {
        "created_at": utc_now_iso(),
        "rows": int(len(result)),
        "model_path": str(model_path.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "decision_threshold": decision_threshold,
        "model_type": bundle.get("model_type", "unknown"),
        "feature_config": bundle.get("feature_config", {}),
        "predicted_labels": result["predicted_label"].value_counts().to_dict(),
        "actual_labels": result["label"].fillna("unknown").value_counts().to_dict(),
        "surfaces": result["surface"].fillna("unknown").value_counts().to_dict(),
        "average_crack_probability": float(result["crack_probability"].mean()),
        "metrics_on_labeled_data": metrics,
    }
    write_json(summary_path, summary)
    return result, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run crack classifier inference across manifest images."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 processes every manifest image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, summary = run_inference(
        args.manifest,
        args.model_path,
        args.predictions_path,
        args.summary_path,
        limit=args.limit,
    )
    print(f"Wrote {len(result):,} predictions to {args.predictions_path}")
    print(f"Predicted : {summary['predicted_labels']}")
    print(f"Actual    : {summary['actual_labels']}")
    if summary.get("metrics_on_labeled_data"):
        m = summary["metrics_on_labeled_data"]
        print(
            f"Metrics   : accuracy={m['accuracy']:.3f}  "
            f"balanced_accuracy={m['balanced_accuracy']:.3f}  "
            f"precision={m['precision']:.3f}  "
            f"recall={m['recall']:.3f}  "
            f"f1={m['f1']:.3f}"
        )


if __name__ == "__main__":
    main()