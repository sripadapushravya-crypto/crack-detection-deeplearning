from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sdnet_pipeline.config import DEFAULT_METHODOLOGY, DEFAULT_SUMMARY, ensure_data_dirs
from sdnet_pipeline.utils import read_json, utc_now_iso, write_json


METHODOLOGY_STAGES: list[dict[str, Any]] = [
    {
        "order": 1,
        "name": "Image Capture",
        "artifact": "manifest.csv",
        "status": "implemented",
        "description": "SDNET2018 and upload project image provenance with dimensions, labels, and split metadata.",
    },
    {
        "order": 2,
        "name": "Preprocessing",
        "artifact": "feature vectors and crack-likelihood maps",
        "status": "implemented",
        "description": "Resize, grayscale statistics, texture features, CLAHE, dark-ridge response, and edge enhancement.",
    },
    {
        "order": 3,
        "name": "Crack Detection",
        "artifact": "predictions.csv",
        "status": "implemented",
        "description": "Binary crack probability and confidence for every image.",
    },
    {
        "order": 4,
        "name": "Crack Segmentation",
        "artifact": "localization/masks/*.png",
        "status": "heuristic_fallback",
        "description": "Estimated pixel masks from morphology because SDNET2018 does not provide supervised masks.",
    },
    {
        "order": 5,
        "name": "Measurement Engine",
        "artifact": "localizations.csv",
        "status": "implemented",
        "description": "Area, area percent, skeleton length, mean width, max width, polygons, and component counts.",
    },
    {
        "order": 6,
        "name": "Severity Classification",
        "artifact": "severity_label and severity_score",
        "status": "implemented",
        "description": "Pixel-estimated severity by default, with a calibration hook for mm-per-pixel thresholds.",
    },
    {
        "order": 7,
        "name": "Explainability",
        "artifact": "localization/overlays and heatmaps",
        "status": "implemented",
        "description": "Polygon overlay, mask, and heatmap views for dataset and upload project inspection.",
    },
    {
        "order": 8,
        "name": "Final Inspection Output",
        "artifact": "summary.json, project.json, API responses",
        "status": "implemented",
        "description": "Dataset dashboard and per-project reports returned through FastAPI and React.",
    },
]


MODEL_ARCHITECTURES: list[dict[str, Any]] = [
    {
        "name": "EfficientNet-B4",
        "role": "classification",
        "input_size": "224x224x3",
        "recommended_loss": "weighted binary cross entropy",
        "implementation_status": "architecture_ready",
        "current_fallback": "local laptop classifier from handcrafted image features",
    },
    {
        "name": "YOLOv8",
        "role": "candidate crack detection",
        "input_size": "640x640x3",
        "recommended_loss": "YOLO detection loss",
        "implementation_status": "annotation_required",
        "current_fallback": "pseudo-boxes can be derived from heuristic masks when needed",
    },
    {
        "name": "U-Net",
        "role": "pixel segmentation",
        "input_size": "256x256x3 or 512x512x3",
        "recommended_loss": "Dice loss + binary cross entropy",
        "implementation_status": "mask_labels_required",
        "current_fallback": "heuristic crack segmentation marked as estimated",
    },
]


PERFORMANCE_RADAR: list[dict[str, Any]] = [
    {
        "model": "ResNet-50",
        "metrics": {
            "accuracy": 87,
            "precision": 78,
            "recall": 88,
            "roc_auc": 91,
            "pr_auc": 85,
            "speed": 72,
        },
    },
    {
        "model": "VGG-16",
        "metrics": {
            "accuracy": 84,
            "precision": 75,
            "recall": 82,
            "roc_auc": 88,
            "pr_auc": 80,
            "speed": 48,
        },
    },
    {
        "model": "EfficientNet-B0",
        "metrics": {
            "accuracy": 89,
            "precision": 82,
            "recall": 87,
            "roc_auc": 93,
            "pr_auc": 88,
            "speed": 86,
        },
    },
]


def build_methodology_payload(summary_path: Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    pipeline_summary = read_json(summary_path)
    localization = pipeline_summary.get("localization", {}) if isinstance(pipeline_summary, dict) else {}
    return {
        "created_at": utc_now_iso(),
        "source_document": "docs/CODEX_CRACKNET_METHODOLOGY_INSTRUCTIONS.md",
        "measurement_mode": "heuristic_estimated_without_pixel_masks",
        "segmentation_source": "heuristic_clahe_frangi_morphology",
        "severity_basis": "pixel_estimate_unless_scale_mm_per_px_is_supplied",
        "stages": METHODOLOGY_STAGES,
        "architectures": MODEL_ARCHITECTURES,
        "performance_radar": {
            "basis": "Illustrative reference baselines from the SDNET2018-adjacent literature for ResNet-50, VGG-16, and EfficientNet-B0; not trained or measured in this work and not directly comparable to the present results.",
            "scale": "0-100 normalized score",
            "metrics": ["accuracy", "precision", "recall", "roc_auc", "pr_auc", "speed"],
            "models": PERFORMANCE_RADAR,
        },
        "current_outputs": {
            "localized_images": int(localization.get("rows", 0) or 0),
            "average_area_pct": localization.get("average_area_pct"),
            "average_length_px": localization.get("average_length_px"),
            "average_mean_width_px": localization.get("average_mean_width_px"),
            "average_max_width_px": localization.get("average_max_width_px"),
            "average_severity_score": localization.get("average_severity_score"),
        },
    }


def write_methodology_summary(output_path: Path = DEFAULT_METHODOLOGY) -> dict[str, Any]:
    ensure_data_dirs()
    payload = build_methodology_payload()
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write CrackNet-style methodology summary artifact.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_METHODOLOGY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_methodology_summary(args.output_path)
    print(f"Wrote methodology summary to {args.output_path}")
    print(f"Stages: {len(payload['stages'])}, radar models: {len(payload['performance_radar']['models'])}")


if __name__ == "__main__":
    main()
