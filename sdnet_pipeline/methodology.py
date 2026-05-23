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
        "description": (
            "SDNET2018 and upload project image provenance with dimensions, "
            "labels, surface domain, and stratified train/validation/test splits."
        ),
    },
    {
        "order": 2,
        "name": "Preprocessing",
        "artifact": "feature vectors and crack-likelihood maps",
        "status": "implemented",
        "description": (
            "Resize, grayscale, CLAHE contrast enhancement, HOG (coarse 16x16 and "
            "fine 8x8), LBP texture, Frangi ridge filter, elongation geometry features, "
            "Sobel edges, and 11 scalar statistics."
        ),
    },
    {
        "order": 3,
        "name": "Crack Detection",
        "artifact": "predictions.csv",
        "status": "implemented",
        "description": (
            "ExtraTrees classifier with class_weight=balanced, balanced_accuracy "
            "threshold tuning, and min_recall=0.70 floor. Outputs crack probability "
            "and confidence for every image."
        ),
    },
    {
        "order": 4,
        "name": "Crack Segmentation",
        "artifact": "localization/masks/*.png",
        "status": "heuristic_fallback",
        "description": (
            "Estimated pixel masks from CLAHE + Frangi ridge morphology pipeline "
            "because SDNET2018 does not provide supervised pixel masks. "
            "Replace with U-Net++ trained on DeepCrack dataset for supervised mode."
        ),
    },
    {
        "order": 5,
        "name": "Measurement Engine",
        "artifact": "localizations.csv",
        "status": "implemented",
        "description": (
            "Skeleton-based crack length, medial-axis distance transform width, "
            "area percentage, contour polygons, and component counts. "
            "Physical mm values available when scale_mm_per_px is supplied."
        ),
    },
    {
        "order": 6,
        "name": "Severity Classification",
        "artifact": "severity_label and severity_score",
        "status": "implemented",
        "description": (
            "ACI 224R-01 width thresholds when calibrated mm values are available "
            "(hairline <0.1mm, fine 0.1-0.3mm, medium 0.3-1.0mm, wide >1.0mm). "
            "Composite pixel-estimate score as provisional fallback."
        ),
    },
    {
        "order": 7,
        "name": "Explainability",
        "artifact": "localization/overlays and heatmaps",
        "status": "implemented",
        "description": (
            "Crack contour polygon overlay, binary mask, and CLAHE/Frangi likelihood "
            "heatmap for every cracked prediction. Dataset dashboard and per-project views."
        ),
    },
    {
        "order": 8,
        "name": "Final Inspection Output",
        "artifact": "summary.json, project.json, API responses",
        "status": "implemented",
        "description": (
            "Dataset dashboard and per-project reports served through FastAPI and React. "
            "Export/report generation planned as next development milestone."
        ),
    },
]


MODEL_ARCHITECTURES: list[dict[str, Any]] = [
    {
        "name": "EfficientNet-B4",
        "role": "classification",
        "input_size": "224x224x3",
        "recommended_loss": "weighted binary cross entropy",
        "implementation_status": "architecture_ready",
        "current_fallback": (
            "ExtraTrees with HOG (coarse+fine), LBP, Frangi, elongation geometry, "
            "and edge features. Feature version: hog_lbp_frangi_geo_v3."
        ),
    },
    {
        "name": "YOLOv8-seg",
        "role": "real-time mobile detection + segmentation",
        "input_size": "640x640x3",
        "recommended_loss": "YOLO detection + segmentation loss",
        "implementation_status": "annotation_required",
        "current_fallback": "Pseudo-boxes derivable from heuristic masks when needed.",
    },
    {
        "name": "U-Net++",
        "role": "pixel-level crack segmentation",
        "input_size": "512x512x3",
        "recommended_loss": "Dice loss + binary cross entropy (50:50)",
        "implementation_status": "mask_labels_required",
        "current_fallback": (
            "Heuristic CLAHE + Frangi morphology segmentation, "
            "marked as heuristic_estimated_without_pixel_masks."
        ),
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


def build_methodology_payload(
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    pipeline_summary = read_json(summary_path)
    localization = (
        pipeline_summary.get("localization", {})
        if isinstance(pipeline_summary, dict)
        else {}
    )
    return {
        "created_at": utc_now_iso(),
        "source_document": "docs/CODEX_CRACKNET_METHODOLOGY_INSTRUCTIONS.md",
        "feature_version": "hog_lbp_frangi_geo_v3",
        "measurement_mode": "heuristic_estimated_without_pixel_masks",
        "segmentation_source": "heuristic_clahe_frangi_morphology",
        "severity_basis": "aci_224r_calibrated_mm_or_pixel_estimate_fallback",
        "threshold_default": "balanced_accuracy",
        "min_recall_default": 0.70,
        "stages": METHODOLOGY_STAGES,
        "architectures": MODEL_ARCHITECTURES,
        "performance_radar": {
            "basis": (
                "Methodology reference baseline for UI comparison. "
                "Replace with trained-model metrics when EfficientNet-B4 / "
                "U-Net++ runs are complete."
            ),
            "scale": "0-100 normalised score",
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


def write_methodology_summary(
    output_path: Path = DEFAULT_METHODOLOGY,
) -> dict[str, Any]:
    ensure_data_dirs()
    payload = build_methodology_payload()
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write CrackNet-style methodology summary artifact."
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_METHODOLOGY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_methodology_summary(args.output_path)
    print(f"Wrote methodology summary to {args.output_path}")
    print(f"Stages          : {len(payload['stages'])}")
    print(f"Radar models    : {len(payload['performance_radar']['models'])}")
    print(f"Feature version : {payload['feature_version']}")


if __name__ == "__main__":
    main()