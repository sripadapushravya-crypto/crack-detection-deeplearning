from __future__ import annotations

import json
import math
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from sdnet_pipeline.config import (
    DEFAULT_LOCALIZATIONS,
    DEFAULT_MANIFEST,
    DEFAULT_METHODOLOGY,
    DEFAULT_METRICS,
    DEFAULT_MODEL,
    DEFAULT_PREDICTIONS,
    DEFAULT_SUMMARY,
    PROJECTS_DIR,
)
from sdnet_pipeline.features import extract_features
from sdnet_pipeline.localization import analyze_image
from sdnet_pipeline.methodology import build_methodology_payload, write_methodology_summary
from sdnet_pipeline.utils import read_json, utc_now_iso, write_json


app = FastAPI(
    title="SDNET Crack Detection API",
    description="Local API for SDNET2018 crack detection pipeline outputs.",
    version="0.1.0",
)

# Allowed browser origins. Defaults cover local dev plus the deployed Vercel
# frontend; override in production with the FRONTEND_ORIGINS env var (comma-sep).
_DEFAULT_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "https://crack-detection-deeplearning-omega.vercel.app"
)
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("FRONTEND_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

# Safety limits for the public, unauthenticated project-upload endpoint.
MAX_UPLOAD_FILES = 20
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per file

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: clean_value(value) for key, value in row.items()} for row in df.to_dict("records")]


def safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    return token or "image"


def load_predictions() -> pd.DataFrame:
    if not DEFAULT_PREDICTIONS.exists():
        raise HTTPException(
            status_code=404,
            detail="Predictions not found. Run ./scripts/run_pipeline.sh first.",
        )
    predictions = pd.read_csv(DEFAULT_PREDICTIONS)
    if DEFAULT_LOCALIZATIONS.exists():
        localizations = pd.read_csv(DEFAULT_LOCALIZATIONS)
        duplicate_columns = {
            column
            for column in localizations.columns
            if column in predictions.columns and column != "image_id"
        }
        localizations = localizations.drop(columns=sorted(duplicate_columns), errors="ignore")
        predictions = predictions.merge(localizations, on="image_id", how="left")
    return predictions


def load_localizations() -> pd.DataFrame:
    if not DEFAULT_LOCALIZATIONS.exists():
        raise HTTPException(
            status_code=404,
            detail="Localizations not found. Run uv run sdnet-localize first.",
        )
    return pd.read_csv(DEFAULT_LOCALIZATIONS)


def load_manifest() -> pd.DataFrame:
    if not DEFAULT_MANIFEST.exists():
        raise HTTPException(
            status_code=404,
            detail="Manifest not found. Run ./scripts/run_pipeline.sh first.",
        )
    return pd.read_csv(DEFAULT_MANIFEST)


def load_model_bundle() -> dict[str, Any] | None:
    """Load the Extra Trees model bundle. Returns None if not available
    (e.g. on Render where only the CNN is deployed) so the CNN path
    can proceed without raising a 404."""
    if not DEFAULT_MODEL.exists():
        return None
    try:
        return joblib.load(DEFAULT_MODEL)
    except Exception:
        return None


def classify_uploaded_image(image_path: Path, image_id: str, bundle: dict[str, Any]) -> dict[str, Any]:
    model = bundle["model"]
    image_size = int(bundle.get("feature_config", {}).get("image_size", 224))
    decision_threshold = float(bundle.get("decision_threshold", 0.5))
    labels = bundle.get("labels", {0: "non_cracked", 1: "cracked"})

    feature_row = extract_features(image_path, image_size=image_size)
    crack_probability = float(model.predict_proba([feature_row])[0][1])
    predicted_target = int(crack_probability >= decision_threshold)
    predicted_label = labels.get(predicted_target, labels.get(str(predicted_target), "unknown"))

    with Image.open(image_path) as image:
        width, height = image.size

    return {
        "image_id": image_id,
        "path": str(image_path.resolve()),
        "relative_path": image_path.name,
        "label": None,
        "target": None,
        "surface": "uploaded",
        "source_folder": "uploaded",
        "width": int(width),
        "height": int(height),
        "predicted_target": predicted_target,
        "predicted_label": predicted_label,
        "crack_probability": crack_probability,
        "confidence": crack_probability if predicted_target == 1 else 1.0 - crack_probability,
    }


# ---------------------------------------------------------------------------
# Deep-learning (CNN) classifier.
#
# If a fine-tuned CNN is present under models/cnn/ (produced by
# scripts/train_cnn.py), the upload endpoint uses it in preference to the
# classical ExtraTrees model. torch is imported lazily so the API still starts
# where torch is not installed (it then falls back to the classical model). Run
# the backend with `uv run --active ...` so the venv that has torch is used.
# ---------------------------------------------------------------------------
CNN_DIR = Path(os.environ.get("CNN_MODEL_DIR", "models/cnn"))
_CNN_CACHE: dict[str, Any] | None = None


def _build_cnn_architecture(tv_models: Any, backbone: str, num_classes: int) -> Any:
    import torch.nn as nn

    if backbone == "resnet18":
        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet50":
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "mobilenet_v2":
        model = tv_models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported CNN backbone: {backbone}")
    return model


def load_cnn_classifier() -> dict[str, Any] | None:
    """Load the fine-tuned CNN once, or return None if unavailable.

    Returns None (so the caller falls back to the classical model) when the CNN
    artifacts are missing or torch/torchvision are not installed. Every failure
    path logs the real exception so Render logs show the actual cause instead of
    a silent None.
    """
    global _CNN_CACHE
    if _CNN_CACHE is not None:
        return _CNN_CACHE

    meta_path = CNN_DIR / "model_meta.json"
    print(f"[cnn] CNN_DIR={CNN_DIR.resolve()}", flush=True)
    print(f"[cnn] meta_path={meta_path.resolve()} exists={meta_path.exists()}", flush=True)
    if not meta_path.exists():
        print("[cnn] model_meta.json not found -> returning None", flush=True)
        return None
    try:
        import torch
        from torchvision import models as tv_models
        from torchvision import transforms as tv_transforms
        print(f"[cnn] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    except Exception as exc:
        print(f"[cnn] IMPORT FAILED: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return None

    try:
        meta = json.loads(meta_path.read_text())
        backbone = str(meta.get("backbone", "resnet18"))
        num_classes = int(meta.get("num_classes", 2))
        img_size = int(meta.get("img_size", 224))
        norm = meta.get("normalization", {})
        mean = norm.get("mean", [0.485, 0.456, 0.406])
        std = norm.get("std", [0.229, 0.224, 0.225])
        raw_labels = meta.get("class_to_label", {0: "non_cracked", 1: "cracked"})
        class_to_label = {int(k): str(v) for k, v in raw_labels.items()}
        threshold = float(meta.get("decision_threshold", 0.5))
        checkpoint = CNN_DIR / str(meta.get("checkpoint", "best_model.pt"))
        print(f"[cnn] checkpoint={checkpoint.resolve()} exists={checkpoint.exists()}", flush=True)
        if not checkpoint.exists():
            print("[cnn] checkpoint file not found -> returning None", flush=True)
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_cnn_architecture(tv_models, backbone, num_classes)
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        print("[cnn] model loaded and ready", flush=True)
    except Exception as exc:
        print(f"[cnn] BUILD/LOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return None

    transform = tv_transforms.Compose(
        [
            tv_transforms.Resize((img_size, img_size)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean, std),
        ]
    )

    _CNN_CACHE = {
        "torch": torch,
        "model": model,
        "device": device,
        "transform": transform,
        "class_to_label": class_to_label,
        "decision_threshold": threshold,
        "backbone": backbone,
    }
    return _CNN_CACHE


def classify_uploaded_image_cnn(
    image_path: Path, image_id: str, cnn: dict[str, Any]
) -> dict[str, Any]:
    torch = cnn["torch"]
    model = cnn["model"]
    device = cnn["device"]
    transform = cnn["transform"]
    class_to_label = cnn["class_to_label"]
    threshold = cnn["decision_threshold"]

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        crack_probability = float(torch.softmax(logits, dim=1)[0, 1].item())

    predicted_target = int(crack_probability >= threshold)
    predicted_label = class_to_label.get(predicted_target, "unknown")

    return {
        "image_id": image_id,
        "path": str(image_path.resolve()),
        "relative_path": image_path.name,
        "label": None,
        "target": None,
        "surface": "uploaded",
        "source_folder": "uploaded",
        "width": int(width),
        "height": int(height),
        "predicted_target": predicted_target,
        "predicted_label": predicted_label,
        "crack_probability": crack_probability,
        "confidence": crack_probability if predicted_target == 1 else 1.0 - crack_probability,
        "classifier_type": cnn["backbone"],
    }


def project_path(project_id: str) -> Path:
    path = PROJECTS_DIR / project_id
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown project_id: {project_id}")
    return path


def project_json_path(project_id: str) -> Path:
    return project_path(project_id) / "project.json"


def read_project(project_id: str) -> dict[str, Any]:
    payload = read_json(project_json_path(project_id))
    if not payload:
        raise HTTPException(status_code=404, detail=f"Project metadata not found: {project_id}")
    return payload


def summarize_project(records: list[dict[str, Any]]) -> dict[str, Any]:
    cracked = [record for record in records if record.get("predicted_label") == "cracked"]
    non_cracked = [record for record in records if record.get("predicted_label") == "non_cracked"]
    localized = [record for record in cracked if record.get("overlay_path")]
    severity: dict[str, int] = {}
    for record in localized:
        label = str(record.get("severity_label") or "unknown")
        severity[label] = severity.get(label, 0) + 1
    return {
        "image_count": len(records),
        "predicted_cracked": len(cracked),
        "predicted_non_cracked": len(non_cracked),
        "localized_cracks": len(localized),
        "severity_labels": severity,
        "average_crack_probability": (
            sum(float(record.get("crack_probability") or 0.0) for record in records) / len(records)
            if records
            else 0.0
        ),
        "total_crack_area_px": int(sum(int(record.get("crack_area_px") or 0) for record in localized)),
        "total_crack_length_px": float(sum(float(record.get("crack_length_px") or 0.0) for record in localized)),
        "average_mean_width_px": (
            sum(float(record.get("mean_width_px") or 0.0) for record in localized) / len(localized)
            if localized
            else 0.0
        ),
        "average_max_width_px": (
            sum(float(record.get("max_width_px") or 0.0) for record in localized) / len(localized)
            if localized
            else 0.0
        ),
        "segmentation_source": "heuristic_clahe_frangi_morphology",
        "measurement_method": "mask_skeleton_distance_transform",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    projects: list[dict[str, Any]] = []
    for path in sorted(PROJECTS_DIR.iterdir(), reverse=True):
        metadata_path = path / "project.json"
        if metadata_path.exists():
            payload = read_json(metadata_path)
            projects.append(
                {
                    "project_id": payload.get("project_id"),
                    "name": payload.get("name"),
                    "created_at": payload.get("created_at"),
                    "summary": payload.get("summary", {}),
                }
            )
    return {"projects": projects}


def build_scale_config(
    scale_source: str,
    scale_mm_per_px: float | None,
    marker_length_mm: float | None,
    aruco_dict: str,
    distance_mm: float | None,
    focal_length_mm: float | None,
    sensor_width_mm: float | None,
) -> dict[str, Any] | None:
    if scale_source == "manual" and scale_mm_per_px:
        return {"source": "manual", "scale_mm_per_px": float(scale_mm_per_px)}
    if scale_source == "aruco" and marker_length_mm:
        return {
            "source": "aruco",
            "marker_length_mm": float(marker_length_mm),
            "aruco_dict": aruco_dict or "DICT_4X4_50",
        }
    if scale_source == "geometry" and distance_mm and focal_length_mm and sensor_width_mm:
        return {
            "source": "geometry",
            "distance_mm": float(distance_mm),
            "focal_length_mm": float(focal_length_mm),
            "sensor_width_mm": float(sensor_width_mm),
        }
    return None


@app.post("/api/projects")
async def create_project(
    name: str = Form("Concrete Inspection Project"),
    files: list[UploadFile] = File(...),
    scale_source: str = Form("none"),
    scale_mm_per_px: float | None = Form(None),
    marker_length_mm: float | None = Form(None),
    aruco_dict: str = Form("DICT_4X4_50"),
    distance_mm: float | None = Form(None),
    focal_length_mm: float | None = Form(None),
    sensor_width_mm: float | None = Form(None),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(files)} (max {MAX_UPLOAD_FILES} per project).",
        )

    cnn = load_cnn_classifier()
    bundle = None if cnn is not None else load_model_bundle()
    print(f"[upload] cnn_loaded={cnn is not None} bundle_loaded={bundle is not None}", flush=True)

    if cnn is None and bundle is None:
        print("[upload] 503 - no classifier available, see [cnn] logs above for the real cause", flush=True)
        raise HTTPException(
            status_code=503,
            detail="No classifier model available. Deploy the CNN weights or run the data pipeline first.",
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    project_id = f"project_{timestamp}_{uuid.uuid4().hex[:8]}"
    base_dir = PROJECTS_DIR / project_id
    uploads_dir = base_dir / "uploads"
    results_dir = base_dir / "results"
    localization_dir = results_dir / "localization"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    localization_dir.mkdir(parents=True, exist_ok=True)

    scale_config = build_scale_config(
        scale_source, scale_mm_per_px, marker_length_mm,
        aruco_dict, distance_mm, focal_length_mm, sensor_width_mm,
    )

    records: list[dict[str, Any]] = []
    localization_records: list[dict[str, Any]] = []
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for index, upload in enumerate(files, start=1):
        original_name = upload.filename or f"upload_{index}.jpg"
        suffix = Path(original_name).suffix.lower()
        if suffix not in allowed_suffixes:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {original_name}")

        image_id = f"{project_id}_{index:04d}"
        destination = uploads_dir / f"{image_id}_{safe_token(Path(original_name).stem)}{suffix}"
        written = 0
        with destination.open("wb") as handle:
            while chunk := upload.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"{original_name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per-file limit.",
                    )
                handle.write(chunk)

        try:
            if cnn is not None:
                record = classify_uploaded_image_cnn(destination, image_id=image_id, cnn=cnn)
            else:
                record = classify_uploaded_image(destination, image_id=image_id, bundle=bundle)
            record["original_filename"] = original_name
            # Always run the geometry localiser (CLAHE + Frangi ridge + skeleton/
            # distance transform), regardless of the classifier's verdict. The
            # SDNET2018-trained classifier does not generalise to arbitrary field
            # photos, but the ridge-based geometry does -- so this surfaces the
            # heatmap and crack measurements for any uploaded image.
            localization = analyze_image(
                pd.Series(record),
                output_dir=localization_dir,
                min_object_size=64,
                max_components=12,
                max_polygon_points=160,
                min_component_length=18,
                min_elongation=1.8,
                scale_config=scale_config,
            )
            record.update(localization)
            localization_records.append(localization)
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "image_id": image_id,
                    "path": str(destination.resolve()),
                    "relative_path": destination.name,
                    "original_filename": original_name,
                    "error": str(exc),
                }
            )

    predictions_path = results_dir / "predictions.csv"
    localizations_path = results_dir / "localizations.csv"
    pd.DataFrame(records).to_csv(predictions_path, index=False)
    pd.DataFrame(localization_records).to_csv(localizations_path, index=False)

    project = {
        "project_id": project_id,
        "name": name,
        "created_at": utc_now_iso(),
        "project_dir": str(base_dir.resolve()),
        "uploads_dir": str(uploads_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "localizations_path": str(localizations_path.resolve()),
        "summary": summarize_project(records),
        "records": records,
    }
    write_json(base_dir / "project.json", project)
    return project


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return read_project(project_id)


@app.get("/api/projects/{project_id}/images/{image_id}/{artifact}")
def project_image_artifact(project_id: str, image_id: str, artifact: str) -> FileResponse:
    if artifact not in {"original", "overlay", "heatmap", "mask"}:
        raise HTTPException(status_code=404, detail=f"Unsupported artifact: {artifact}")

    project = read_project(project_id)
    records = project.get("records", [])
    match = next((record for record in records if record.get("image_id") == image_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")

    path_key = "path" if artifact == "original" else f"{artifact}_path"
    artifact_path = match.get(path_key)
    if not artifact_path:
        raise HTTPException(status_code=404, detail=f"{artifact} not available for image_id: {image_id}")

    path = Path(str(artifact_path))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{artifact} file no longer exists: {path}")
    return FileResponse(path)


@app.get("/api/status")
def status() -> dict[str, Any]:
    artifacts = {
        "manifest": DEFAULT_MANIFEST,
        "metrics": DEFAULT_METRICS,
        "predictions": DEFAULT_PREDICTIONS,
        "localizations": DEFAULT_LOCALIZATIONS,
        "methodology": DEFAULT_METHODOLOGY,
        "summary": DEFAULT_SUMMARY,
    }
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "modified": path.stat().st_mtime if path.exists() else None,
        }
        for name, path in artifacts.items()
    }


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    methodology = read_json(DEFAULT_METHODOLOGY) or build_methodology_payload()
    output = {
        "summary": read_json(DEFAULT_SUMMARY),
        "metrics": read_json(DEFAULT_METRICS),
        "manifest": read_json(DEFAULT_MANIFEST.with_suffix(".summary.json")),
        "methodology": methodology,
        "status": status(),
    }
    return output


@app.get("/api/methodology")
def methodology() -> dict[str, Any]:
    payload = read_json(DEFAULT_METHODOLOGY)
    if payload:
        return payload
    return write_methodology_summary(DEFAULT_METHODOLOGY)


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    payload = read_json(DEFAULT_METRICS)
    if not payload:
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    return payload


@app.get("/api/predictions")
def predictions(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    surface: str | None = None,
    predicted_label: str | None = None,
    actual_label: str | None = None,
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    sort_by: str = Query("confidence", pattern="^(confidence|crack_probability|image_id)$"),
    direction: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    df = load_predictions()
    if surface:
        df = df[df["surface"] == surface]
    if predicted_label:
        df = df[df["predicted_label"] == predicted_label]
    if actual_label:
        df = df[df["label"] == actual_label]
    if min_confidence is not None:
        df = df[df["confidence"] >= min_confidence]

    total = len(df)
    ascending = direction == "asc"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    page = df.iloc[offset : offset + limit].copy()
    return {
        "total": int(total),
        "offset": int(offset),
        "limit": int(limit),
        "records": clean_records(page),
    }


@app.get("/api/options")
def options() -> dict[str, list[str]]:
    df = load_predictions()
    return {
        "surfaces": sorted(value for value in df["surface"].dropna().unique().tolist()),
        "predicted_labels": sorted(value for value in df["predicted_label"].dropna().unique().tolist()),
        "actual_labels": sorted(value for value in df["label"].dropna().unique().tolist()),
        "severity_labels": sorted(value for value in df.get("severity_label", pd.Series(dtype=str)).dropna().unique().tolist()),
    }


@app.get("/api/predictions/{image_id}/image")
def prediction_image(image_id: str) -> FileResponse:
    df = load_predictions()
    match = df[df["image_id"] == image_id]
    if match.empty:
        manifest = load_manifest()
        match = manifest[manifest["image_id"] == image_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown image_id: {image_id}")

    path = Path(str(match.iloc[0]["path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image file no longer exists: {path}")
    return FileResponse(path)


@app.get("/api/predictions/{image_id}/localization")
def prediction_localization(image_id: str) -> dict[str, Any]:
    df = load_localizations()
    match = df[df["image_id"] == image_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No localization found for image_id: {image_id}")
    record = clean_records(match.head(1))[0]
    polygons_json = record.get("polygons_json")
    if polygons_json:
        try:
            record["polygons"] = json.loads(polygons_json)
        except Exception:
            record["polygons"] = []
    return record


@app.get("/api/predictions/{image_id}/{artifact}")
def prediction_artifact(image_id: str, artifact: str) -> FileResponse:
    if artifact not in {"overlay", "heatmap", "mask"}:
        raise HTTPException(status_code=404, detail=f"Unsupported artifact: {artifact}")

    df = load_localizations()
    match = df[df["image_id"] == image_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"No localization found for image_id: {image_id}")

    path_column = f"{artifact}_path"
    path = Path(str(match.iloc[0].get(path_column, "")))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{artifact} file no longer exists: {path}")
    return FileResponse(path)

@app.get("/")
def root():
    return {"message": "SDNET API is running"}