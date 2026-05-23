from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SDNET_DATA_DIR", REPO_ROOT / "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
PROJECTS_DIR = DATA_DIR / "projects"

DEFAULT_DATASET_DIR = RAW_DIR / "sdnet2018"
DEFAULT_MANIFEST = PROCESSED_DIR / "manifest.csv"
DEFAULT_MODEL = MODELS_DIR / "crack_classifier.joblib"
DEFAULT_METRICS = RESULTS_DIR / "metrics.json"
DEFAULT_PREDICTIONS = RESULTS_DIR / "predictions.csv"
DEFAULT_SUMMARY = RESULTS_DIR / "summary.json"
DEFAULT_LOCALIZATIONS = RESULTS_DIR / "localizations.csv"
DEFAULT_METHODOLOGY = RESULTS_DIR / "methodology_summary.json"
LOCALIZATION_DIR = RESULTS_DIR / "localization"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_data_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, PROJECTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)