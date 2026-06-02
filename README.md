# SDNET2018 Crack Detection and Quantification PoC

This repository is a local, laptop-friendly proof of concept for processing the SDNET2018 concrete crack dataset and user-uploaded inspection images.

The current workflow follows a CrackNet-style methodology:

```text
Image Capture -> Preprocessing -> Crack Detection -> Crack Segmentation -> Measurement Engine -> Severity Classification -> Explainability -> Final Inspection Output
```

The solution can:

- download or use the Kaggle SDNET2018 dataset
- build a labeled manifest with train/validation/test splits
- train a local crack classifier
- run inference across the full dataset
- estimate crack masks, contour-following polygons, heatmaps, area, length, mean width, max width, and severity
- store upload batches as separate inspection projects
- expose artifacts through FastAPI
- present dataset and project results in a React/Vite dashboard
- show a methodology Performance Radar for ResNet-50, VGG-16, and EfficientNet-B0

Important: SDNET2018 provides image-level crack labels, not pixel masks or bounding boxes. The current segmentation and measurements are therefore marked as heuristic estimates. The repository is structured so true U-Net or YOLO-style supervised stages can be added when annotations are available.

## Quick Start: Full Kaggle Processing on macOS

```bash
cd /Users/shrav/Documents/code_codex/SDNET
./scripts/bootstrap.sh
./scripts/run_fresh_data_processing.sh \
  --download-mode copy \
  --sample-size 0 \
  --model-type extra_trees \
  --threshold-metric accuracy \
  --n-estimators 500 \
  --max-depth 0 \
  --image-size 224 \
  --localization-limit 0 \
  --log-file logs/fresh_kaggle_data_processing.log
```

Start the backend:

```bash
./scripts/run_backend.sh
```

Start the frontend in another terminal:

```bash
./scripts/run_frontend.sh
```

Open:

```text
http://localhost:5173
```

## Quick Start: Full Kaggle Processing on Windows

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd C:\Users\<your-user>\Documents\code_codex\SDNET
.\scripts\bootstrap.ps1
.\scripts\run_fresh_data_processing.ps1 `
  -DownloadMode copy `
  -SampleSize 0 `
  -ModelType extra_trees `
  -ThresholdMetric accuracy `
  -NEstimators 500 `
  -MaxDepth 0 `
  -ImageSize 224 `
  -LocalizationLimit 0 `
  -LogFile logs\fresh_kaggle_data_processing.log
```

Start the backend:

```powershell
.\scripts\run_backend.ps1
```

Start the frontend in another PowerShell window:

```powershell
.\scripts\run_frontend.ps1
```

Open:

```text
http://localhost:5173
```

## Normal Pipeline Commands

Use these when data already exists under `data/raw/sdnet2018`:

```bash
uv run sdnet-build-manifest --dataset-dir data/raw/sdnet2018
uv run sdnet-train --sample-size 0 --model-type extra_trees --threshold-metric accuracy --image-size 224
uv run sdnet-infer --limit 0
uv run sdnet-localize --limit 0
uv run sdnet-methodology-summary
```

## Backend API

Default API URL:

```text
http://localhost:8000
```

Key endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | API health check |
| `GET /api/status` | Artifact status |
| `GET /api/summary` | Summary, metrics, manifest, methodology, and artifact status |
| `GET /api/methodology` | CrackNet methodology stages, architecture status, and radar metadata |
| `GET /api/predictions` | Dataset predictions with filters |
| `GET /api/predictions/{image_id}/image` | Source image |
| `GET /api/predictions/{image_id}/overlay` | Crack polygon overlay |
| `GET /api/predictions/{image_id}/heatmap` | Crack likelihood heatmap |
| `GET /api/predictions/{image_id}/mask` | Estimated crack mask |
| `POST /api/projects` | Upload one or more images as a separate inspection project |
| `GET /api/projects/{project_id}` | Project predictions and measurements |

## Frontend Pages

The React app provides:

- dataset results dashboard
- artifact status strip
- class distribution and confusion matrix
- CrackNet methodology stage panel
- Performance Radar for ResNet-50, VGG-16, and EfficientNet-B0
- predictions table with area, length, mean/max width, severity, and confidence
- original, overlay, heatmap, and mask preview tabs
- upload project page for one or more inspection images

## Main Outputs

| Artifact | Description |
| --- | --- |
| `data/processed/manifest.csv` | Image paths, labels, surfaces, dimensions, and splits |
| `data/models/crack_classifier.joblib` | Trained classifier bundle |
| `data/results/metrics.json` | Training/evaluation metrics |
| `data/results/predictions.csv` | Per-image crack probabilities and labels |
| `data/results/localizations.csv` | Crack polygons and geometry measurements |
| `data/results/localization/overlays/` | Polygon overlay images |
| `data/results/localization/heatmaps/` | Crack likelihood heatmaps |
| `data/results/localization/masks/` | Estimated binary crack masks |
| `data/results/summary.json` | Aggregate dataset summary |
| `data/results/methodology_summary.json` | Methodology stages, architecture status, and radar metadata |
| `data/projects/<project_id>/` | Uploaded project images and generated artifacts |

## Documentation

- `docs/BUSINESS_PROJECT_OVERVIEW.md`: business idea, methodology, results interpretation, and limitations
- `docs/TECHNICAL_SOLUTION.md`: components, API, pipeline, artifacts, and execution details
- `docs/LOCAL_ENVIRONMENT_SETUP.md`: macOS and Windows setup/run instructions
- `docs/IMPROVED_MODEL_EXECUTION_GUIDE.md`: model tuning and accuracy improvement commands
- `docs/CODEX_CRACKNET_METHODOLOGY_INSTRUCTIONS.md`: methodology instructions for future Codex work
