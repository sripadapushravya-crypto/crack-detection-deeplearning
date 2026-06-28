"""
Reference: passing a calibration scale through the web upload flow.

This shows the ONLY backend change needed to make the dashboard return
millimetres: read the optional scale fields from the upload request, build a
``scale_config``, and hand it to ``analyze_image()``. You do not need a new
route — splice the three marked blocks into your existing upload/localization
endpoint. Everything else (saving the file, running the ResNet-18 detector,
choosing the project output dir) is your current logic, shown here as
placeholder calls: save_upload(), run_detector(), PROJECT_OUTPUT_DIR.
"""
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile

# Adjust this import to your module path. analyze_image() is the function in the
# updated localization.py that now accepts scale_config=...
from localization import analyze_image

router = APIRouter()


# ---------------------------------------------------------------------------
# Block 1 — HTTP-form equivalent of build_scale_config() from localization.py
# ---------------------------------------------------------------------------
def build_scale_config_from_request(
    scale_source: str = "none",
    scale_mm_per_px: Optional[float] = None,
    marker_length_mm: Optional[float] = None,
    aruco_dict: str = "DICT_4X4_50",
    distance_mm: Optional[float] = None,
    focal_length_mm: Optional[float] = None,
    sensor_width_mm: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Return a scale_config dict, or None to keep measurements pixel-domain."""
    if scale_source == "manual" and scale_mm_per_px:
        return {"source": "manual", "scale_mm_per_px": scale_mm_per_px}
    if scale_source == "aruco" and marker_length_mm:
        return {
            "source": "aruco",
            "marker_length_mm": marker_length_mm,
            "aruco_dict": aruco_dict,
        }
    if scale_source == "geometry" and distance_mm and focal_length_mm and sensor_width_mm:
        return {
            "source": "geometry",
            "distance_mm": distance_mm,
            "focal_length_mm": focal_length_mm,
            "sensor_width_mm": sensor_width_mm,
        }
    return None


@router.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    # -----------------------------------------------------------------------
    # Block 2 — optional scale fields. All default to "off", so an upload with
    # no scale behaves exactly as it does today (pixel-domain output).
    # -----------------------------------------------------------------------
    scale_source: str = Form("none"),          # none | manual | aruco | geometry
    scale_mm_per_px: Optional[float] = Form(None),
    marker_length_mm: Optional[float] = Form(None),
    aruco_dict: str = Form("DICT_4X4_50"),
    distance_mm: Optional[float] = Form(None),
    focal_length_mm: Optional[float] = Form(None),
    sensor_width_mm: Optional[float] = Form(None),
):
    # --- your existing logic: persist the upload and run the detector ---
    saved_path = save_upload(file)                       # <- your helper
    image_id = Path(saved_path).stem
    predicted_label, probability = run_detector(saved_path)  # <- your ResNet-18 call

    # -----------------------------------------------------------------------
    # Block 3 — build the scale_config and pass it to analyze_image(). The
    # `scale_config=...` argument is the entire backend change; without it the
    # localizer cannot produce millimetres no matter what the image contains.
    # -----------------------------------------------------------------------
    scale_config = build_scale_config_from_request(
        scale_source=scale_source,
        scale_mm_per_px=scale_mm_per_px,
        marker_length_mm=marker_length_mm,
        aruco_dict=aruco_dict,
        distance_mm=distance_mm,
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
    )

    record = None
    if predicted_label == "cracked":
        row = pd.Series(
            {
                "path": str(saved_path),
                "image_id": image_id,
                "predicted_label": predicted_label,
                "crack_probability": probability,
            }
        )
        record = analyze_image(
            row,
            output_dir=PROJECT_OUTPUT_DIR,     # <- your existing output dir
            min_object_size=64,
            max_components=12,
            max_polygon_points=120,
            min_component_length=18,
            min_elongation=1.8,
            scale_config=scale_config,         # <-- unlocks the *_mm fields
        )

    # `record` already contains scale_mm_per_px, scale_source, *_mm, severity_basis
    return {
        "prediction": predicted_label,
        "confidence": probability,
        "localization": record,
    }
