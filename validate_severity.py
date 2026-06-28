"""
Severity / width measurement validation for M2.

For each ruler-calibrated photo, this runs the REAL pipeline measurement engine
(sdnet_pipeline.localization.analyze_image) with that image's mm-per-pixel scale,
and prints the pipeline's estimated max and mean crack width in mm.

You supply, per image:
  - the image file path
  - the scale (mm per pixel), measured from the ruler in that photo
  - (separately, by hand) the ground-truth crack width in mm

USAGE (edit the IMAGES list below, then run):
  python validate_severity.py
"""
from pathlib import Path
import pandas as pd
from sdnet_pipeline.localization import analyze_image

# ---------------------------------------------------------------------------
# EDIT THIS LIST: one entry per ruler photo.
#   path  : full path to the image
#   scale : mm per pixel for THIS image (known ruler mm / its pixel span)
#   gt_mm : your hand-measured ground-truth crack max width in mm
# ---------------------------------------------------------------------------
IMAGES = [
    # {"path": r"C:\Users\shrav\Downloads\ruler_photos\img1.jpg", "scale": 0.125, "gt_mm": 2.0},
    # {"path": r"C:\Users\shrav\Downloads\ruler_photos\img2.jpg", "scale": 0.090, "gt_mm": 0.8},
    # ... add 8-12 entries ...
]

OUT_DIR = Path("data/validation_severity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for i, item in enumerate(IMAGES, start=1):
    row = pd.Series({
        "image_id": f"val_{i:02d}",
        "path": item["path"],
        "predicted_label": "cracked",
        "crack_probability": 1.0,
    })
    result = analyze_image(
        row,
        output_dir=OUT_DIR,
        min_object_size=64,
        max_components=12,
        max_polygon_points=160,
        min_component_length=18,
        min_elongation=1.8,
        scale_mm_per_px=item["scale"],
    )
    pipe_max_mm = result.get("max_width_mm")
    pipe_mean_mm = result.get("mean_width_mm")
    gt = item["gt_mm"]
    err = abs(pipe_max_mm - gt) if pipe_max_mm is not None else None
    rows.append({
        "image": f"val_{i:02d}",
        "scale_mm_per_px": item["scale"],
        "ground_truth_mm": gt,
        "pipeline_max_mm": round(pipe_max_mm, 3) if pipe_max_mm is not None else None,
        "pipeline_mean_mm": round(pipe_mean_mm, 3) if pipe_mean_mm is not None else None,
        "abs_error_mm": round(err, 3) if err is not None else None,
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
if len(df) and df["abs_error_mm"].notna().any():
    mae = df["abs_error_mm"].mean()
    print(f"\nMAE (max-width, mm): {mae:.3f} over {df['abs_error_mm'].notna().sum()} images")
df.to_csv(OUT_DIR / "validation_results.csv", index=False)
print(f"Saved to {OUT_DIR / 'validation_results.csv'}")