from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage import exposure, filters, measure, morphology, util
from tqdm import tqdm

from sdnet_pipeline.config import (
    DEFAULT_LOCALIZATIONS,
    DEFAULT_PREDICTIONS,
    DEFAULT_SUMMARY,
    LOCALIZATION_DIR,
    ensure_data_dirs,
)
from sdnet_pipeline.utils import read_json, utc_now_iso, write_json


# --------------------------------------------------------------------------- #
# Colour map                                                                   #
# --------------------------------------------------------------------------- #

def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def heatmap_rgb(score: np.ndarray) -> np.ndarray:
    score = np.clip(score, 0.0, 1.0)
    stops = np.array(
        [
            [20,  30,  90],
            [30,  120, 190],
            [80,  190, 160],
            [245, 210, 75],
            [220, 65,  40],
        ],
        dtype=np.float32,
    )
    scaled = score * (len(stops) - 1)
    idx = np.floor(scaled).astype(int)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = (scaled - idx)[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    return np.clip(rgb, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Crack likelihood map                                                          #
# --------------------------------------------------------------------------- #

def crack_likelihood(gray: np.ndarray) -> np.ndarray:
    """
    Produce a per-pixel crack likelihood map from a normalised grayscale image.
    Pipeline:
      1. Gaussian smoothing to suppress sensor noise.
      2. CLAHE to enhance low-contrast crack edges.
      3. Dark-pixel response (cracks are dark on bright concrete).
      4. Frangi ridge filter for thin elongated structures.
      5. Sobel edge response.
      6. Weighted combination and final smoothing.
    """
    smoothed = filters.gaussian(gray, sigma=0.8, preserve_range=True)
    clahe = exposure.equalize_adapthist(smoothed, clip_limit=0.03)

    dark_response = normalize01(1.0 - clahe)
    edge_response = normalize01(filters.sobel(clahe))

    try:
        ridge_response = filters.frangi(dark_response, sigmas=(1, 2, 3), black_ridges=False)
        ridge_response = normalize01(np.nan_to_num(ridge_response, nan=0.0))
    except Exception:
        ridge_response = np.zeros_like(dark_response, dtype=np.float32)

    likelihood = 0.55 * dark_response + 0.30 * ridge_response + 0.15 * edge_response
    return normalize01(filters.gaussian(likelihood, sigma=0.6, preserve_range=True))


# --------------------------------------------------------------------------- #
# Component filtering                                                           #
# --------------------------------------------------------------------------- #

def _component_elongation(region: Any) -> float:
    minor = float(getattr(region, "axis_minor_length", None) or getattr(region, "minor_axis_length", 0.0) or 0.0)
    major = float(getattr(region, "axis_major_length", None) or getattr(region, "major_axis_length", 0.0) or 0.0)
    return major / max(minor, 1.0)


def filter_crack_components(
    mask: np.ndarray,
    likelihood: np.ndarray,
    min_object_size: int,
    min_component_length: int,
    min_elongation: float,
) -> np.ndarray:
    """
    Remove non-crack blobs from the binary mask.

    Rejection criteria (any one causes rejection):
    - Area below min_object_size              — too small to be a meaningful crack
    - Skeleton length below min_component_length — too short to be a crack
    - Compact blob (high extent + low elongation) — stain or shadow, not a crack
    - Low mean likelihood intensity            — the region is weak, likely noise

    Acceptance criteria (any one causes acceptance):
    - High elongation (crack-like shape)
    - High skeleton length density
    - Very long skeleton (obviously a crack)
    """
    labeled = measure.label(mask)
    filtered = np.zeros_like(mask, dtype=bool)

    for region in measure.regionprops(labeled, intensity_image=likelihood):
        if region.area < min_object_size:
            continue

        component = labeled == region.label
        skeleton = morphology.skeletonize(component)
        skeleton_length = int(skeleton.sum())

        if skeleton_length < min_component_length:
            continue

        elongation = _component_elongation(region)
        length_density = skeleton_length / max(float(np.sqrt(region.area)), 1.0)
        mean_intensity = float(
            getattr(region, "intensity_mean", None)
            or getattr(region, "mean_intensity", 0.0)
            or 0.0
        )

        # Reject compact blobs (stains, shadows)
        compact_blob = region.extent > 0.72 and elongation < min_elongation
        # Reject very weak regions (noise)
        weak_region = mean_intensity < 0.40

        if compact_blob or weak_region:
            continue

        crack_like = (
            elongation >= min_elongation
            or length_density >= 2.0
            or skeleton_length >= min_component_length * 2
        )
        if crack_like:
            filtered[component] = True

    return filtered


# --------------------------------------------------------------------------- #
# Segmentation                                                                  #
# --------------------------------------------------------------------------- #

def segment_crack(
    gray: np.ndarray,
    min_object_size: int,
    min_component_length: int,
    min_elongation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce a cleaned binary crack mask and the raw likelihood map.

    Steps:
      1. Compute crack likelihood map.
      2. Threshold at max(Otsu, 93rd percentile) — conservative to avoid false positives.
      3. Intersect with dark-pixel constraint.
      4. Morphological cleaning (close small gaps, fill small holes).
      5. Component-level filtering to keep only crack-like elongated structures.
      6. Final dilation to restore crack width removed by skeletonisation.
    """
    likelihood = crack_likelihood(gray)

    if np.allclose(likelihood.max(), likelihood.min()):
        return np.zeros_like(gray, dtype=bool), likelihood

    otsu = float(filters.threshold_otsu(likelihood))
    high_quantile = float(np.quantile(likelihood, 0.93))
    threshold = max(otsu, high_quantile)
    mask = likelihood >= threshold

    dark_pixels = gray <= np.quantile(gray, 0.36)
    high_likelihood = likelihood >= np.quantile(likelihood, 0.975)
    mask = np.logical_and(mask, dark_pixels | high_likelihood)

    mask = morphology.closing(mask, morphology.disk(1))
    mask = morphology.remove_small_holes(mask, max_size=max(16, min_object_size // 2))

    mask = filter_crack_components(
        mask,
        likelihood=likelihood,
        min_object_size=min_object_size,
        min_component_length=min_component_length,
        min_elongation=min_elongation,
    )

    # Restore crack body after filtering and skeletonisation
    mask = morphology.dilation(mask, morphology.disk(1))
    mask = morphology.closing(mask, morphology.disk(1))
    mask = morphology.remove_small_holes(mask, max_size=max(24, min_object_size))

    return mask.astype(bool), likelihood


# --------------------------------------------------------------------------- #
# Polygon extraction                                                            #
# --------------------------------------------------------------------------- #

def simplify_contour(contour: np.ndarray, max_points: int) -> list[list[int]]:
    tolerance = 1.5
    simplified = measure.approximate_polygon(contour, tolerance=tolerance)
    while len(simplified) > max_points and tolerance < 12:
        tolerance += 1.5
        simplified = measure.approximate_polygon(contour, tolerance=tolerance)
    if len(simplified) < 3:
        simplified = contour
    if len(simplified) > max_points:
        step = int(np.ceil(len(simplified) / max_points))
        simplified = simplified[::step]
    return [[int(round(pt[1])), int(round(pt[0]))] for pt in simplified]


def extract_polygons(
    mask: np.ndarray,
    max_components: int,
    max_polygon_points: int,
) -> list[dict[str, Any]]:
    labeled = measure.label(mask)
    regions = sorted(measure.regionprops(labeled), key=lambda r: r.area, reverse=True)
    polygons: list[dict[str, Any]] = []

    for region in regions[:max_components]:
        component = labeled == region.label
        contours = measure.find_contours(component.astype(np.float32), level=0.5)
        if not contours:
            min_row, min_col, max_row, max_col = region.bbox
            points = [
                [int(min_col), int(min_row)],
                [int(max_col), int(min_row)],
                [int(max_col), int(max_row)],
                [int(min_col), int(max_row)],
            ]
        else:
            contour = max(contours, key=len)
            points = simplify_contour(contour, max_points=max_polygon_points)

        min_row, min_col, max_row, max_col = region.bbox
        polygons.append(
            {
                "component_id": int(region.label),
                "area_px": int(region.area),
                "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)],
                "polygon": points,
            }
        )
    return polygons


# --------------------------------------------------------------------------- #
# Severity                                                                      #
# --------------------------------------------------------------------------- #

def severity_from_measurements(
    crack_area_pct: float,
    crack_length_px: float,
    max_width_px: float,
    width: int,
    height: int,
    probability: float,
    scale_mm_per_px: float | None = None,
) -> tuple[float, str, str]:
    """
    Classify crack severity.

    When a mm/px scale is available (from EXIF calibration or user input):
      - Uses ACI 224R width thresholds: <0.1 mm hairline, 0.1-0.3 fine, 0.3-1.0 medium, >1.0 wide.

    Without scale (pixel-estimate mode):
      - Composite score from area %, length ratio, and model probability.
      - Severity labels are provisional and marked as pixel_estimate.
    """
    max_width_mm = max_width_px * scale_mm_per_px if scale_mm_per_px else None
    length_ratio = crack_length_px / max(width, height, 1)

    # Composite severity score (0–1)
    score = min(
        1.0,
        (crack_area_pct / 0.08) * 0.45
        + min(length_ratio / 1.50, 1.0) * 0.35
        + probability * 0.20,
    )

    if max_width_mm is not None:
        # ACI 224R-01 width thresholds
        if max_width_mm >= 1.0:
            label = "high"
        elif max_width_mm >= 0.3:
            label = "medium"
        elif max_width_mm >= 0.1:
            label = "low"
        else:
            label = "hairline"
        return float(score), label, "calibrated_mm"

    # Pixel-estimate fallback
    if score >= 0.66:
        label = "high"
    elif score >= 0.33:
        label = "medium"
    else:
        label = "low"
    return float(score), label, "pixel_estimate"


# --------------------------------------------------------------------------- #
# Artifact writers                                                              #
# --------------------------------------------------------------------------- #

def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_heatmap(path: Path, image: Image.Image, likelihood: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGB")
    heat = Image.fromarray(heatmap_rgb(likelihood), mode="RGB").resize(base.size)
    blended = Image.blend(base, heat, alpha=0.45)
    blended.save(path, quality=92)


def save_overlay(
    path: Path,
    image: Image.Image,
    mask: np.ndarray,
    polygons: list[dict[str, Any]],
) -> None:
    """
    Overlay the crack mask (semi-transparent red fill) and crack contour
    polygons (bright yellow outline) on the original image.
    The polygon traces the actual crack contour, not a bounding rectangle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    mask_image = Image.fromarray((mask.astype(np.uint8) * 175), mode="L").resize(base.size)
    red_layer = Image.new("RGBA", base.size, (230, 55, 40, 0))
    red_layer.putalpha(mask_image)
    overlay = Image.alpha_composite(base, red_layer)

    draw = ImageDraw.Draw(overlay)
    for item in polygons:
        points = [tuple(pt) for pt in item["polygon"]]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=(255, 222, 75, 255), width=2)

    overlay.convert("RGB").save(path, quality=92)


# --------------------------------------------------------------------------- #
# Per-image analysis                                                            #
# --------------------------------------------------------------------------- #

def analyze_image(
    row: pd.Series,
    output_dir: Path,
    min_object_size: int,
    max_components: int,
    max_polygon_points: int,
    min_component_length: int = 18,
    min_elongation: float = 1.8,
    scale_mm_per_px: float | None = None,
) -> dict[str, Any]:
    image_path = Path(str(row["path"]))
    image = Image.open(image_path).convert("RGB")
    gray = util.img_as_float(np.asarray(image.convert("L")))
    width, height = image.size

    mask, likelihood = segment_crack(
        gray,
        min_object_size=min_object_size,
        min_component_length=min_component_length,
        min_elongation=min_elongation,
    )

    skeleton = morphology.skeletonize(mask)
    crack_area_px = int(mask.sum())
    crack_length_px = float(skeleton.sum())
    crack_area_pct = float(crack_area_px / max(width * height, 1))

    if crack_length_px > 0:
        _, distance = morphology.medial_axis(mask, return_distance=True)
        skeleton_widths = np.asarray(distance[skeleton], dtype=np.float32) * 2.0
        mean_width_px = float(np.mean(skeleton_widths)) if skeleton_widths.size else 0.0
        max_width_px = float(np.max(skeleton_widths)) if skeleton_widths.size else 0.0
    else:
        mean_width_px = 0.0
        max_width_px = 0.0

    polygons = extract_polygons(
        mask,
        max_components=max_components,
        max_polygon_points=max_polygon_points,
    )

    probability = float(row.get("crack_probability", 0.0) or 0.0)
    severity_score, severity_label, severity_basis = severity_from_measurements(
        crack_area_pct=crack_area_pct,
        crack_length_px=crack_length_px,
        max_width_px=max_width_px,
        width=width,
        height=height,
        probability=probability,
        scale_mm_per_px=scale_mm_per_px,
    )

    image_id = str(row["image_id"])
    overlay_path = output_dir / "overlays" / f"{image_id}.jpg"
    heatmap_path = output_dir / "heatmaps" / f"{image_id}.jpg"
    mask_path = output_dir / "masks" / f"{image_id}.png"

    save_overlay(overlay_path, image, mask, polygons)
    save_heatmap(heatmap_path, image, likelihood)
    save_mask(mask_path, mask)

    return {
        "image_id": image_id,
        "predicted_label": row.get("predicted_label"),
        "crack_probability": probability,
        "width": int(width),
        "height": int(height),
        "component_count": int(len(polygons)),
        "crack_area_px": crack_area_px,
        "crack_area_pct": crack_area_pct,
        "crack_length_px": crack_length_px,
        "mean_width_px": mean_width_px,
        "max_width_px": max_width_px,
        "scale_mm_per_px": scale_mm_per_px,
        "crack_length_mm": crack_length_px * scale_mm_per_px if scale_mm_per_px else None,
        "mean_width_mm": mean_width_px * scale_mm_per_px if scale_mm_per_px else None,
        "max_width_mm": max_width_px * scale_mm_per_px if scale_mm_per_px else None,
        "severity_score": severity_score,
        "severity_label": severity_label,
        "severity_basis": severity_basis,
        "segmentation_source": "heuristic_clahe_frangi_morphology",
        "measurement_method": "mask_skeleton_distance_transform",
        "overlay_path": str(overlay_path.resolve()),
        "heatmap_path": str(heatmap_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "polygons_json": json.dumps(polygons),
    }


# --------------------------------------------------------------------------- #
# Batch runner                                                                  #
# --------------------------------------------------------------------------- #

def summarize_localizations(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    if df.empty:
        return {
            "created_at": utc_now_iso(),
            "rows": 0,
            "localizations_path": str(output_path.resolve()),
        }
    return {
        "created_at": utc_now_iso(),
        "rows": int(len(df)),
        "localizations_path": str(output_path.resolve()),
        "severity_labels": df["severity_label"].value_counts().to_dict(),
        "average_area_pct": float(df["crack_area_pct"].mean()),
        "average_length_px": float(df["crack_length_px"].mean()),
        "average_mean_width_px": float(df["mean_width_px"].mean()),
        "average_max_width_px": float(df["max_width_px"].mean()),
        "average_severity_score": float(df["severity_score"].mean()),
        "total_crack_area_px": int(df["crack_area_px"].sum()),
        "total_crack_length_px": float(df["crack_length_px"].sum()),
        "segmentation_source": "heuristic_clahe_frangi_morphology",
        "measurement_method": "mask_skeleton_distance_transform",
        "severity_basis": sorted(df["severity_basis"].dropna().unique().tolist())
        if "severity_basis" in df
        else ["pixel_estimate"],
    }


def run_localization(
    predictions_path: Path,
    output_path: Path,
    output_dir: Path,
    predicted_label: str,
    limit: int,
    min_object_size: int,
    max_components: int,
    max_polygon_points: int,
    min_component_length: int,
    min_elongation: float,
    scale_mm_per_px: float | None,
    update_summary: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_data_dirs()
    predictions = pd.read_csv(predictions_path)
    selected = predictions[predictions["predicted_label"] == predicted_label].copy()
    selected = selected.sort_values("crack_probability", ascending=False)
    if limit > 0:
        selected = selected.head(limit).copy()

    rows: list[dict[str, Any]] = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Localizing cracks"):
        try:
            rows.append(
                analyze_image(
                    row,
                    output_dir=output_dir,
                    min_object_size=min_object_size,
                    max_components=max_components,
                    max_polygon_points=max_polygon_points,
                    min_component_length=min_component_length,
                    min_elongation=min_elongation,
                    scale_mm_per_px=scale_mm_per_px,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "image_id": row.get("image_id"),
                    "predicted_label": row.get("predicted_label"),
                    "crack_probability": row.get("crack_probability"),
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    clean = result[result["error"].isna()] if "error" in result.columns else result
    summary = summarize_localizations(clean, output_path)
    write_json(output_path.with_suffix(".summary.json"), summary)

    if update_summary:
        pipeline_summary = read_json(DEFAULT_SUMMARY)
        pipeline_summary["localization"] = summary
        write_json(DEFAULT_SUMMARY, pipeline_summary)

    return result, summary


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate crack polygons, area, length, severity, and heatmaps for cracked predictions."
    )
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_LOCALIZATIONS)
    parser.add_argument("--output-dir", type=Path, default=LOCALIZATION_DIR)
    parser.add_argument("--predicted-label", default="cracked")
    parser.add_argument("--limit", type=int, default=0, help="0 localizes every cracked prediction.")
    parser.add_argument("--min-object-size", type=int, default=64)
    parser.add_argument("--min-component-length", type=int, default=18)
    parser.add_argument("--min-elongation", type=float, default=1.8)
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--max-polygon-points", type=int, default=120)
    parser.add_argument(
        "--scale-mm-per-px",
        type=float,
        default=None,
        help="Calibration scale. If omitted severity uses pixel estimates and ACI 224R thresholds are unavailable.",
    )
    parser.add_argument("--no-summary-update", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, summary = run_localization(
        predictions_path=args.predictions_path,
        output_path=args.output_path,
        output_dir=args.output_dir,
        predicted_label=args.predicted_label,
        limit=args.limit,
        min_object_size=args.min_object_size,
        max_components=args.max_components,
        max_polygon_points=args.max_polygon_points,
        min_component_length=args.min_component_length,
        min_elongation=args.min_elongation,
        scale_mm_per_px=args.scale_mm_per_px,
        update_summary=not args.no_summary_update,
    )
    print(f"Wrote {len(result):,} localizations to {args.output_path}")
    print(f"Severity labels : {summary.get('severity_labels', {})}")
    print(f"Avg area %      : {summary.get('average_area_pct', 0):.4f}")
    print(f"Avg length (px) : {summary.get('average_length_px', 0):.1f}")
    print(f"Avg width  (px) : {summary.get('average_mean_width_px', 0):.2f}")


if __name__ == "__main__":
    main()