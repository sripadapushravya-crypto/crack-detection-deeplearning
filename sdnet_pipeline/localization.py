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
            [20, 30, 90],
            [30, 120, 190],
            [80, 190, 160],
            [245, 210, 75],
            [220, 65, 40],
        ],
        dtype=np.float32,
    )
    scaled = score * (len(stops) - 1)
    idx = np.floor(scaled).astype(int)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = (scaled - idx)[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    return np.clip(rgb, 0, 255).astype(np.uint8)


def crack_likelihood(gray: np.ndarray) -> np.ndarray:
    smoothed = filters.gaussian(gray, sigma=0.8, preserve_range=True)
    equalized = exposure.equalize_adapthist(smoothed, clip_limit=0.03)
    dark_response = normalize01(1.0 - equalized)
    edge_response = normalize01(filters.sobel(equalized))

    try:
        ridge_response = filters.frangi(
            dark_response,
            sigmas=(1, 2, 3),
            black_ridges=False,
        )
        ridge_response = normalize01(np.nan_to_num(ridge_response, nan=0.0))
    except Exception:
        ridge_response = np.zeros_like(dark_response, dtype=np.float32)

    likelihood = 0.60 * dark_response + 0.25 * ridge_response + 0.15 * edge_response
    return normalize01(filters.gaussian(likelihood, sigma=0.6, preserve_range=True))


def component_elongation(region: measure._regionprops.RegionProperties) -> float:
    minor_value = region.axis_minor_length if hasattr(region, "axis_minor_length") else region.minor_axis_length
    major_value = region.axis_major_length if hasattr(region, "axis_major_length") else region.major_axis_length
    minor = float(minor_value or 0.0)
    major = float(major_value or 0.0)
    if minor <= 0:
        return major
    return major / minor


def filter_crack_components(
    mask: np.ndarray,
    likelihood: np.ndarray,
    min_object_size: int,
    min_component_length: int,
    min_elongation: float,
) -> np.ndarray:
    labeled = measure.label(mask)
    filtered = np.zeros_like(mask, dtype=bool)

    for region in measure.regionprops(labeled, intensity_image=likelihood):
        if region.area < min_object_size:
            continue

        component = labeled == region.label
        skeleton_length = int(morphology.skeletonize(component).sum())
        if skeleton_length < min_component_length:
            continue

        elongation = component_elongation(region)
        length_density = skeleton_length / max(float(np.sqrt(region.area)), 1.0)
        compact_blob = region.extent > 0.72 and elongation < min_elongation
        intensity_value = region.intensity_mean if hasattr(region, "intensity_mean") else region.mean_intensity
        mean_intensity = float(intensity_value or 0.0)
        weak_region = mean_intensity < 0.42

        if compact_blob or weak_region:
            continue

        if elongation >= min_elongation or length_density >= 2.0 or skeleton_length >= min_component_length * 2:
            filtered[component] = True

    return filtered


def segment_crack(
    gray: np.ndarray,
    min_object_size: int,
    min_component_length: int,
    min_elongation: float,
) -> tuple[np.ndarray, np.ndarray]:
    likelihood = crack_likelihood(gray)
    if np.allclose(likelihood.max(), likelihood.min()):
        return np.zeros_like(gray, dtype=bool), likelihood

    otsu = float(filters.threshold_otsu(likelihood))
    high_quantile = float(np.quantile(likelihood, 0.93))
    threshold = max(otsu, high_quantile)
    mask = likelihood >= threshold

    dark_pixels = gray <= np.quantile(gray, 0.36)
    mask = np.logical_and(mask, dark_pixels | (likelihood >= np.quantile(likelihood, 0.975)))
    mask = morphology.closing(mask, morphology.disk(1))
    mask = morphology.remove_small_holes(mask, max_size=max(16, min_object_size // 2))
    mask = filter_crack_components(
        mask,
        likelihood=likelihood,
        min_object_size=min_object_size,
        min_component_length=min_component_length,
        min_elongation=min_elongation,
    )
    # Width is measured from THIS mask (skeleton + distance transform) in
    # analyze_image. The earlier cosmetic dilation/closing inflated measured
    # crack width by ~2 px, so it is removed; only enclosed pin-holes are filled
    # to keep geometry faithful while the mask stays visible in the overlay.
    mask = morphology.remove_small_holes(mask, max_size=max(24, min_object_size))
    return mask.astype(bool), likelihood


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
    return [[int(round(point[1])), int(round(point[0]))] for point in simplified]


def extract_polygons(mask: np.ndarray, max_components: int, max_polygon_points: int) -> list[dict[str, Any]]:
    labeled = measure.label(mask)
    regions = sorted(measure.regionprops(labeled), key=lambda region: region.area, reverse=True)
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


# ---------------------------------------------------------------------------
# Scale calibration: pixels -> millimetres
# ---------------------------------------------------------------------------
# A pixel measurement only becomes a physical length once the image scale
# (millimetres per pixel) is known. The scale can be recovered three ways, in
# descending order of field trustworthiness:
#   "aruco"    - a printed ArUco marker of known side length placed in frame;
#                per-image and self-correcting when the camera distance changes.
#   "geometry" - ground sample distance from camera standoff and intrinsics.
#   "manual"   - an explicit, externally supplied mm-per-pixel constant
#                (e.g. a one-time fixed-rig calibration).
# When no scale is available the result is None and measurements stay
# pixel-domain, exactly as before.


def scale_from_reference_object(
    image: Image.Image,
    marker_length_mm: float,
    aruco_dict: str = "DICT_4X4_50",
) -> float | None:
    """Millimetres-per-pixel from a square fiducial of known side length.

    The marker is detected in the SAME working-resolution image whose cracks are
    measured, so the returned scale is already in working-pixel units and needs
    no resize adjustment. Returns None if OpenCV is unavailable or no marker is
    found.
    """
    try:
        import cv2  # optional dependency; only needed for marker calibration
    except Exception:
        return None

    gray = np.asarray(image.convert("L"))
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict))
    try:  # OpenCV >= 4.7 object API
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:  # OpenCV < 4.7 functional API
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)

    if ids is None or len(corners) == 0:
        return None

    # Average the four edge lengths of every detected marker, so the estimate is
    # robust to mild perspective and detection noise.
    edges_px: list[float] = []
    for quad in corners:
        pts = quad.reshape(4, 2)
        edges_px.extend(float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4))
    mean_side_px = float(np.mean(edges_px)) if edges_px else 0.0
    if mean_side_px <= 0:
        return None
    return marker_length_mm / mean_side_px


def scale_from_geometry(
    distance_mm: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
) -> float | None:
    """Millimetres-per-pixel (ground sample distance) from capture geometry.

        s = (sensor_width_mm * distance_mm) / (focal_length_mm * image_width_px)

    Assumes the sensor is roughly parallel to the surface. ``image_width_px`` is
    the width of the ORIGINAL capture that the intrinsics describe.
    """
    if min(distance_mm, focal_length_mm, sensor_width_mm, image_width_px) <= 0:
        return None
    return (sensor_width_mm * distance_mm) / (focal_length_mm * image_width_px)


def resolve_scale(
    image: Image.Image,
    scale_config: dict[str, Any] | None,
    working_width_px: int,
    original_width_px: int,
) -> tuple[float | None, str]:
    """Return (scale_mm_per_px, scale_source) expressed in WORKING pixels.

    ``scale_config`` selects the method and carries its parameters, e.g.::

        {"source": "aruco",    "marker_length_mm": 50.0, "aruco_dict": "DICT_4X4_50"}
        {"source": "geometry", "distance_mm": 500, "focal_length_mm": 4.3,
                               "sensor_width_mm": 6.4}
        {"source": "manual",   "scale_mm_per_px": 0.18}   # refers to ORIGINAL capture

    ArUco scales are measured on the working image and are exact. Geometry and
    manual scales refer to the original capture resolution, so when an oversized
    upload was downscaled they are divided by the resize factor (working pixels
    are larger and therefore span more millimetres). Returns (None, "none") when
    no scale can be resolved.
    """
    if not scale_config:
        return None, "none"

    source = scale_config.get("source", "none")
    f = (working_width_px / original_width_px) if original_width_px else 1.0
    f = f if f > 0 else 1.0

    if source == "aruco":
        s = scale_from_reference_object(
            image,
            marker_length_mm=float(scale_config["marker_length_mm"]),
            aruco_dict=scale_config.get("aruco_dict", "DICT_4X4_50"),
        )
        return (s, "aruco") if s else (None, "none")

    if source == "geometry":
        s = scale_from_geometry(
            distance_mm=float(scale_config["distance_mm"]),
            focal_length_mm=float(scale_config["focal_length_mm"]),
            sensor_width_mm=float(scale_config["sensor_width_mm"]),
            image_width_px=int(original_width_px),
        )
        return (s / f, "geometry") if s else (None, "none")

    if source == "manual":
        s = scale_config.get("scale_mm_per_px")
        return (float(s) / f, "manual") if s else (None, "none")

    return None, "none"


def severity_from_measurements(
    crack_area_pct: float,
    crack_length_px: float,
    max_width_px: float,
    width: int,
    height: int,
    probability: float,
    scale_mm_per_px: float | None = None,
) -> tuple[float, str, str]:
    max_width_mm = max_width_px * scale_mm_per_px if scale_mm_per_px else None
    length_ratio = crack_length_px / max(width, height, 1)
    score = min(
        1.0,
        (crack_area_pct / 0.08) * 0.45
        + min(length_ratio / 1.50, 1.0) * 0.35
        + probability * 0.20,
    )

    if max_width_mm is not None:
        if max_width_mm >= 1.0:
            label = "high"
        elif max_width_mm >= 0.3:
            label = "medium"
        else:
            label = "low"
        return float(score), label, "calibrated_mm"

    if score >= 0.66:
        label = "high"
    elif score >= 0.33:
        label = "medium"
    else:
        label = "low"
    return float(score), label, "pixel_estimate"


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_heatmap(path: Path, image: Image.Image, likelihood: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGB")
    heat = Image.fromarray(heatmap_rgb(likelihood), mode="RGB").resize(base.size)
    blended = Image.blend(base, heat, alpha=0.45)
    blended.save(path, quality=92)


def save_overlay(path: Path, image: Image.Image, mask: np.ndarray, polygons: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    mask_image = Image.fromarray((mask.astype(np.uint8) * 175), mode="L").resize(base.size)
    red = Image.new("RGBA", base.size, (230, 55, 40, 0))
    red.putalpha(mask_image)
    overlay = Image.alpha_composite(base, red)

    draw = ImageDraw.Draw(overlay)
    for item in polygons:
        points = [tuple(point) for point in item["polygon"]]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=(255, 222, 75, 255), width=2)
    overlay.convert("RGB").save(path, quality=92)


def analyze_image(
    row: pd.Series,
    output_dir: Path,
    min_object_size: int,
    max_components: int,
    max_polygon_points: int,
    min_component_length: int = 18,
    min_elongation: float = 1.8,
    scale_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_path = Path(str(row["path"]))
    image = Image.open(image_path).convert("RGB")
    original_width_px = int(image.size[0])
    # Real-life photos are often very large; the Frangi sigmas and component-size
    # thresholds below are tuned for ~256-1024 px. Downscale oversized uploads so
    # ridge detection works at a sensible scale and stays fast. SDNET images
    # (~256 px) are unaffected.
    _MAX_DIM = 1024
    if max(image.size) > _MAX_DIM:
        _resize = _MAX_DIM / max(image.size)
        image = image.resize(
            (max(1, round(image.size[0] * _resize)), max(1, round(image.size[1] * _resize))),
            Image.LANCZOS,
        )
    gray = util.img_as_float(np.asarray(image.convert("L")))
    width, height = image.size

    # Recover the millimetre scale for THIS image (marker / geometry / manual),
    # measured against the working resolution actually used for segmentation.
    scale_mm_per_px, scale_source = resolve_scale(
        image,
        scale_config=scale_config,
        working_width_px=width,
        original_width_px=original_width_px,
    )

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
    polygons = extract_polygons(mask, max_components=max_components, max_polygon_points=max_polygon_points)

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

    # Area scales with the SQUARE of the linear scale, unlike length and width.
    crack_area_mm2 = float(crack_area_px) * scale_mm_per_px ** 2 if scale_mm_per_px else None

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
        "scale_source": scale_source,
        "crack_area_mm2": crack_area_mm2,
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


def summarize_localizations(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    if df.empty:
        summary = {
            "created_at": utc_now_iso(),
            "rows": 0,
            "localizations_path": str(output_path.resolve()),
        }
        return summary

    return {
        "created_at": utc_now_iso(),
        "rows": int(len(df)),
        "localizations_path": str(output_path.resolve()),
        "severity_labels": df["severity_label"].value_counts().to_dict(),
        "average_area_pct": float(df["crack_area_pct"].mean()),
        "average_length_px": float(df["crack_length_px"].mean()),
        "average_mean_width_px": float(df["mean_width_px"].mean()),
        "average_max_width_px": float(df["max_width_px"].mean()),
        **(
            {
                "average_mean_width_mm": float(df["mean_width_mm"].dropna().mean()),
                "average_max_width_mm": float(df["max_width_mm"].dropna().mean()),
            }
            if "max_width_mm" in df and df["max_width_mm"].notna().any()
            else {}
        ),
        "scale_sources": df["scale_source"].value_counts().to_dict()
        if "scale_source" in df
        else {"none": int(len(df))},
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
    scale_config: dict[str, Any] | None,
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
                    scale_config=scale_config,
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
    summary = summarize_localizations(result[result.get("error").isna()] if "error" in result else result, output_path)
    write_json(output_path.with_suffix(".summary.json"), summary)

    if update_summary:
        pipeline_summary = read_json(DEFAULT_SUMMARY)
        pipeline_summary["localization"] = summary
        write_json(DEFAULT_SUMMARY, pipeline_summary)

    return result, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate crack polygons, area, length, severity, and heatmaps for cracked predictions."
    )
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_LOCALIZATIONS)
    parser.add_argument("--output-dir", type=Path, default=LOCALIZATION_DIR)
    parser.add_argument("--predicted-label", default="cracked")
    parser.add_argument("--limit", type=int, default=0, help="0 means localize every cracked prediction.")
    parser.add_argument("--min-object-size", type=int, default=64)
    parser.add_argument("--min-component-length", type=int, default=18)
    parser.add_argument("--min-elongation", type=float, default=1.8)
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--max-polygon-points", type=int, default=120)
    parser.add_argument(
        "--scale-source",
        choices=["none", "manual", "aruco", "geometry"],
        default="none",
        help="How to obtain the mm-per-pixel scale. 'none' keeps measurements pixel-domain.",
    )
    parser.add_argument(
        "--scale-mm-per-px",
        type=float,
        default=None,
        help="Manual calibration scale (mm per pixel of the original capture). "
        "Used when --scale-source manual, or as a bare back-compatible override.",
    )
    parser.add_argument("--marker-length-mm", type=float, default=None,
                        help="Physical side length of the ArUco fiducial (--scale-source aruco).")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50",
                        help="OpenCV ArUco dictionary name (--scale-source aruco).")
    parser.add_argument("--distance-mm", type=float, default=None,
                        help="Camera-to-surface standoff in mm (--scale-source geometry).")
    parser.add_argument("--focal-length-mm", type=float, default=None,
                        help="Lens focal length in mm (--scale-source geometry).")
    parser.add_argument("--sensor-width-mm", type=float, default=None,
                        help="Camera sensor width in mm (--scale-source geometry).")
    parser.add_argument("--no-summary-update", action="store_true")
    return parser.parse_args()


def build_scale_config(args: argparse.Namespace) -> dict[str, Any] | None:
    """Translate CLI arguments into a scale_config dict for resolve_scale()."""
    source = args.scale_source
    if source == "none":
        # Back-compat: a bare --scale-mm-per-px still works as a manual scale.
        if args.scale_mm_per_px:
            return {"source": "manual", "scale_mm_per_px": args.scale_mm_per_px}
        return None
    if source == "manual":
        return {"source": "manual", "scale_mm_per_px": args.scale_mm_per_px}
    if source == "aruco":
        return {
            "source": "aruco",
            "marker_length_mm": args.marker_length_mm,
            "aruco_dict": args.aruco_dict,
        }
    if source == "geometry":
        return {
            "source": "geometry",
            "distance_mm": args.distance_mm,
            "focal_length_mm": args.focal_length_mm,
            "sensor_width_mm": args.sensor_width_mm,
        }
    return None


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
        scale_config=build_scale_config(args),
        update_summary=not args.no_summary_update,
    )
    print(f"Wrote {len(result):,} localizations to {args.output_path}")
    print(f"Severity labels: {summary.get('severity_labels', {})}")


if __name__ == "__main__":
    main()
