from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage import exposure, measure
from skimage.feature import hog, local_binary_pattern
from skimage.filters import frangi, sobel


def load_grayscale(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize((image_size, image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def _safe_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=True)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    return hist.astype(np.float32)


def _elongation_features(arr: np.ndarray, dark_quantile: float) -> np.ndarray:
    """
    Elongation statistics of dark connected regions at a given darkness threshold.
    Cracks appear as elongated dark structures; stains and shadows are compact blobs.
    Returns 3 features: max elongation, mean elongation, fraction of highly elongated regions.
    """
    threshold = float(np.quantile(arr, dark_quantile))
    dark_mask = arr < threshold
    labeled = measure.label(dark_mask)
    regions = measure.regionprops(labeled)

    elongations: list[float] = []
    for region in regions:
        if region.area < 10:
            continue
        minor = float(region.minor_axis_length or 1.0)
        major = float(region.major_axis_length or 0.0)
        elongations.append(major / max(minor, 1.0))

    if not elongations:
        return np.zeros(3, dtype=np.float32)

    return np.array(
        [
            float(np.max(elongations)),
            float(np.mean(elongations)),
            float(sum(1 for e in elongations if e > 5.0) / max(len(elongations), 1)),
        ],
        dtype=np.float32,
    )


def _frangi_features(clahe_arr: np.ndarray) -> np.ndarray:
    """
    Frangi ridge filter responses — specifically designed for thin elongated structures.
    Two passes: dark ridges (cracks on bright concrete) and inverted for bright ridges.
    Returns 8 scalar statistics that strongly discriminate cracks from flat surfaces.
    """
    try:
        dark_ridge = frangi(clahe_arr, sigmas=(1, 2, 3), black_ridges=True)
        dark_ridge = np.nan_to_num(dark_ridge, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        dark_ridge = np.zeros_like(clahe_arr, dtype=np.float32)

    try:
        bright_ridge = frangi(1.0 - clahe_arr, sigmas=(1, 2, 3), black_ridges=False)
        bright_ridge = np.nan_to_num(bright_ridge, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        bright_ridge = np.zeros_like(clahe_arr, dtype=np.float32)

    q90_dark = float(np.quantile(dark_ridge, 0.90)) if dark_ridge.max() > 0 else 0.0
    q90_bright = float(np.quantile(bright_ridge, 0.90)) if bright_ridge.max() > 0 else 0.0

    return np.array(
        [
            float(dark_ridge.max()),
            float(dark_ridge.mean()),
            float(np.quantile(dark_ridge, 0.95)),
            float(np.mean(dark_ridge > q90_dark)) if q90_dark > 0 else 0.0,
            float(bright_ridge.max()),
            float(bright_ridge.mean()),
            float(np.quantile(bright_ridge, 0.95)),
            float(np.mean(bright_ridge > q90_bright)) if q90_bright > 0 else 0.0,
        ],
        dtype=np.float32,
    )


def _fine_hog(arr: np.ndarray) -> np.ndarray:
    """
    Fine-scale HOG on a 112x112 downsampled image using 8x8 cells.
    Captures thin crack patterns that the coarse 16x16-cell HOG misses.
    Produces the same dimensionality as the standard HOG because the image
    is halved while the cell size is halved, keeping the cell grid identical.
    """
    small = np.array(
        Image.fromarray((arr * 255).astype(np.uint8)).resize((112, 112)),
        dtype=np.float32,
    ) / 255.0
    return hog(
        small,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def extract_features(path: str | Path, image_size: int = 224) -> np.ndarray:
    arr = load_grayscale(Path(path), image_size=image_size)

    # ------------------------------------------------------------------ #
    # Block 1 — coarse HOG: shape and texture at 16x16 cells             #
    # ------------------------------------------------------------------ #
    hog_coarse = hog(
        arr,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Block 2 — fine HOG: thin crack patterns at 8x8 cells (112x112)     #
    # ------------------------------------------------------------------ #
    hog_fine = _fine_hog(arr)

    # ------------------------------------------------------------------ #
    # Block 3 — LBP texture                                               #
    # ------------------------------------------------------------------ #
    lbp = local_binary_pattern((arr * 255).astype(np.uint8), P=8, R=1, method="uniform")
    lbp_hist = _safe_histogram(lbp, bins=10, value_range=(0, 10))

    # ------------------------------------------------------------------ #
    # Block 4 — Sobel edge histogram                                      #
    # ------------------------------------------------------------------ #
    edges = sobel(arr)
    edge_hist = _safe_histogram(edges, bins=8, value_range=(0, float(edges.max() or 1.0)))

    # ------------------------------------------------------------------ #
    # Block 5 — intensity histogram                                       #
    # ------------------------------------------------------------------ #
    intensity_hist = _safe_histogram(arr, bins=16, value_range=(0, 1))

    # ------------------------------------------------------------------ #
    # Block 6 — Frangi ridge filter (crack-specific)                      #
    # CLAHE is applied first to enhance low-contrast crack edges.         #
    # ------------------------------------------------------------------ #
    clahe_arr = exposure.equalize_adapthist(arr, clip_limit=0.03).astype(np.float32)
    frangi_stats = _frangi_features(clahe_arr)

    # ------------------------------------------------------------------ #
    # Block 7 — elongation geometry at 3 darkness thresholds             #
    # Cracks are elongated dark regions; shadows/stains are compact blobs.#
    # ------------------------------------------------------------------ #
    geo_15 = _elongation_features(arr, dark_quantile=0.15)
    geo_20 = _elongation_features(arr, dark_quantile=0.20)
    geo_25 = _elongation_features(arr, dark_quantile=0.25)

    # ------------------------------------------------------------------ #
    # Block 8 — scalar statistics                                         #
    # ------------------------------------------------------------------ #
    dark_pixels = arr < np.quantile(arr, 0.15)
    strong_edges = edges > np.quantile(edges, 0.85)
    dark_edge_density = float(np.mean(dark_pixels & strong_edges))

    stats = np.array(
        [
            arr.mean(),
            arr.std(),
            np.quantile(arr, 0.10),
            np.quantile(arr, 0.50),
            np.quantile(arr, 0.90),
            edges.mean(),
            edges.std(),
            np.quantile(edges, 0.75),
            np.quantile(edges, 0.90),
            np.mean(edges > np.quantile(edges, 0.90)),
            dark_edge_density,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            hog_coarse,     # 6,084 — coarse shape and texture
            hog_fine,       # 6,084 — fine crack patterns
            lbp_hist,       #    10 — local texture
            intensity_hist, #    16 — brightness distribution
            edge_hist,      #     8 — edge strength distribution
            frangi_stats,   #     8 — ridge/crack filter response (NEW)
            geo_15,         #     3 — elongation of dark regions at Q15 (NEW)
            geo_20,         #     3 — elongation of dark regions at Q20 (NEW)
            geo_25,         #     3 — elongation of dark regions at Q25 (NEW)
            stats,          #    11 — scalar statistics
        ]
    )


def feature_matrix(paths: list[str], image_size: int) -> np.ndarray:
    return np.vstack([extract_features(path, image_size=image_size) for path in paths])
