"""
Regenerate all Group C figures using REAL SDNET2018 images and your own
pipeline functions.

This script must be run from the project root, so it can import
sdnet_pipeline.features and sdnet_pipeline.localization. It uses YOUR code
to produce the heatmaps, masks, polygons — meaning the figures faithfully
illustrate exactly what your pipeline actually does on real data.

USAGE:
    uv run python scripts/build_thesis_figures.py

What it produces (in data/thesis_figures/):
    fig_1_1_typical_cracks.png         (6-panel grid of real SDNET examples)
    fig_3_6_feature_extraction.png     (real input → HOG / LBP / Sobel / DED)
    fig_3_7_crack_likelihood.png       (CLAHE → dark → Frangi → Sobel → blend)
    fig_3_8_cc_filtering.png           (5-rule filter before/after, real mask)
    fig_3_9_skeletonisation.png        (skeleton + medial-axis on real mask)
    fig_4_4_qualitative.png            (3 real samples × 4 views)

YOU PICK 6 IMAGES FROM YOUR DATASET:
    The script needs you to populate the EXAMPLE_IMAGES list below with
    paths to 6 cracked images from your data/raw/sdnet2018 folder.
    Pick 2 from each surface (D/W/P) for variety, and make sure they have
    clearly visible cracks.

    Easy way to find good candidates:
        uv run python -c "import pandas as pd; df = pd.read_csv('data/results/predictions.csv'); print(df[df.predicted_label == 'cracked'].sort_values('crack_probability', ascending=False).head(20)[['image_id', 'path', 'surface']].to_string())"

    Then copy 6 of those paths into EXAMPLE_IMAGES below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters, morphology, measure, util

# Your own pipeline functions
from sdnet_pipeline.features import load_grayscale
from sdnet_pipeline.localization import (
    crack_likelihood,
    segment_crack,
    extract_polygons,
)


# ===========================================================================
# CONFIGURATION — edit these before running
# ===========================================================================

# IMPORTANT: replace these placeholders with REAL paths from your dataset.
# Use the one-liner at the top of this docstring to find good candidates.
EXAMPLE_IMAGES = [
    # (surface_label, image_path, short_caption_for_fig_1_1)
    ("Bridge Deck (D)", "data/raw/sdnet2018/Decks/Cracked/7019-59.jpg",      "Branched crack on bridge deck"),
    ("Bridge Deck (D)", "data/raw/sdnet2018/Decks/Cracked/7019-77.jpg",      "Crack near deck joint"),
    ("Wall (W)",        "data/raw/sdnet2018/Walls/Cracked/7092-153.jpg",     "Hairline crack on wall"),
    ("Wall (W)",        "data/raw/sdnet2018/Walls/Cracked/7081-207.jpg",     "Crack with surface staining"),
    ("Pavement (P)",    "data/raw/sdnet2018/Pavements/Cracked/087-189.jpg",  "Long thin pavement crack"),
    ("Pavement (P)",    "data/raw/sdnet2018/Pavements/Cracked/087-80.jpg",   "Wide pavement crack with branches"),
]

# The 3 images used for Fig 4.4 (the qualitative results grid). They should
# be your most-photogenic cracked predictions — usually high-confidence rows
# from predictions.csv with clearly visible cracks. Use indices 0-5 to pick
# from EXAMPLE_IMAGES, or override with different paths.
FIG_4_4_INDICES = [0, 1, 2]  # picks the first three images above

# Output folder (will be created)
OUT = Path("data/thesis_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Image size used by your training pipeline
IMAGE_SIZE = 224


# ===========================================================================
# Consistent matplotlib style — matches Figure 4.5 of the thesis
# ===========================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 180,
    "savefig.facecolor": "white",
})


# ===========================================================================
# Helper — load a real image, return (display_rgb, grayscale_float)
# ===========================================================================

def load_image_pair(path: str | Path, size: int = IMAGE_SIZE):
    """Load image, return (RGB array at native size, float grayscale at `size`)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Image not found: {p}\n"
            "Edit EXAMPLE_IMAGES at the top of this script with real paths."
        )
    rgb = np.asarray(Image.open(p).convert("RGB").resize((size, size)))
    gray = load_grayscale(p, image_size=size)
    return rgb, gray


# ===========================================================================
# Figure 1.1 — 6-panel grid of typical cracks
# ===========================================================================

def build_fig_1_1():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (surface, path, label) in zip(axes.flat, EXAMPLE_IMAGES):
        rgb, _ = load_image_pair(path)
        ax.imshow(rgb)
        ax.set_title(f"{label}\n[{surface}]", fontsize=11)
        ax.axis("off")
    plt.suptitle("Typical surface cracks observed on concrete infrastructure",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(OUT / "fig_1_1_typical_cracks.png")
    plt.close()
    print(f"  ✓ Saved {OUT / 'fig_1_1_typical_cracks.png'}")


# ===========================================================================
# Figure 3.6 — Feature extraction overview on a real image
# ===========================================================================

def build_fig_3_6():
    from skimage.feature import hog, local_binary_pattern
    from skimage.filters import sobel

    # Use the first cracked image as the example
    _, gray = load_image_pair(EXAMPLE_IMAGES[0][1])

    # Recompute the descriptors the same way features.py does
    _, hog_image = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        feature_vector=False, visualize=True,
    )
    edges = sobel(gray)
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform")

    # Dark-edge density set
    dark_pixels = gray < np.quantile(gray, 0.15)
    strong_edges = edges > np.quantile(edges, 0.85)
    dark_edge_set = dark_pixels & strong_edges
    dark_edge_density = float(dark_edge_set.mean())

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"(a) Input grayscale ({IMAGE_SIZE}×{IMAGE_SIZE})")

    axes[0, 1].imshow(hog_image, cmap="gray")
    axes[0, 1].set_title("(b) HOG gradient map\n9 orientations, 16×16 cells")

    axes[0, 2].imshow(lbp, cmap="viridis")
    axes[0, 2].set_title("(c) LBP texture map\nuniform, P=8 R=1")

    axes[1, 0].imshow(edges, cmap="inferno")
    axes[1, 0].set_title("(d) Sobel edge magnitude")

    # Dark-edge set: overlay on the grayscale
    overlay = np.stack([gray] * 3, axis=-1)
    overlay[dark_edge_set] = [1.0, 0.0, 0.0]  # paint red where DED set
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"(e) Dark-edge-density set (red)\nvalue = {dark_edge_density:.4f}")

    # Feature dimension bar chart
    dims = {"HOG": 1764, "LBP\nhist": 10, "Intensity\nhist": 16,
            "Edge\nhist": 8, "Stats": 11}
    bars = axes[1, 2].bar(
        list(dims.keys()), list(dims.values()),
        color=["#2E75B6", "#ED7D31", "#70AD47", "#A14B0F", "#7030A0"],
        edgecolor="white",
    )
    for bar, val in zip(bars, dims.values()):
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2, val * 1.05,
            f"{val}", ha="center", fontsize=11, fontweight="bold",
        )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_ylabel("Feature dimension")
    axes[1, 2].set_title(f"(f) Final feature vector\ntotal length = {sum(dims.values())}")
    axes[1, 2].set_ylim(1, 3000)

    for ax in axes.flat[:5]:
        ax.axis("off")

    plt.suptitle("Feature extraction pipeline (HOG + LBP + intensity + edge + dark-edge density)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(OUT / "fig_3_6_feature_extraction.png")
    plt.close()
    print(f"  ✓ Saved {OUT / 'fig_3_6_feature_extraction.png'}")


# ===========================================================================
# Figure 3.7 — Crack-likelihood computation steps on a real image
# ===========================================================================

def build_fig_3_7():
    from skimage import exposure

    _, gray = load_image_pair(EXAMPLE_IMAGES[0][1])

    # Replicate the inner steps of crack_likelihood() for visualisation
    smoothed = filters.gaussian(gray, sigma=0.8, preserve_range=True)
    equalised = exposure.equalize_adapthist(smoothed, clip_limit=0.03)

    dark_response = 1.0 - equalised
    dark_response = (dark_response - dark_response.min()) / (dark_response.max() - dark_response.min() + 1e-9)

    edge_response = filters.sobel(equalised)
    edge_response = (edge_response - edge_response.min()) / (edge_response.max() - edge_response.min() + 1e-9)

    try:
        ridge_response = filters.frangi(dark_response, sigmas=(1, 2, 3), black_ridges=False)
        ridge_response = np.nan_to_num(ridge_response, nan=0.0)
        ridge_response = (ridge_response - ridge_response.min()) / (ridge_response.max() - ridge_response.min() + 1e-9)
    except Exception:
        ridge_response = np.zeros_like(dark_response)

    likelihood = crack_likelihood(gray)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("(a) Input grayscale")

    axes[0, 1].imshow(equalised, cmap="gray")
    axes[0, 1].set_title("(b) CLAHE-equalised\n(clip_limit=0.03)")

    axes[0, 2].imshow(dark_response, cmap="Blues")
    axes[0, 2].set_title("(c) Dark response\n(1 − equalised, normalised)")

    axes[1, 0].imshow(ridge_response, cmap="Greens")
    axes[1, 0].set_title("(d) Frangi ridge response\nσ ∈ {1, 2, 3} px")

    axes[1, 1].imshow(edge_response, cmap="Oranges")
    axes[1, 1].set_title("(e) Sobel edge response")

    im = axes[1, 2].imshow(likelihood, cmap="inferno")
    axes[1, 2].set_title("(f) Crack-likelihood map\n0.60×dark + 0.25×ridge + 0.15×edge")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("Crack-likelihood computation: CLAHE → Frangi → Sobel → weighted sum",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(OUT / "fig_3_7_crack_likelihood.png")
    plt.close()
    print(f"  ✓ Saved {OUT / 'fig_3_7_crack_likelihood.png'}")


# ===========================================================================
# Figure 3.8 — Connected-component filter on a real image
# ===========================================================================

def build_fig_3_8():
    _, gray = load_image_pair(EXAMPLE_IMAGES[0][1])
    likelihood = crack_likelihood(gray)

    # Threshold the likelihood map as in segment_crack
    otsu = float(filters.threshold_otsu(likelihood))
    high_quantile = float(np.quantile(likelihood, 0.93))
    threshold = max(otsu, high_quantile)
    raw_mask = likelihood >= threshold
    # Intersect with dark pixels — matches segment_crack
    dark_pixels = gray <= np.quantile(gray, 0.36)
    raw_mask = np.logical_and(raw_mask, dark_pixels | (likelihood >= np.quantile(likelihood, 0.975)))

    # Get the final filtered mask (after the 5-rule filter inside segment_crack)
    final_mask, _ = segment_crack(
        gray, min_object_size=60, min_component_length=18, min_elongation=1.8,
    )

    labeled_raw = measure.label(raw_mask)
    n_raw_components = labeled_raw.max()
    labeled_final = measure.label(final_mask)
    n_final_components = labeled_final.max()

    # Build a three-panel display: raw mask, accepted-vs-rejected, final
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # (a) Raw thresholded mask — every component in red on the grayscale background
    bg = np.stack([gray * 0.6 + 0.4] * 3, axis=-1)  # lightened background
    raw_display = bg.copy()
    raw_display[raw_mask] = [0.85, 0.15, 0.15]
    axes[0].imshow(raw_display)
    axes[0].set_title(f"(a) Raw thresholded mask\n{n_raw_components} connected components")

    # (b) Classification: components in final mask are green, others red
    final_pixels = final_mask
    cls_display = bg.copy()
    cls_display[raw_mask & ~final_pixels] = [0.85, 0.15, 0.15]  # rejected = red
    cls_display[final_pixels] = [0.20, 0.70, 0.20]              # accepted = green
    n_accepted = n_final_components
    n_rejected = n_raw_components - n_accepted
    axes[1].imshow(cls_display)
    axes[1].set_title(f"(b) 5-rule classification\n{n_accepted} accepted (green), {n_rejected} rejected (red)")

    # (c) Final filtered mask only
    final_display = bg.copy()
    final_display[final_pixels] = [0.10, 0.10, 0.10]
    axes[2].imshow(final_display)
    axes[2].set_title(f"(c) Final filtered crack mask\n{n_final_components} components retained")

    for ax in axes:
        ax.axis("off")

    plt.suptitle("Connected-component filtering with the 5-rule elongation filter",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(OUT / "fig_3_8_cc_filtering.png")
    plt.close()
    print(f"  ✓ Saved {OUT / 'fig_3_8_cc_filtering.png'}")


# ===========================================================================
# Figure 3.9 — Skeletonisation and medial-axis distance transform
# ===========================================================================

def build_fig_3_9():
    _, gray = load_image_pair(EXAMPLE_IMAGES[0][1])
    mask, _ = segment_crack(
        gray, min_object_size=60, min_component_length=18, min_elongation=1.8,
    )
    skeleton = morphology.skeletonize(mask)
    _, distance = morphology.medial_axis(mask, return_distance=True)
    skeleton_widths = np.asarray(distance[skeleton], dtype=np.float32) * 2.0

    # Light grey background of the gray image
    bg = np.stack([gray * 0.6 + 0.4] * 3, axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1.05, 1.4]})

    # (a) Mask
    mask_display = bg.copy()
    mask_display[mask] = [0.1, 0.1, 0.1]
    axes[0].imshow(mask_display)
    axes[0].set_title("(a) Crack mask")

    # (b) Skeleton
    skel_display = bg.copy()
    skel_display[skeleton] = [0.85, 0.65, 0.10]
    axes[1].imshow(skel_display)
    axes[1].set_title(f"(b) Skeletonisation\nlength = {skeleton.sum()} px")

    # (c) Medial-axis distance (colour-coded along skeleton)
    cmap_data = np.zeros_like(gray)
    cmap_data[skeleton] = distance[skeleton]
    axes[2].imshow(bg)
    im = axes[2].imshow(np.ma.masked_where(cmap_data == 0, cmap_data), cmap="plasma")
    axes[2].set_title("(c) Medial-axis distance\n(brighter = wider)")
    plt.colorbar(im, ax=axes[2], shrink=0.85, label="½ width (px)")

    # (d) Width histogram
    if skeleton_widths.size > 0:
        axes[3].hist(skeleton_widths, bins=15, color="#ED7D31", edgecolor="#A14B0F")
        mw = float(skeleton_widths.mean())
        xw = float(skeleton_widths.max())
        axes[3].axvline(mw, color="#2E75B6", linestyle="--", linewidth=2,
                        label=f"Mean = {mw:.1f} px")
        axes[3].axvline(xw, color="#C0504D", linestyle=":", linewidth=2,
                        label=f"Max = {xw:.1f} px")
        axes[3].legend(loc="upper right", fontsize=10)
        axes[3].set_xlabel("Width along skeleton (px)")
        axes[3].set_ylabel("Skeleton pixels")
    axes[3].set_title("(d) Width distribution")
    axes[3].grid(alpha=0.3); axes[3].set_axisbelow(True)

    for ax in axes[:3]:
        ax.axis("off")

    plt.suptitle("Skeletonisation and medial-axis distance-transform width measurement",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(OUT / "fig_3_9_skeletonisation.png")
    plt.close()
    print(f"  ✓ Saved {OUT / 'fig_3_9_skeletonisation.png'}")


# ===========================================================================
# Figure 4.4 — Qualitative results (3 real samples × Original/Overlay/Heatmap/Mask)
# ===========================================================================

def build_fig_4_4():
    from sdnet_pipeline.localization import heatmap_rgb

    samples = [EXAMPLE_IMAGES[i] for i in FIG_4_4_INDICES]
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing works

    col_titles = ["Original", "Polygon overlay", "Heatmap", "Mask"]

    for i, (surface, path, label) in enumerate(samples):
        rgb, gray = load_image_pair(path)
        mask, likelihood = segment_crack(
            gray, min_object_size=60, min_component_length=18, min_elongation=1.8,
        )
        polygons = extract_polygons(mask, max_components=10, max_polygon_points=80)

        # (a) Original
        axes[i, 0].imshow(rgb)

        # (b) Polygon overlay
        overlay = rgb.astype(np.float32) / 255.0
        overlay[mask] = overlay[mask] * 0.35 + np.array([0.90, 0.20, 0.15]) * 0.65
        axes[i, 1].imshow(overlay)
        for poly in polygons:
            pts = np.array(poly["polygon"])
            if len(pts) >= 3:
                axes[i, 1].plot(
                    np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="#FFD93D", linewidth=1.5,
                )

        # (c) Heatmap
        heat = heatmap_rgb(likelihood)
        blended = (rgb.astype(np.float32) * 0.45 + heat.astype(np.float32) * 0.55).astype(np.uint8)
        axes[i, 2].imshow(blended)

        # (d) Mask
        axes[i, 3].imshow(mask, cmap="gray")

        # Row label
        axes[i, 0].set_ylabel(f"Sample {i+1}\n{label}", fontsize=11, fontweight="bold")

        # Column titles only on the top row
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=12, fontweight="bold")

        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Qualitative results: original / polygon overlay / colourised heatmap / binary mask",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.savefig(OUT / "fig_4_4_qualitative.png")
    plt.close()
    print(f"  ✓ Saved {OUT / 'fig_4_4_qualitative.png'}")


# ===========================================================================
# Run everything
# ===========================================================================

if __name__ == "__main__":
    print("Building Group C figures from REAL SDNET2018 images...")
    print(f"Output folder: {OUT.resolve()}")
    print()

    # Sanity-check the EXAMPLE_IMAGES list
    missing = [p for _, p, _ in EXAMPLE_IMAGES if not Path(p).exists()]
    if missing:
        print("ERROR — the following image paths don't exist:")
        for p in missing:
            print(f"   {p}")
        print()
        print("Edit the EXAMPLE_IMAGES list at the top of this script.")
        print("Quick way to find good candidates:")
        print('   uv run python -c "import pandas as pd; df = pd.read_csv(\'data/results/predictions.csv\'); '
              'print(df[df.predicted_label == \'cracked\'].sort_values(\'crack_probability\', ascending=False).'
              'head(20)[[\'image_id\', \'path\', \'surface\']].to_string())"')
        raise SystemExit(1)

    build_fig_1_1()
    build_fig_3_6()
    build_fig_3_7()
    build_fig_3_8()
    build_fig_3_9()
    build_fig_4_4()
    print()
    print(f"All six figures saved to: {OUT.resolve()}")
    print("Drop them into the Word document, replacing the synthetic versions.")
