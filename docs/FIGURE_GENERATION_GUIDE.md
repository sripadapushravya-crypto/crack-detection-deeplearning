# How to Regenerate All Your Thesis Figures Yourself

This is your end-to-end guide. Total time: about 90 minutes, mostly waiting.

## What you'll produce

| Group | Figures | Source | Status now |
|---|---|---|---|
| **A** (conceptual diagrams) | 1.2, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5, 4.7 | Built in chat | ✅ Use the files in `figures_complete/` as they are |
| **B** (real-data plots) | 4.1, 4.2, 4.3, 4.5, 4.6 | Built from your real metrics | ✅ Use the files in `figures_updated/` as they are |
| **C** (sample-image figures on REAL SDNET2018) | 1.1, 3.6, 3.7, 3.8, 3.9, 4.4 | **You'll regenerate these now** | 🟡 Run the script in Step 2 below |
| **D** (UI screenshots) | 3.10, 3.11, 4.8 | **You'll capture these now** | 🟡 Follow Step 3 below |

---

## Step 1 — Save the figure-builder script (1 minute)

1. Download `build_thesis_figures.py` from the chat
2. Save it into your project as: `C:\Users\shrav\Downloads\SDNET\SDNET\scripts\build_thesis_figures.py`

---

## Step 2 — Regenerate Group C with real SDNET images (30 minutes)

### 2a. Pick six good cracked images from your dataset

Run this in PowerShell at the project root:

```powershell
uv run python -c "import pandas as pd; df = pd.read_csv('data/results/predictions.csv'); print(df[df.predicted_label == 'cracked'].sort_values('crack_probability', ascending=False).head(20)[['image_id', 'path', 'surface']].to_string())"
```

This shows the 20 most-confidently-cracked images in your test data. **Pick six** — two from each surface (bridge_deck, wall, pavement). Look for images with cracks that are clearly visible in the photograph. The first few in the list are usually the most photogenic.

### 2b. Edit the EXAMPLE_IMAGES list in the script

Open `scripts/build_thesis_figures.py` in your editor. Near the top you'll see:

```python
EXAMPLE_IMAGES = [
    ("Wall (W)",        "data/raw/sdnet2018/W/CW/<file>.jpg",   "Hairline crack on wall"),
    ("Bridge Deck (D)", "data/raw/sdnet2018/D/CD/<file>.jpg",   "Branched crack on bridge deck"),
    ...
]
```

Replace each placeholder path with one of the real paths from step 2a. The third element is the caption that will appear above the panel — write a short descriptive label.

**Example after editing:**

```python
EXAMPLE_IMAGES = [
    ("Wall (W)",        "data/raw/sdnet2018/W/CW/7008-101.jpg",  "Hairline crack on wall"),
    ("Bridge Deck (D)", "data/raw/sdnet2018/D/CD/7008-15.jpg",   "Branched deck crack"),
    ("Pavement (P)",    "data/raw/sdnet2018/P/CP/7008-23.jpg",   "Long thin pavement crack"),
    ("Wall (W)",        "data/raw/sdnet2018/W/CW/7008-104.jpg",  "Crack with surface staining"),
    ("Bridge Deck (D)", "data/raw/sdnet2018/D/CD/7008-44.jpg",   "Crack near joint"),
    ("Pavement (P)",    "data/raw/sdnet2018/P/CP/7008-67.jpg",   "Wide branched crack"),
]
```

(Your actual filenames will be different — use what comes out of the query in step 2a.)

### 2c. Run the script

```powershell
uv run python scripts/build_thesis_figures.py
```

It produces six PNGs in `data/thesis_figures/`:

- `fig_1_1_typical_cracks.png` — your 6-panel grid
- `fig_3_6_feature_extraction.png` — input + HOG + LBP + Sobel + DED + bar chart
- `fig_3_7_crack_likelihood.png` — CLAHE → Frangi → Sobel → weighted sum
- `fig_3_8_cc_filtering.png` — before/after the 5-rule filter
- `fig_3_9_skeletonisation.png` — skeleton + distance transform + width histogram
- `fig_4_4_qualitative.png` — 3 real samples × 4 views (original/overlay/heatmap/mask)

Should take 1-2 minutes total.

### 2d. Inspect them before inserting

Open each PNG and check:

- **Fig 1.1** — all six panels show clearly visible cracks
- **Fig 3.6** — the input image (panel a) is recognisable concrete with a crack; panel (e) shows the red dark-edge-density set tracing the crack body
- **Fig 3.7** — the final crack-likelihood map (panel f) lights up brightly along the crack
- **Fig 3.8** — panel (c) "Final filtered crack mask" actually contains the crack mask, not just background
- **Fig 3.9** — panel (b) shows a skeleton (thin yellow line tracing the crack); panel (c) has colour-coded distance values along it
- **Fig 4.4** — the polygon overlay (column 2) outlines the crack region; the heatmap (column 3) is colourful where the crack is; the mask (column 4) is a clear binary blob

**If any of those don't look right**, the most common cause is that you picked an image whose crack is too subtle for the heuristic to find. Replace that image path with a different one (try `crack_probability > 0.7` in step 2a) and rerun the script.

---

## Step 3 — Capture the three UI screenshots (15 minutes)

### 3a. Start the backend and frontend

In one PowerShell window:
```powershell
cd C:\Users\shrav\Downloads\SDNET\SDNET
uv run sdnet-server --reload
```

In a **second** PowerShell window:
```powershell
cd C:\Users\shrav\Downloads\SDNET\SDNET\frontend
npm run dev
```

Then open Chrome or Edge: **http://localhost:5173**

### 3b. Set up for clean screenshots

- Press **F11** to enter full-screen mode (removes browser UI clutter)
- Press **Ctrl + 0** to make sure browser zoom is 100%
- Take a quick scroll through the whole dashboard to confirm everything loaded

### 3c. Capture Fig 3.10 — Methodology + Performance Radar

**What it should show:** The 8-stage methodology banner *and* the Performance Radar chart visible together in one frame.

1. Scroll the dashboard so the methodology panel (with the 8 stage chips) and the performance radar (the hexagonal chart) are both visible at once
2. Press **Win + Shift + S** (Windows Snipping Tool)
3. Drag a rectangle around both elements — aim for an image roughly 1400 × 900 pixels
4. The screenshot is now in your clipboard. Paste into **Paint** (`Win + R`, type `mspaint`, paste with `Ctrl + V`)
5. Save as PNG: `data/thesis_figures/fig_3_10_dashboard_methodology.png`

### 3d. Capture Fig 3.11 — Predictions table + image detail

**What it should show:** The predictions table on top, and an expanded image-detail view below showing the four tabs (Original / Overlay / Heatmap / Mask) with one of them active.

1. Scroll to the predictions table on the dashboard
2. Click on any predicted-cracked row to expand its image detail
3. Click the **Overlay** tab inside the detail view (this is usually the most visually striking)
4. Snip the entire area: filter row at top, a few visible predictions rows, and the open image-detail view
5. Save as `data/thesis_figures/fig_3_11_dashboard_predictions.png`

**Tip:** Pick a row with `severity_label = "high"` or `"medium"` — those tend to have more visible polygon overlays.

### 3e. Capture Fig 4.8 — Project upload screen

**What it should show:** Your project-upload page after running an upload, with the per-image report visible.

1. Navigate to the "Projects" / "Upload" tab in the dashboard
2. If you haven't uploaded anything yet, upload 5-10 test images (you can use any cracked SDNET images) — it'll process them in a minute or two
3. Once processing completes, you'll see a per-project summary with a thumbnail grid, severity counts, and per-image rows
4. Snip the summary view including the project metadata header and at least 3-4 thumbnails with their severity labels
5. Save as `data/thesis_figures/fig_4_8_project_report.png`

---

## Step 4 — Insert all 24 figures into the thesis (30 minutes)

Open `SDNET_Thesis_Updated.docx` in Word.

For each figure caption in the document:

1. **Find the figure caption** (e.g., "Figure 3.6  Feature extraction pipeline").
2. **Click immediately above the caption line** — that's where the image goes.
3. If there's already an image placeholder there, click it once to select it and press **Delete**.
4. **Insert → Pictures → This Device** and pick the matching PNG. Use this table to find the right file:

| Caption | Source file |
|---|---|
| Figure 1.1 | `data/thesis_figures/fig_1_1_typical_cracks.png` (yours) |
| Figure 1.2 | `figures_complete/fig_1_2_pipeline.png` |
| Figure 2.1 | `figures_complete/fig_2_1_taxonomy.png` |
| Figure 2.2 | `figures_complete/fig_2_2_unet.png` |
| Figure 2.3 | `figures_complete/fig_2_3_arch_compare.png` |
| Figure 3.1 | `figures_complete/fig_3_1_architecture.png` |
| Figure 3.2 | `figures_complete/fig_3_2_dfd.png` |
| Figure 3.3 | `figures_complete/fig_3_3_usecase.png` |
| Figure 3.4 | `figures_complete/fig_3_4_class.png` |
| Figure 3.5 | `figures_complete/fig_3_5_activity.png` |
| Figure 3.6 | `data/thesis_figures/fig_3_6_feature_extraction.png` (yours) |
| Figure 3.7 | `data/thesis_figures/fig_3_7_crack_likelihood.png` (yours) |
| Figure 3.8 | `data/thesis_figures/fig_3_8_cc_filtering.png` (yours) |
| Figure 3.9 | `data/thesis_figures/fig_3_9_skeletonisation.png` (yours) |
| Figure 3.10 | `data/thesis_figures/fig_3_10_dashboard_methodology.png` (yours) |
| Figure 3.11 | `data/thesis_figures/fig_3_11_dashboard_predictions.png` (yours) |
| Figure 4.1 | `figures_updated/fig_4_1_confusion_matrix.png` |
| Figure 4.2 | `figures_updated/fig_4_2_roc_pr_curves.png` |
| Figure 4.3 | `figures_updated/fig_4_3_per_surface.png` |
| Figure 4.4 | `data/thesis_figures/fig_4_4_qualitative.png` (yours) |
| Figure 4.5 | `figures_updated/fig_4_5_distributions.png` |
| Figure 4.6 | `figures_updated/fig_4_6_severity_dist.png` |
| Figure 4.7 | `figures_complete/fig_4_7_performance_radar.png` |
| Figure 4.8 | `data/thesis_figures/fig_4_8_project_report.png` (yours) |

5. **Right-click the inserted image → Wrap Text → In Line with Text.** This ensures the caption stays attached.

6. **Resize if needed:** drag a corner handle so the figure roughly spans the page width (about 6 inches / 15 cm wide). For multi-panel figures like 3.6, 3.7, 4.4, keep them at full width. For small diagrams like 2.3 they can be a bit smaller.

7. Save the document. Repeat for all 24 figures.

---

## Step 5 — Clear the yellow highlights (1 minute)

In Word:

1. Press **Ctrl + A** to select the entire document
2. Click **Home → Text Highlight Color dropdown → No Color**

All the yellow review marks vanish at once.

---

## Step 6 — Final scroll-through (5 minutes)

Scroll the whole document from cover page to last reference. Check:

- Every figure has loaded correctly
- Every figure caption appears directly below its figure
- Page numbers continue smoothly
- The TOC is up to date — right-click anywhere on it → **Update Field → Update entire table**
- The List of Figures and List of Tables are up to date — same process if you have those as auto-fields

---

## Step 7 — Export to PDF for submission (2 minutes)

1. **File → Save As → PDF**
2. Click **Options**
3. Tick **PDF/A compliant** (this embeds all fonts, which JNTUH requires for archival)
4. Save with a final filename like `SDNET_Thesis_<YourName>_<RollNo>_Final.pdf`

That file is what you submit.

---

## What to do if anything goes wrong

**The script crashes on `from sdnet_pipeline.localization import ...`:**
You're not in the project root when running. From PowerShell, `cd C:\Users\shrav\Downloads\SDNET\SDNET` first.

**A figure looks empty / no crack visible:**
The image you picked has a crack that's too subtle for the heuristic. Pick a different image from your top-20 list (try the higher-probability ones).

**Fig 3.8 panel (c) is mostly blank:**
The chosen image's crack didn't survive the 5-rule filter. Use a different image with a clearer crack.

**Fig 4.4 polygon doesn't trace the crack tightly:**
This is normal — the polygon is a simplified contour, not pixel-perfect. As long as it broadly covers the crack region, it's correct.

**The UI screenshots look blurry:**
Use Chrome's built-in DevTools screenshot instead of Snipping Tool. Press F12, then `Ctrl + Shift + P`, type "Capture screenshot" and use "Capture full size screenshot" — produces much higher resolution.

**The dashboard won't load some images:**
Make sure both backend AND frontend are running. The backend serves the image files via API; if it's not running, the dashboard shows placeholder errors.

---

## After you're done

You'll have a polished, defensible thesis PDF with all 24 figures generated from your actual project and data. Every figure has a clear provenance — Group A is your design diagrams, Group B is your real metrics, Group C is your real images run through your real pipeline, and Group D is screenshots of your actually-running deployed application.

If anything breaks during these steps, paste the error message back here and I'll help you debug. Good luck.
