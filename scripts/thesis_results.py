"""
thesis_results.py  —  regenerate Chapter 4 tables and figures from your own data.

Run from the repo root (sdnet-enhanced):
    uv run --active python scripts/thesis_results.py

It READS the result files your pipeline already produced, PRINTS every table value
to the console (copy these into the thesis tables), and SAVES every figure as a PNG
in a new folder  thesis_figures/.

Nothing is invented: each number is computed from data/results/*.  If a column or
file is named differently in your project, the script prints what it actually found
so you can tell me and I will adjust it.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

RESULTS = Path("data/results")
CNN_META = Path("models/cnn/metrics_cnn.json")
OUT = Path("thesis_figures"); OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Serif"
STEEL, AMBER = "#3E6B7C", "#E0823B"


def hr(title): print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
def find_col(df, *cands):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None
def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- load predictions
pred_path = RESULTS / "predictions.csv"
if not pred_path.exists():
    raise SystemExit(f"Could not find {pred_path}. Run sdnet-infer first, or fix the path.")
df = pd.read_csv(pred_path)
print(f"Loaded {pred_path}  ({len(df):,} rows)")
print("Columns found:", list(df.columns))

# ---- Table 4.2 VALIDATION column, computed from the CURRENT predictions.csv ----
# (Reported alongside the test column; the threshold 0.21 was tuned on validation.)
if "split" in df.columns:
    _vt = find_col(df, "target", "y_true")
    _vp = find_col(df, "predicted_target", "pred_target")
    _vpr = find_col(df, "crack_probability", "crack_prob", "probability", "score")
    _v = df[df["split"].astype(str).str.lower() == "validation"]
    if len(_v) and _vt and _vp:
        _yt = _v[_vt].astype(int); _yp = _v[_vp].astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(_yt, _yp, labels=[0, 1]).ravel()
        _ba = 0.5 * (_tp / max(_tp + _fn, 1) + _tn / max(_tn + _fp, 1))
        if _vpr:
            _f, _t, _ = roc_curve(_yt, _v[_vpr]); _au = auc(_f, _t)
        else:
            _au = float("nan")
        print("\n--- Table 4.2 VALIDATION column (current run, n=%d) ---" % len(_v))
        print(f"  Accuracy {accuracy_score(_yt, _yp):.3f}   Balanced Acc {_ba:.3f}   "
              f"Precision {precision_score(_yt, _yp, zero_division=0):.3f}   "
              f"Recall {recall_score(_yt, _yp, zero_division=0):.3f}   "
              f"F1 {f1_score(_yt, _yp, zero_division=0):.3f}   ROC AUC {_au:.3f}")

# Evaluate the classifier on the HELD-OUT TEST SPLIT only. Computing metrics over
# all rows would include the training data and massively inflate the scores, so
# this filter is essential for an honest, CNN-comparable result.
SPLIT = "test"
if "split" in df.columns:
    before = len(df)
    df = df[df["split"].astype(str).str.lower() == SPLIT].reset_index(drop=True)
    print(f"Filtered to split == '{SPLIT}' for classifier metrics: {len(df):,} of {before:,} rows")
else:
    print("WARNING: no 'split' column found — metrics would include training data!")

c_target = find_col(df, "target", "y_true")
c_pred_t = find_col(df, "predicted_target", "pred_target")
c_prob   = find_col(df, "crack_probability", "crack_prob", "probability", "score")
c_label  = find_col(df, "label", "actual", "actual_label")
c_predl  = find_col(df, "predicted_label", "prediction")
c_surf   = find_col(df, "surface")

# derive 0/1 truth and prediction if only string labels exist
def to01(series): return (series.astype(str).str.lower() == "cracked").astype(int)
y_true = df[c_target].astype(int) if c_target else to01(df[c_label])
y_pred = df[c_pred_t].astype(int) if c_pred_t else to01(df[c_predl])

# ---------------------------------------------------------------- Table 4.2: classifier metrics
hr("TABLE 4.2  Classical classifier performance (held-out test split)")
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
print(f"  n images        : {len(df):,}")
print(f"  Accuracy        : {accuracy_score(y_true, y_pred):.3f}")
print(f"  Precision       : {precision_score(y_true, y_pred, zero_division=0):.3f}")
print(f"  Recall          : {recall_score(y_true, y_pred, zero_division=0):.3f}")
print(f"  F1-score        : {f1_score(y_true, y_pred, zero_division=0):.3f}")
print(f"  Confusion (TP/FP/FN/TN): {tp} / {fp} / {fn} / {tn}")
if c_prob:
    fpr, tpr, _ = roc_curve(y_true, df[c_prob]); roc_auc = auc(fpr, tpr)
    print(f"  ROC AUC         : {roc_auc:.3f}")

# ---------------------------------------------------------------- Figure 4.1: confusion matrices (classical + CNN)
cnn = load_json(CNN_META)
def draw_cm(ax, mat, title, hexc, sub):
    base = tuple(int(hexc[i:i+2], 16) / 255 for i in (1, 3, 5))
    cmap = LinearSegmentedColormap.from_list("c", [(1, 1, 1), base])
    norm = mat / mat.max()
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontsize=18,
                    fontweight="bold", color="white" if norm[i, j] > 0.55 else "#1F2933")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-cracked", "Cracked"]); ax.set_yticklabels(["Non-cracked", "Cracked"], rotation=90, va="center")
    ax.set_xlabel("Predicted", fontweight="bold"); ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.text(0.5, -0.32, sub, transform=ax.transAxes, ha="center", va="top", style="italic", color="#555")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)

classical_cm = np.array([[tn, fp], [fn, tp]])
if cnn:
    cm = cnn["test_at_tuned_threshold"]["confusion_matrix"]
    cnn_cm = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    draw_cm(ax[0], classical_cm, "Classical (Extra-Trees)", STEEL,
            f"F1 {f1_score(y_true, y_pred, zero_division=0):.3f}")
    draw_cm(ax[1], cnn_cm, f"ResNet-18 (\u03c4={cnn['decision_threshold']:.2f})", AMBER,
            f"F1 {cnn['test_at_tuned_threshold']['f1']:.3f}")
else:
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.6)); draw_cm(ax, classical_cm, "Classical (Extra-Trees)", STEEL, "")
fig.tight_layout(); fig.savefig(OUT / "fig_4_1_confusion.png", dpi=300, bbox_inches="tight", facecolor="white")
print("[saved]", OUT / "fig_4_1_confusion.png")

# ---------------------------------------------------------------- Table 4.7: classical vs CNN
if cnn:
    hr("TABLE 4.7  Classical vs fine-tuned ResNet-18")
    t = cnn["test_at_tuned_threshold"]
    print(f"  Classical : acc {accuracy_score(y_true,y_pred):.3f}  P {precision_score(y_true,y_pred,zero_division=0):.3f}  R {recall_score(y_true,y_pred,zero_division=0):.3f}  F1 {f1_score(y_true,y_pred,zero_division=0):.3f}")
    print(f"  ResNet-18 : acc {t['accuracy']:.3f}  P {t['precision']:.3f}  R {t['recall']:.3f}  F1 {t['f1']:.3f}  (\u03c4={cnn['decision_threshold']:.2f})")

# ---------------------------------------------------------------- Figure 4.2: ROC + PR (classical)
if c_prob:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    ax[0].plot(fpr, tpr, color=STEEL, lw=2, label=f"Extra-Trees (AUC = {roc_auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="#aaa", label="Random")
    ax[0].set_xlabel("False Positive Rate"); ax[0].set_ylabel("True Positive Rate"); ax[0].set_title("(a) ROC curve"); ax[0].legend(loc="lower right", fontsize=9)
    prec, rec, _ = precision_recall_curve(y_true, df[c_prob]); ap = average_precision_score(y_true, df[c_prob])
    ax[1].plot(rec, prec, color=AMBER, lw=2, label=f"Extra-Trees (AP = {ap:.3f})")
    ax[1].axhline(y_true.mean(), ls="--", color="#aaa", label=f"Prevalence = {y_true.mean():.3f}")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("(b) Precision-Recall curve"); ax[1].legend(loc="upper right", fontsize=9)
    for a in ax: a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(OUT / "fig_4_2_roc_pr.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("[saved]", OUT / "fig_4_2_roc_pr.png")

# ---------------------------------------------------------------- Table 4.3 + Figure 4.3: per-surface
if c_surf:
    hr("TABLE 4.3  Per-surface precision / recall / F1")
    rows = []
    for surf, g in df.groupby(c_surf):
        yt = y_true[g.index]; yp = y_pred[g.index]
        P = precision_score(yt, yp, zero_division=0); R = recall_score(yt, yp, zero_division=0); F = f1_score(yt, yp, zero_division=0)
        rows.append((surf, P, R, F)); print(f"  {surf:<14} P {P:.3f}  R {R:.3f}  F1 {F:.3f}")
    labels = [r[0] for r in rows]; x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, [r[1] for r in rows], w, label="Precision", color=STEEL)
    ax.bar(x,     [r[2] for r in rows], w, label="Recall", color=AMBER)
    ax.bar(x + w, [r[3] for r in rows], w, label="F1", color="#6E96A3")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_4_3_per_surface.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("[saved]", OUT / "fig_4_3_per_surface.png")

# ---------------------------------------------------------------- Measurements: Table 4.4, Figs 4.5/4.6
loc_path = next((RESULTS / n for n in ["localizations.csv", "localization.csv"] if (RESULTS / n).exists()), None)
mdf = pd.read_csv(loc_path) if loc_path else df
c_area = find_col(mdf, "crack_area_pct", "area_pct", "area")
c_len  = find_col(mdf, "crack_length_px", "length_px", "length")
c_mw   = find_col(mdf, "mean_width_px", "mean_width")
c_xw   = find_col(mdf, "max_width_px", "max_width")
c_sev  = find_col(mdf, "severity_label", "severity")

if c_area:
    sub = mdf[mdf[c_area].notna() & (mdf[c_area] > 0)]
    # crack_area_pct is stored as a fraction (e.g. 0.016); display it as a percentage.
    area_scale = 100.0 if sub[c_area].astype(float).median() < 1.0 else 1.0
    hr("TABLE 4.4  Crack measurement distributions (cracked images)")
    def stats(col, name, scale=1.0):
        v = sub[col].dropna().astype(float) * scale
        print(f"  {name:<12} mean {v.mean():.2f}  median {v.median():.2f}  P95 {v.quantile(.95):.2f}  P99 {v.quantile(.99):.2f}")
    if c_area: stats(c_area, "Area %", area_scale)
    if c_len:  stats(c_len, "Length px")
    if c_mw:   stats(c_mw, "Mean width")
    if c_xw:   stats(c_xw, "Max width")
    print(f"  (n = {len(sub):,} cracked images)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(sub[c_area].dropna().astype(float) * area_scale, bins=40, color=STEEL, edgecolor="white")
    ax[0].set_xlabel("Crack area (% of image)"); ax[0].set_ylabel("Count"); ax[0].set_title("(a) Crack area")
    if c_len:
        ax[1].hist(sub[c_len].dropna().astype(float), bins=40, color=AMBER, edgecolor="white")
        ax[1].set_xlabel("Skeleton length (px)"); ax[1].set_ylabel("Count"); ax[1].set_title("(b) Crack length")
    fig.tight_layout(); fig.savefig(OUT / "fig_4_5_distributions.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("[saved]", OUT / "fig_4_5_distributions.png")

if c_sev:
    hr("FIGURE 4.6  Severity distribution")
    order = ["low", "medium", "high"]
    counts = mdf[c_sev].astype(str).str.lower().value_counts()
    counts = counts.reindex(order).fillna(0).astype(int)
    for k in order: print(f"  {k:<8}: {counts[k]:,}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(order, [counts[k] for k in order], color=["#6E96A3", STEEL, AMBER])
    for i, k in enumerate(order): ax.text(i, counts[k], f"{counts[k]:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Number of cracked images")
    fig.tight_layout(); fig.savefig(OUT / "fig_4_6_severity.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("[saved]", OUT / "fig_4_6_severity.png")

hr("DONE")
print(f"All figures saved in:  {OUT.resolve()}")
print("Copy the printed table values into your thesis, and send me the PNGs to insert.")
