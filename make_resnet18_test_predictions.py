#!/usr/bin/env python3
"""
Generate per-image ResNet-18 predictions on the SDNET2018 TEST split, and print
overall test metrics as a sanity check against the thesis headline
(expected ~ accuracy 0.95, F1 0.82 at threshold 0.81).

Output CSV columns: image_id, surface, target, crack_probability, predicted_label
Run from anywhere (paths are absolute):  python make_resnet18_test_predictions.py
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ----- paths confirmed from your machine -----------------------------------
ROOT       = Path(r"C:\Users\shrav\Downloads\SDNET\sdnet-enhanced")
MANIFEST   = ROOT / "data" / "processed" / "manifest.csv"
CHECKPOINT = ROOT / "models" / "cnn" / "best_model.pt"
OUT_CSV    = ROOT / "resnet18_test_predictions.csv"
THRESHOLD  = 0.81          # ResNet-18 F1-optimal tau (NOT the 0.21 used for Extra Trees)
IMAGE_SIZE = 224
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Inspect confirmed: plain ResNet-18 state_dict, fc -> 2 classes.
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(state)
model.to(device).eval()
print("Checkpoint loaded cleanly.")

# NOTE: these must match the eval transforms used during TRAINING. If the
# sanity-check metrics below do not reproduce the thesis (~0.95 acc / 0.82 F1),
# the transform is the thing to fix -- paste your training transform and I'll align it.
tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
])

manifest = pd.read_csv(MANIFEST)
test = manifest[manifest["split"] == "test"].reset_index(drop=True)
print(f"Test images: {len(test)}  (expecting ~8,414)")

probs = []
with torch.no_grad():
    for i, row in test.iterrows():
        img_path = Path(str(row["path"]))
        if not img_path.is_absolute():
            img_path = ROOT / img_path        # resolve relative manifest paths
        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        probs.append(F.softmax(model(x), dim=1)[0, 1].item())  # P(cracked)
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(test)}")

test["crack_probability"] = probs
test["predicted_label"] = test["crack_probability"].ge(THRESHOLD).map(
    {True: "cracked", False: "non_cracked"}
)

# ---- overall sanity check (should reproduce the thesis headline) ----------
y_true = test["target"].astype(int).to_numpy()
y_pred = test["crack_probability"].ge(THRESHOLD).astype(int).to_numpy()
tp = int(((y_pred == 1) & (y_true == 1)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum())
acc  = (tp + tn) / max(len(y_true), 1)
prec = tp / max(tp + fp, 1)
rec  = tp / max(tp + fn, 1)
f1   = 2 * prec * rec / max(prec + rec, 1e-9)
print("\n--- OVERALL TEST METRICS @ tau=0.81  (sanity check vs thesis ~0.95 acc / 0.82 F1) ---")
print(f"  accuracy : {acc:.3f}")
print(f"  precision: {prec:.3f}")
print(f"  recall   : {rec:.3f}")
print(f"  F1       : {f1:.3f}")
print(f"  TP/FP/FN/TN: {tp}/{fp}/{fn}/{tn}")

cols = ["image_id", "surface", "target", "crack_probability", "predicted_label"]
test[cols].to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}  ({len(test)} rows)")
