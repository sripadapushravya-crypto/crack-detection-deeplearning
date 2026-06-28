"""Fine-tune a pretrained CNN on SDNET2018 for crack classification.

This is the deep-learning counterpart to the classical (ExtraTrees +
handcrafted-features) baseline. It deliberately reuses the SAME train/val/test
split stored in data/processed/manifest.csv, so the resulting metrics are
directly comparable to the classical model on the identical test set.

Pipeline:
  - read manifest.csv, split by the `split` column (train/val/test)
  - fine-tune an ImageNet-pretrained backbone (default ResNet18)
  - handle the ~16% cracked class imbalance with class-weighted loss
  - pick the decision threshold that maximises F1 on the VALIDATION set
    (mirrors the classical model's tuned threshold, kept honest by never
    touching the test set during tuning)
  - evaluate on the test set at that threshold AND at the default 0.5
  - save the best checkpoint, a metadata bundle, full metrics, and a per-epoch
    history for the thesis.

Usage (from the repo root, with a CUDA GPU):
    uv run python scripts/train_cnn.py --epochs 10 --batch-size 32

Swap the backbone with e.g. --backbone efficientnet_b0 (lighter, slightly
stronger) or --backbone resnet50 (heavier).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_TO_LABEL = {0: "non_cracked", 1: "cracked"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(row: pd.Series, raw_root: Path | None) -> Path:
    """Return a usable image path.

    Prefers the absolute `path` from the manifest; if that file is missing
    (e.g. the project was copied to a new machine), falls back to
    raw_root / relative_path.
    """
    primary = Path(str(row["path"]))
    if primary.exists():
        return primary
    if raw_root is not None:
        rel = str(row["relative_path"]).replace("\\", "/")
        return raw_root / rel
    return primary


class SDNetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Any, raw_root: Path | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.raw_root = raw_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = resolve_path(row, self.raw_root)
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        target = int(row["target"])
        return image, target


def build_transforms(img_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_model(backbone: str, num_classes: int = 2) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def compute_class_weights(targets: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp) -> float:
    model.train()
    running = 0.0
    seen = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        running += loss.item() * images.size(0)
        seen += images.size(0)
    return running / max(seen, 1)


@torch.no_grad()
def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(cracked)
        probs_all.append(probs.cpu().numpy())
        targets_all.append(targets.numpy())
    return np.concatenate(probs_all), np.concatenate(targets_all)


def metrics_at_threshold(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "n": int(len(targets)),
    }


def best_f1_threshold(probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(targets, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a CNN on SDNET2018.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Fallback image root (joined with relative_path) if absolute paths are missing.",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2"],
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="If the manifest has no 'val' split, carve this stratified fraction out of train.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4, help="Set 0 if Windows worker errors occur.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/cnn"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"Device: {device} | AMP: {use_amp} | backbone: {args.backbone}")

    manifest = pd.read_csv(args.manifest)
    required = {"path", "relative_path", "target", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    print("split distribution:")
    print(manifest["split"].value_counts(dropna=False).to_string())

    split_norm = manifest["split"].astype(str).str.strip().str.lower()
    val_aliases = {"val", "valid", "validation", "dev"}
    test_df = manifest[split_norm == "test"].copy()
    val_df = manifest[split_norm.isin(val_aliases)].copy()
    train_pool = manifest[split_norm == "train"].copy()

    if len(test_df) == 0:
        raise ValueError("No rows with split == 'test'; cannot evaluate.")
    if len(train_pool) == 0:
        raise ValueError("No rows with split == 'train'; cannot train.")

    # Use the dataset's own validation split if present. Only if there is none do
    # we carve one out of TRAIN (stratified by target), always leaving the TEST
    # split untouched so it stays identical to the classical baseline's.
    if len(val_df) == 0:
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            train_pool,
            test_size=args.val_frac,
            stratify=train_pool["target"],
            random_state=args.seed,
        )
        train_df = train_df.copy()
        val_df = val_df.copy()
        print(
            f"No validation split found; carved {len(val_df)} stratified rows "
            f"from train ({args.val_frac:.0%}). Test split untouched."
        )
    else:
        train_df = train_pool
        print(f"Using the manifest's own validation split ({len(val_df)} rows).")

    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = SDNetDataset(train_df, train_tf, args.raw_root)
    val_ds = SDNetDataset(val_df, eval_tf, args.raw_root)
    test_ds = SDNetDataset(test_df, eval_tf, args.raw_root)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    model = build_model(args.backbone).to(device)
    class_weights = compute_class_weights(train_df["target"].to_numpy().astype(int)).to(device)
    print(f"class weights (non_cracked, cracked): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_threshold = 0.5
    history: list[dict[str, Any]] = []
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_probs, val_targets = predict_probs(model, val_loader, device)
        tuned_t, tuned_f1 = best_f1_threshold(val_probs, val_targets)
        val_metrics = metrics_at_threshold(val_probs, val_targets, tuned_t)
        scheduler.step()

        print(
            f"epoch {epoch:02d} | train_loss {train_loss:.4f} | "
            f"val_F1 {val_metrics['f1']:.4f} @ t={tuned_t:.2f} | "
            f"val_recall {val_metrics['recall']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = tuned_t
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ↳ new best val F1; checkpoint saved to {checkpoint_path}")

    # Final evaluation on the test set with the best checkpoint.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_probs, test_targets = predict_probs(model, test_loader, device)
    test_tuned = metrics_at_threshold(test_probs, test_targets, best_threshold)
    test_default = metrics_at_threshold(test_probs, test_targets, 0.5)

    print("\n=== TEST (val-tuned threshold) ===")
    print(json.dumps(test_tuned, indent=2))
    print("\n=== TEST (threshold 0.50) ===")
    print(json.dumps(test_default, indent=2))

    metrics_out = {
        "model_type": f"{args.backbone}_finetuned",
        "backbone": args.backbone,
        "feature_version": f"cnn_{args.backbone}",
        "img_size": args.img_size,
        "decision_threshold": best_threshold,
        "threshold_basis": "val_f1_optimal",
        "test_at_tuned_threshold": test_tuned,
        "test_at_threshold_0p5": test_default,
        "best_val_f1": best_val_f1,
    }
    (args.output_dir / "metrics_cnn.json").write_text(json.dumps(metrics_out, indent=2))
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    bundle = {
        "backbone": args.backbone,
        "num_classes": 2,
        "img_size": args.img_size,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "class_to_label": CLASS_TO_LABEL,
        "decision_threshold": best_threshold,
        "threshold_basis": "val_f1_optimal",
        "checkpoint": str(checkpoint_path.name),
    }
    (args.output_dir / "model_meta.json").write_text(json.dumps(bundle, indent=2))

    print(f"\nSaved: {checkpoint_path}")
    print(f"Saved: {args.output_dir / 'model_meta.json'}")
    print(f"Saved: {args.output_dir / 'metrics_cnn.json'}")
    print(f"Saved: {args.output_dir / 'history.json'}")


if __name__ == "__main__":
    main()
