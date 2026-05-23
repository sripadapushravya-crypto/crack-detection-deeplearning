from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sdnet_pipeline.config import DEFAULT_DATASET_DIR, DEFAULT_MANIFEST, ensure_data_dirs
from sdnet_pipeline.utils import image_size, infer_label, infer_surface, is_image, utc_now_iso, write_json


def discover_images(dataset_dir: Path, inspect_dimensions: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    image_paths = sorted(path for path in dataset_dir.rglob("*") if is_image(path))
    for idx, path in enumerate(tqdm(image_paths, desc="Scanning images")):
        label = infer_label(path.relative_to(dataset_dir))
        width, height = image_size(path) if inspect_dimensions else (None, None)
        rows.append(
            {
                "image_id": f"img_{idx:07d}",
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(dataset_dir)),
                "label": label,
                "target": 1 if label == "cracked" else 0 if label == "non_cracked" else None,
                "surface": infer_surface(path.relative_to(dataset_dir)),
                "source_folder": str(path.parent.relative_to(dataset_dir)),
                "width": width,
                "height": height,
            }
        )
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["split"] = "unassigned"
    labeled = df[df["target"].notna()].copy()

    if len(labeled) < 10 or labeled["target"].nunique() < 2:
        df.loc[labeled.index, "split"] = "train"
        return df

    stratify = labeled["surface"].astype(str) + "_" + labeled["target"].astype(int).astype(str)
    if stratify.value_counts().min() < 2:
        stratify = labeled["target"].astype(int)

    train_idx, temp_idx = train_test_split(
        labeled.index,
        test_size=0.30,
        random_state=seed,
        stratify=stratify,
    )
    temp = labeled.loc[temp_idx]
    temp_stratify = temp["surface"].astype(str) + "_" + temp["target"].astype(int).astype(str)
    if temp_stratify.value_counts().min() < 2:
        temp_stratify = temp["target"].astype(int)

    if temp_stratify.value_counts().min() < 2 or len(temp) < 4:
        val_idx = temp.index[: len(temp) // 2]
        test_idx = temp.index[len(temp) // 2 :]
    else:
        val_idx, test_idx = train_test_split(
            temp.index,
            test_size=0.50,
            random_state=seed,
            stratify=temp_stratify,
        )

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "validation"
    df.loc[test_idx, "split"] = "test"
    return df


def build_manifest(
    dataset_dir: Path,
    output: Path,
    inspect_dimensions: bool,
    seed: int,
) -> pd.DataFrame:
    ensure_data_dirs()
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Run scripts/download_data.sh or scripts/make_demo_data.sh first."
        )
    df = discover_images(dataset_dir, inspect_dimensions=inspect_dimensions)
    if df.empty:
        raise RuntimeError(f"No images found under {dataset_dir}")

    df = assign_splits(df, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    summary = {
        "created_at": utc_now_iso(),
        "dataset_dir": str(dataset_dir.resolve()),
        "rows": int(len(df)),
        "labels": df["label"].fillna("unknown").value_counts().to_dict(),
        "surfaces": df["surface"].value_counts().to_dict(),
        "splits": df["split"].value_counts().to_dict(),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an SDNET image manifest.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-dimensions", action="store_true", help="Skip reading image sizes.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_manifest(
        args.dataset_dir,
        args.output,
        inspect_dimensions=not args.no_dimensions,
        seed=args.seed,
    )
    print(f"Wrote manifest with {len(df):,} rows to {args.output}")
    print(f"Labels  : {df['label'].fillna('unknown').value_counts().to_dict()}")
    print(f"Surfaces: {df['surface'].value_counts().to_dict()}")
    print(f"Splits  : {df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()