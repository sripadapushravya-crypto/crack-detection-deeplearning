from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sdnet_pipeline.config import DEFAULT_DATASET_DIR, RAW_DIR, ensure_data_dirs


DATASET_SLUG = "aniruddhsharma/structural-defects-network-concrete-crack-images"


def link_or_copy_dataset(
    source: Path,
    destination: Path,
    copy: bool,
    force: bool,
) -> None:
    if destination.exists() or destination.is_symlink():
        if not force:
            print(f"Dataset destination already exists: {destination}")
            return
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(source, destination)
        print(f"Copied dataset to {destination}")
    else:
        destination.symlink_to(source, target_is_directory=True)
        print(f"Linked dataset to {destination}")


def download_dataset(destination: Path, copy: bool, force: bool) -> Path:
    import kagglehub

    ensure_data_dirs()
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG)).resolve()
    (RAW_DIR / "kaggle_dataset_path.txt").write_text(str(dataset_path), encoding="utf-8")
    link_or_copy_dataset(dataset_path, destination, copy=copy, force=force)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SDNET2018 from Kaggle via kagglehub."
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating a symlink.")
    parser.add_argument("--force", action="store_true", help="Replace an existing destination.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = download_dataset(args.destination, copy=args.copy, force=args.force)
    print(f"Dataset ready at: {destination}")


if __name__ == "__main__":
    main()