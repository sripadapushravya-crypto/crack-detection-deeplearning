from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from sdnet_pipeline.config import DEFAULT_DATASET_DIR, ensure_data_dirs


SURFACES = {
    "D": "bridge_deck",
    "P": "pavement",
    "W": "wall",
}


def concrete_texture(size: int, rng: random.Random) -> Image.Image:
    base = np.full((size, size), rng.randint(125, 175), dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 999_999)).normal(0, 18, (size, size))
    image = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="L").filter(ImageFilter.GaussianBlur(radius=0.7)).convert("RGB")


def draw_crack(image: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    x = rng.randint(12, width - 12)
    points: list[tuple[int, int]] = []
    segments = rng.randint(5, 9)
    for step in range(segments):
        y = int(step * height / (segments - 1))
        x += rng.randint(-22, 22)
        x = max(5, min(width - 5, x))
        points.append((x, y))
    line_width = rng.choice([1, 1, 2, 2, 3])
    draw.line(points, fill=(35, 35, 35), width=line_width, joint="curve")
    for x, y in points[1:-1:2]:
        branch_len = rng.randint(12, 36)
        angle = rng.uniform(-math.pi * 0.85, -math.pi * 0.15)
        end = (
            int(x + math.cos(angle) * branch_len),
            int(y + math.sin(angle) * branch_len),
        )
        draw.line([(x, y), end], fill=(45, 45, 45), width=1)


def add_obstructions(image: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(rng.randint(1, 4)):
        x0 = rng.randint(0, image.width - 30)
        y0 = rng.randint(0, image.height - 30)
        x1 = x0 + rng.randint(20, 80)
        y1 = y0 + rng.randint(10, 45)
        shade = rng.randint(20, 80)
        draw.rectangle((x0, y0, x1, y1), fill=(shade, shade, shade, rng.randint(18, 38)))


def create_demo_dataset(
    destination: Path,
    images_per_class: int,
    force: bool,
    seed: int,
) -> None:
    if destination.exists() and force:
        import shutil
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ensure_data_dirs()

    for surface_code in SURFACES:
        for class_code, cracked in [("C" + surface_code, True), ("U" + surface_code, False)]:
            folder = destination / surface_code / class_code
            folder.mkdir(parents=True, exist_ok=True)
            for idx in range(images_per_class):
                image = concrete_texture(256, rng)
                if cracked:
                    draw_crack(image, rng)
                if rng.random() < 0.35:
                    add_obstructions(image, rng)
                image.save(
                    folder / f"demo_{surface_code}_{class_code}_{idx:04d}.jpg",
                    quality=92,
                )

    print(f"Demo dataset ready at: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic SDNET-like demo dataset."
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--images-per-class", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_demo_dataset(args.destination, args.images_per_class, args.force, args.seed)


if __name__ == "__main__":
    main()