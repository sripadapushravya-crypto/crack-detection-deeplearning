from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from sdnet_pipeline.config import IMAGE_EXTENSIONS


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def infer_label(path: Path) -> str | None:
    tokens = [normalize_token(part) for part in path.parts]
    compact = " ".join(tokens)

    negative_tokens = {
        "u", "ud", "up", "uw",
        "uncracked", "non_cracked", "noncracked",
        "no_crack", "nocrack", "negative",
    }
    positive_tokens = {"c", "cd", "cp", "cw", "cracked", "crack", "positive"}

    if any(token in negative_tokens for token in tokens):
        return "non_cracked"
    if "non_cracked" in compact or "noncracked" in compact or "uncracked" in compact:
        return "non_cracked"
    if any(token in positive_tokens for token in tokens):
        return "cracked"
    if "crack" in compact:
        return "cracked"
    return None


def infer_surface(path: Path) -> str:
    tokens = [normalize_token(part) for part in path.parts]
    if any(token in {"d", "deck", "decks", "bridge_deck", "bridge_decks", "bridge"} for token in tokens):
        return "bridge_deck"
    if any(token in {"p", "pavement", "pavements", "road", "sidewalk"} for token in tokens):
        return "pavement"
    if any(token in {"w", "wall", "walls"} for token in tokens):
        return "wall"
    return "unknown"


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None