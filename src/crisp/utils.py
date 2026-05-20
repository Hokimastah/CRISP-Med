from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dcm"
}


def list_images(folder: str, extensions: Sequence[str] | None = None) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    exts = {e.lower() for e in (extensions or IMAGE_EXTENSIONS)}
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(images)


def infer_label_from_parent(image_path: str | Path) -> str:
    path = Path(image_path)
    if path.parent.name == "":
        raise ValueError(f"Cannot infer label from path: {image_path}")
    return path.parent.name


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
