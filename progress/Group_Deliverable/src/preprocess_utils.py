"""
Shared helpers for Progress Review I preprocessing / EDA notebooks.
Works from Group_Deliverable/notebooks or Group_Deliverable root.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

SEED = 42
SOURCE_URL = (
    "https://ieee-dataport.org/open-access/"
    "leaves-indias-most-famous-basil-plant-leaves-quality-dataset"
)
DOI = "10.21227/a4f6-4413"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FOLDERS = {
    "Amravati_Region_Basil_Plant_Healthy": ("Healthy", "Amravati", 31),
    "Nagpur_Region_Basil_Plant_Healthy": ("Healthy", "Nagpur", 473),
    "Pune_Region_Basil_Plant_Healthy": ("Healthy", "Pune", 146),
    "Basil_Plant_Unhealthy": ("Unhealthy", "Unknown", 481),
}
CLASSES = ["Healthy", "Unhealthy"]
Image.MAX_IMAGE_PIXELS = 25_000_000


def find_deliverable_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "notebooks").is_dir() and (path / "results").is_dir() and (path / "data").is_dir():
            return path
        if (path / "Group_Deliverable" / "notebooks").is_dir():
            return path / "Group_Deliverable"
    raise FileNotFoundError("Could not locate Group_Deliverable root.")


def paths(start: Path | None = None) -> dict[str, Path]:
    root = find_deliverable_root(start)
    return {
        "root": root,
        "raw": root / "data" / "raw",
        "external": root / "data" / "external",
        "viz": root / "results" / "eda_visualizations",
        "logs": root / "results" / "logs",
        "outputs": root / "results" / "outputs",
        "notebooks": root / "notebooks",
    }


def read_rgb(source) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(source) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                raise ValueError(f"Unsupported format: {opened.format}")
            if min(opened.size) < 16:
                raise ValueError("Image too small (<16 px on a side).")
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")


def inventory_table(data_dir: Path) -> pd.DataFrame:
    rows = []
    for folder, (label, region, expected) in FOLDERS.items():
        folder_path = data_dir / folder
        count = 0
        if folder_path.is_dir():
            count = sum(
                p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                for p in folder_path.rglob("*")
            )
        rows.append(
            {
                "folder": folder,
                "label": label,
                "region": region,
                "expected": expected,
                "found": count,
                "missing_count": max(0, expected - count),
                "extra_count": max(0, count - expected),
                "complete": count == expected,
            }
        )
    return pd.DataFrame(rows)


def discover_images(data_dir: Path) -> pd.DataFrame:
    rows = []
    for folder, (label, region, _) in FOLDERS.items():
        folder_path = data_dir / folder
        if not folder_path.is_dir():
            continue
        for path in sorted(folder_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rel = path.relative_to(data_dir).as_posix()
                rows.append(
                    {
                        "path": rel,
                        "folder": folder,
                        "label": label,
                        "region": region,
                        "filename": path.name,
                        "extension": path.suffix.lower(),
                        "file_bytes": path.stat().st_size,
                    }
                )
    if not rows:
        raise FileNotFoundError(
            "No images found under data/raw. Place the four labelled folders there."
        )
    return pd.DataFrame(rows)


def audit_images(data_dir: Path, frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid_df, rejected_df). Missing/corrupt reads go to rejected."""
    frame = discover_images(data_dir) if frame is None else frame
    valid, rejected = [], []
    for row in frame.to_dict("records"):
        abs_path = data_dir / row["path"]
        try:
            if not abs_path.exists():
                raise FileNotFoundError("File missing on disk")
            im = read_rgb(abs_path)
            pixel_hash = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            file_hash = hashlib.sha256(abs_path.read_bytes()).hexdigest()
            valid.append(
                {
                    **row,
                    "width": im.width,
                    "height": im.height,
                    "aspect_ratio": im.width / im.height,
                    "pixels": im.width * im.height,
                    "sha256": file_hash,
                    "pixel_sha256": pixel_hash,
                    "mean_r": float(np.asarray(im)[:, :, 0].mean()),
                    "mean_g": float(np.asarray(im)[:, :, 1].mean()),
                    "mean_b": float(np.asarray(im)[:, :, 2].mean()),
                }
            )
        except Exception as exc:
            rejected.append({**row, "reason": str(exc)})
    return pd.DataFrame(valid), pd.DataFrame(rejected)


def remove_exact_duplicates(valid: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(valid)
    cleaned = valid.drop_duplicates("pixel_sha256").reset_index(drop=True)
    return cleaned, before - len(cleaned)


def iqr_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return (series >= low) & (series <= high)


def simple_color_features(image: Image.Image, size: int = 64) -> np.ndarray:
    """Fast RGB+HSV histogram features for Progress Review demos."""
    from skimage.color import rgb2hsv

    rgb = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    hsv = rgb2hsv(rgb)
    parts = []
    for array in (rgb, hsv):
        for channel in range(3):
            hist, _ = np.histogram(array[:, :, channel], bins=8, range=(0, 1))
            parts.append(hist.astype(np.float32) / (hist.sum() + 1e-8))
        parts.append(array.mean(axis=(0, 1)))
        parts.append(array.std(axis=(0, 1)))
    return np.concatenate(parts).astype(np.float32)


def stratified_sample(frame: pd.DataFrame, n_per_class: int, seed: int = SEED) -> pd.DataFrame:
    """Sample up to n_per_class rows from each label without pandas groupby-apply warnings."""
    parts = [
        group.sample(n=min(len(group), n_per_class), random_state=seed)
        for _, group in frame.groupby("label")
    ]
    return pd.concat(parts, ignore_index=True)


def extract_feature_matrix(
    data_dir: Path,
    frame: pd.DataFrame,
    max_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    subset = frame if max_images is None else frame.head(max_images).copy()
    vectors, labels, kept = [], [], []
    for row in subset.to_dict("records"):
        try:
            im = read_rgb(data_dir / row["path"])
            vectors.append(simple_color_features(im))
            labels.append(row["label"])
            kept.append(row)
        except Exception:
            continue
    if not vectors:
        raise ValueError("Feature extraction produced no vectors.")
    return np.vstack(vectors), np.array(labels), pd.DataFrame(kept)
