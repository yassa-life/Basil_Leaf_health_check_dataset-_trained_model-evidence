"""
Shared basil leaf data pipeline used by parts notebooks and the batch runner.
Each model notebook can also inline this logic; this module avoids six feature extractions when running all models.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
FEATURE_VERSION = "rgb-hsv-lbp-hog-v1"
SOURCE_URL = "https://ieee-dataport.org/open-access/leaves-indias-most-famous-basil-plant-leaves-quality-dataset"
DOI = "10.21227/a4f6-4413"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FOLDERS = {
    "Amravati_Region_Basil_Plant_Healthy": ("Healthy", 31),
    "Nagpur_Region_Basil_Plant_Healthy": ("Healthy", 473),
    "Pune_Region_Basil_Plant_Healthy": ("Healthy", 146),
    "Basil_Plant_Unhealthy": ("Unhealthy", 481),
}
CLASSES = ["Healthy", "Unhealthy"]
Image.MAX_IMAGE_PIXELS = 25_000_000
CAPTURE_GROUPS: dict[str, str] = {}


def find_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "data" / "raw").is_dir() and (path / "Basil_Leaf_ML_Workflow.ipynb").exists():
            return path
    for path in (start, *start.parents):
        if (path / "data" / "raw").is_dir():
            return path
    raise FileNotFoundError("Could not find project root containing data/raw.")


def inside(root: Path, path: Path) -> Path:
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path must stay inside the data directory: {path.name}")
    return path


def read_rgb(source):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(source) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                raise ValueError("Unsupported image format. Use a JPEG, PNG, WebP, BMP or TIFF image.")
            if min(opened.size) < 16:
                raise ValueError("Image is too small; both dimensions must be at least 16 pixels.")
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")


def discover(root: Path) -> pd.DataFrame:
    root = Path(root).resolve()
    rows = []
    for folder, (label, _) in FOLDERS.items():
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rel = inside(root, path).relative_to(root).as_posix()
                rows.append({"path": rel, "label": label, "group": CAPTURE_GROUPS.get(rel, "")})
    if not rows:
        raise FileNotFoundError(
            "No images found. Place the four labelled folders under data/raw "
            "(Amravati/Nagpur/Pune Healthy + Basil_Plant_Unhealthy)."
        )
    return pd.DataFrame(rows)


def inventory_table(data_dir: Path) -> pd.DataFrame:
    rows = []
    for folder, (label, expected) in FOLDERS.items():
        count = sum(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            for p in (data_dir / folder).rglob("*")
        )
        rows.append(
            {
                "folder": folder,
                "class": label,
                "expected": expected,
                "downloaded": count,
                "missing": max(0, expected - count),
                "complete": count == expected,
            }
        )
    return pd.DataFrame(rows)


def audit_dataset(root: Path) -> tuple[pd.DataFrame, dict]:
    frame = discover(root)
    rows, rejected = [], []
    for row in frame.to_dict("records"):
        try:
            p = inside(root, Path(root) / row["path"])
            im = read_rgb(p)
            pixel_hash = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            file_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            gray = np.asarray(im.resize((32, 32)).convert("L"), dtype=float)
            low = dctn(gray, norm="ortho")[:8, :8].ravel()[1:]
            bits = low > np.median(low)
            phash = sum(int(bit) << i for i, bit in enumerate(bits))
            rows.append(
                {
                    **row,
                    "width": im.width,
                    "height": im.height,
                    "sha256": file_hash,
                    "pixel_sha256": pixel_hash,
                    "phash": phash,
                }
            )
        except Exception as exc:
            rejected.append({"path": row["path"], "reason": str(exc)})
    valid = pd.DataFrame(rows)
    if valid.empty:
        raise ValueError("No valid images survived decoding.")
    if set(valid.label) != set(frame.label):
        raise ValueError("An entire class was lost during image validation.")
    conflicts = valid.groupby("pixel_sha256").label.nunique()
    if (conflicts > 1).any():
        raise ValueError("Identical decoded images have conflicting labels.")

    parent = list(range(len(valid)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(b)] = find(a)

    first_hash, first_group = {}, {}
    hashes = valid.phash.tolist()
    for i, row in valid.iterrows():
        for lookup, key in ((first_hash, row.pixel_sha256), (first_group, row.group)):
            if key:
                if key in lookup:
                    union(lookup[key], i)
                else:
                    lookup[key] = i
        for j in range(i):
            if (int(hashes[i]) ^ int(hashes[j])).bit_count() <= 4:
                union(i, j)
    valid["split_group"] = [f"group-{find(i):05d}" for i in range(len(valid))]
    duplicate_count = int(valid.pixel_sha256.duplicated().sum())
    valid = valid.drop_duplicates("pixel_sha256").reset_index(drop=True)
    summary = {
        "source": SOURCE_URL,
        "doi": DOI,
        "input_count": len(frame),
        "valid_unique_count": len(valid),
        "exact_duplicates_removed": duplicate_count,
        "rejected_images": rejected,
        "class_counts": {str(k): int(v) for k, v in valid.label.value_counts().items()},
        "split_groups": int(valid.split_group.nunique()),
    }
    if valid.label.nunique() < 2:
        raise ValueError("At least two verified classes are required.")
    return valid, summary


def make_splits(frame: pd.DataFrame):
    y, groups = frame.label.to_numpy(), frame.split_group.to_numpy()
    group_counts = frame.groupby("label").split_group.nunique()
    if group_counts.min() < 5:
        raise ValueError("Need at least five independent groups per class for holdout and grouped CV.")
    dev, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(frame, y, groups))
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(dev, y[dev], groups[dev]))
    classes = set(y)
    for a, b in [(dev, test)] + [(dev[a], dev[b]) for a, b in cv]:
        if set(groups[a]) & set(groups[b]):
            raise AssertionError("Data leakage: shared groups across a split.")
        if set(y[a]) != classes or set(y[b]) != classes:
            raise ValueError("A split lacks a class.")
        if len(a) < 5:
            raise ValueError("Each training fold needs at least five examples.")
    return dev, test, cv


def handcrafted(image) -> np.ndarray:
    rgb = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    hsv = rgb2hsv(rgb)
    parts = []
    for array in (rgb, hsv):
        for channel in range(3):
            h, _ = np.histogram(array[:, :, channel], bins=16, range=(0, 1))
            parts.append(h.astype(np.float32) / h.sum())
        parts.extend([array.mean(axis=(0, 1)), array.std(axis=(0, 1))])
    gray = (rgb2gray(rgb) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    h, _ = np.histogram(lbp, bins=np.arange(11), density=False)
    parts.append(h.astype(np.float32) / h.sum())
    small = np.asarray(image.resize((64, 64)).convert("L"), dtype=np.float32) / 255
    parts.append(hog(small, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
    return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)


def extract_features(frame: pd.DataFrame, root: Path, batch_size: int = 16) -> np.ndarray:
    vectors = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        images = [read_rgb(Path(root) / p) for p in batch.path]
        vectors.extend(handcrafted(im) for im in images)
        print(f"Features: {min(start + batch_size, len(frame))}/{len(frame)}", flush=True)
    return np.asarray(vectors, dtype=np.float32)


def prepare_dataset(root: Path | None = None):
    root = find_root(root) if root is None or not (Path(root) / "data" / "raw").is_dir() else Path(root).resolve()
    data_dir = root / "data" / "raw"
    inv = inventory_table(data_dir)
    if inv.downloaded.sum() == 0:
        raise FileNotFoundError(f"No images under {data_dir}")
    if not inv.groupby("class").downloaded.sum().gt(0).all():
        raise ValueError("Both Healthy and Unhealthy classes are required.")
    frame, audit = audit_dataset(data_dir)
    audit["download_coverage"] = {
        "partial_dataset": not bool(inv.complete.all()),
        "expected_images": int(inv.expected.sum()),
        "downloaded_images": int(inv.downloaded.sum()),
        "missing_images": int(inv.missing.sum()),
        "folders": inv.to_dict("records"),
    }
    dev, test, cv = make_splits(frame)
    X = extract_features(frame, data_dir)
    split = np.full(len(frame), "development", dtype=object)
    split[test] = "test"
    manifest = frame.copy()
    manifest["split"] = split
    return {
        "root": root,
        "data_dir": data_dir,
        "inventory": inv,
        "frame": frame,
        "manifest": manifest,
        "audit": audit,
        "dev": dev,
        "test": test,
        "cv": cv,
        "X": X,
        "y": frame.label.to_numpy(),
        "X_tr": X[dev],
        "y_tr": frame.label.to_numpy()[dev],
        "X_te": X[test],
        "y_te": frame.label.to_numpy()[test],
    }
