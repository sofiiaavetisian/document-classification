"""Classical visual feature extraction for document classification.

Features are intentionally lightweight and deterministic:
- HOG descriptor (OpenCV)
- LBP histogram (uniform coding, implementation-local)
- Horizontal/vertical projection summaries
- Connected-component style density cues
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class VisualFeatureConfig:
    """Configuration for handcrafted visual features."""

    resize_height: int = 256
    resize_width: int = 192
    hog_orientations: int = 9
    hog_cell_size: int = 8
    hog_block_size: int = 2
    lbp_radius: int = 1
    lbp_points: int = 8
    projection_bins: int = 16
    threshold_value: int = 180


def _safe_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if not p.exists():
        raise FileNotFoundError(f"Image path not found: {p}")
    return p


def _load_gray(path_like: str | Path) -> np.ndarray:
    p = _safe_path(path_like)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"OpenCV failed to load image: {p}")
    return img


def _resize(img: np.ndarray, cfg: VisualFeatureConfig) -> np.ndarray:
    return cv2.resize(
        img,
        (cfg.resize_width, cfg.resize_height),
        interpolation=cv2.INTER_AREA,
    )


def _compute_hog(img_gray: np.ndarray, cfg: VisualFeatureConfig) -> np.ndarray:
    win_size = (cfg.resize_width, cfg.resize_height)
    cell = (cfg.hog_cell_size, cfg.hog_cell_size)
    block = (cfg.hog_block_size * cfg.hog_cell_size, cfg.hog_block_size * cfg.hog_cell_size)
    block_stride = cell
    nbins = int(cfg.hog_orientations)

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block,
        _blockStride=block_stride,
        _cellSize=cell,
        _nbins=nbins,
    )
    feat = hog.compute(img_gray)
    if feat is None:
        return np.zeros((0,), dtype=np.float32)
    return feat.reshape(-1).astype(np.float32)


def _lbp_image(img_gray: np.ndarray, points: int, radius: int) -> np.ndarray:
    """Compute non-rotation-invariant LBP image with simple bilinear sampling."""
    h, w = img_gray.shape[:2]
    img_f = img_gray.astype(np.float32)
    out = np.zeros((h, w), dtype=np.uint16)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    center = img_f

    for p in range(points):
        theta = 2.0 * np.pi * p / points
        y = yy - radius * np.sin(theta)
        x = xx + radius * np.cos(theta)

        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        sample = (
            wa * img_f[y0, x0]
            + wb * img_f[y0, x1]
            + wc * img_f[y1, x0]
            + wd * img_f[y1, x1]
        )
        out |= ((sample >= center).astype(np.uint16) << p)

    return out


def _uniform_lbp_hist(lbp_img: np.ndarray, points: int) -> np.ndarray:
    n_bins = points + 2
    hist = np.zeros(n_bins, dtype=np.float32)
    flat = lbp_img.ravel().astype(np.uint16)

    for code in flat:
        bits = [(code >> i) & 1 for i in range(points)]
        transitions = 0
        for i in range(points):
            transitions += int(bits[i] != bits[(i + 1) % points])
        if transitions <= 2:
            hist[int(sum(bits))] += 1.0
        else:
            hist[-1] += 1.0

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _projection_features(img_bin: np.ndarray, bins: int) -> np.ndarray:
    rows = (img_bin > 0).mean(axis=1)
    cols = (img_bin > 0).mean(axis=0)

    def _reduce(arr: np.ndarray, n_bins: int) -> np.ndarray:
        if arr.size == 0:
            return np.zeros(n_bins, dtype=np.float32)
        chunks = np.array_split(arr, n_bins)
        return np.array([float(np.mean(c)) if c.size else 0.0 for c in chunks], dtype=np.float32)

    return np.concatenate([_reduce(rows, bins), _reduce(cols, bins)], axis=0)


def _component_density_features(img_bin: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_bin, connectivity=8)
    if n_labels <= 1:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # Ignore background at index 0.
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    aspect = np.divide(widths, np.maximum(heights, 1.0))

    h, w = img_bin.shape[:2]
    page_area = float(max(h * w, 1))
    fg_ratio = float((img_bin > 0).mean())

    return np.array(
        [
            float(len(areas)),
            float(np.mean(areas) / page_area),
            float(np.median(aspect)),
            fg_ratio,
        ],
        dtype=np.float32,
    )


def extract_visual_feature_vector(
    image_path: str | Path,
    cfg: VisualFeatureConfig | None = None,
) -> np.ndarray:
    cfg = cfg or VisualFeatureConfig()
    img = _load_gray(image_path)
    img = _resize(img, cfg)

    hog = _compute_hog(img, cfg)
    lbp_img = _lbp_image(img, points=cfg.lbp_points, radius=cfg.lbp_radius)
    lbp_hist = _uniform_lbp_hist(lbp_img, points=cfg.lbp_points)

    _, img_bin = cv2.threshold(img, cfg.threshold_value, 255, cv2.THRESH_BINARY_INV)
    proj = _projection_features(img_bin, bins=cfg.projection_bins)
    cc = _component_density_features(img_bin)

    return np.concatenate([hog, lbp_hist, proj, cc], axis=0).astype(np.float32)


def build_visual_feature_table(
    metadata_df: pd.DataFrame,
    image_col: str = "file_path",
    doc_id_col: str = "doc_id",
    cfg: VisualFeatureConfig | None = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a dense visual feature table keyed by doc_id.

    Returns
    -------
    (features_df, failed_doc_ids)
    """
    if doc_id_col not in metadata_df.columns:
        raise ValueError(f"metadata_df must contain '{doc_id_col}'")
    if image_col not in metadata_df.columns:
        raise ValueError(f"metadata_df must contain '{image_col}'")

    cfg = cfg or VisualFeatureConfig()
    records: List[Dict[str, float]] = []
    failed: List[str] = []

    rows = metadata_df.to_dict(orient="records")
    iterator: Iterable[Dict[str, object]] = tqdm(rows, desc="Visual features", disable=not show_progress)
    feature_length: int | None = None

    for row in iterator:
        doc_id = str(row[doc_id_col])
        try:
            vec = extract_visual_feature_vector(row[image_col], cfg=cfg)
            if feature_length is None:
                feature_length = int(vec.shape[0])
            rec: Dict[str, float] = {"doc_id": doc_id}
            for i, value in enumerate(vec):
                rec[f"visual_{i:05d}"] = float(value)
            records.append(rec)
        except Exception:
            failed.append(doc_id)
            continue

    if not records:
        return pd.DataFrame(columns=["doc_id"]), failed

    df = pd.DataFrame(records)
    ordered_cols = ["doc_id"] + sorted([c for c in df.columns if c != "doc_id"])
    df = df[ordered_cols]
    return df, failed


def align_feature_tables(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    key: str = "doc_id",
) -> pd.DataFrame:
    """Left-join helper with deterministic row order."""
    if key not in base_df.columns:
        raise ValueError(f"base_df must contain '{key}'")
    if key not in feature_df.columns:
        raise ValueError(f"feature_df must contain '{key}'")

    out = base_df[[key]].merge(feature_df, on=key, how="left")
    return out
