"""Image loading and OCR-oriented preprocessing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class LoadedImage:
    image: np.ndarray
    width: int
    height: int
    source: str  # pil|opencv


def load_image_robust(path: str | Path) -> LoadedImage:
    """Load image with PIL-first, OpenCV fallback.

    Raises:
        RuntimeError: when image cannot be read by either backend.
    """
    p = str(path)

    # PIL first for robust format handling.
    try:
        with Image.open(p) as img:
            rgb = img.convert("RGB")
            arr = np.array(rgb)
        h, w = arr.shape[:2]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return LoadedImage(image=bgr, width=w, height=h, source="pil")
    except Exception:
        pass

    # OpenCV fallback.
    arr = cv2.imread(p, cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"Unreadable image: {p}")
    h, w = arr.shape[:2]
    return LoadedImage(image=arr, width=w, height=h, source="opencv")


def resize_preserve_aspect(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return image, 1.0
    scale = float(max_dim) / float(current_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized, scale


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)


def _otsu(gray: np.ndarray) -> np.ndarray:
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr


def _adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def _deskew(gray_or_bin: np.ndarray) -> np.ndarray:
    """Lightweight deskew using min-area rectangle on foreground pixels.

    If no stable angle can be estimated, image is returned unchanged.
    """
    if gray_or_bin.ndim != 2:
        return gray_or_bin

    # Invert so text tends to be white foreground for coordinate extraction.
    inv = cv2.bitwise_not(gray_or_bin)
    coords = cv2.findNonZero(inv)
    if coords is None or len(coords) < 10:
        return gray_or_bin

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ignore tiny angles to avoid introducing artifacts.
    if abs(angle) < 0.2:
        return gray_or_bin

    h, w = gray_or_bin.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_or_bin,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_for_ocr(
    image_bgr: np.ndarray,
    mode: str = "adaptive",
    resize_max_dim: int = 1800,
    enable_grayscale: bool = True,
    enable_denoise: bool = True,
    enable_deskew: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply configurable preprocessing pipeline for OCR."""
    processed = image_bgr.copy()
    processed, scale = resize_preserve_aspect(processed, resize_max_dim)

    meta: Dict[str, Any] = {
        "mode": mode,
        "resize_max_dim": resize_max_dim,
        "resize_scale": scale,
        "grayscale_applied": False,
        "denoise_applied": False,
        "deskew_applied": False,
    }

    gray = _to_gray(processed) if enable_grayscale or mode != "none" else processed
    if isinstance(gray, np.ndarray) and gray.ndim == 2:
        meta["grayscale_applied"] = True

    if mode in {"denoise", "adaptive_denoise"} or enable_denoise:
        gray = _denoise(gray)
        meta["denoise_applied"] = True

    if mode == "none":
        out = gray
    elif mode == "gray":
        out = gray
    elif mode == "otsu":
        out = _otsu(gray)
    elif mode in {"adaptive", "adaptive_denoise", "denoise"}:
        out = _adaptive(gray)
    else:
        # conservative fallback
        out = _adaptive(gray)
        meta["mode_fallback"] = "adaptive"

    if enable_deskew:
        out = _deskew(out)
        meta["deskew_applied"] = True

    return out, meta
