"""Shared OCR engine with cache-first execution and structured outputs.

Public API (import from this module):
- check_tesseract_installation
- ocr_document
- ocr_batch
- load_ocr_result
- load_ocr_text
- load_ocr_words
- load_ocr_lines
- load_ocr_blocks
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output
from tqdm import tqdm

from .config import OCRConfig
from .image_preprocessing import load_image_robust, preprocess_for_ocr
from .utils import append_jsonl, ensure_dir, read_json, setup_logger, utc_now_iso, write_json

LOGGER = setup_logger("ocr_engine")


def check_tesseract_installation(tesseract_cmd: Optional[str] = None, verbose: bool = True) -> bool:
    """Check that Tesseract is installed and reachable.

    Returns False with explicit guidance instead of failing silently.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    cmd = shutil.which("tesseract")
    try:
        version = pytesseract.get_tesseract_version()
        if verbose:
            LOGGER.info("Tesseract available: %s | path=%s", version, cmd)
        return True
    except Exception:
        if verbose:
            LOGGER.error("Tesseract is not available. OCR cannot run.")
            LOGGER.error("Install guidance:")
            LOGGER.error("- macOS: brew install tesseract")
            LOGGER.error("- Ubuntu/Debian: sudo apt-get install -y tesseract-ocr")
            LOGGER.error("- Windows: install Tesseract and add to PATH")
        return False


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _build_ocr_config_string(cfg: OCRConfig) -> str:
    base = f"--oem {cfg.oem} --psm {cfg.psm}"
    if cfg.extra_config:
        base = f"{base} {cfg.extra_config}".strip()
    return base


def _cache_paths(doc_id: str, cfg: OCRConfig) -> Dict[str, Path]:
    cache_root = Path(cfg.cache_dir)
    raw_dir = ensure_dir(cache_root / "raw")
    parsed_dir = ensure_dir(cache_root / "parsed")
    text_dir = ensure_dir(cache_root / "text")
    logs_dir = ensure_dir(cache_root / "logs")
    _ = logs_dir

    return {
        "raw": raw_dir / f"{doc_id}.csv",
        "parsed": parsed_dir / f"{doc_id}.json",
        "text": text_dir / f"{doc_id}.txt",
        "diag": ensure_dir(cfg.diagnostics_dir) / f"{doc_id}.png",
    }


def _prepare_words_df(raw_df: pd.DataFrame, img_w: int, img_h: int, min_confidence: float) -> pd.DataFrame:
    df = _coerce_numeric(
        raw_df,
        [
            "left",
            "top",
            "width",
            "height",
            "conf",
            "page_num",
            "block_num",
            "par_num",
            "line_num",
            "word_num",
        ],
    )

    df["text"] = df.get("text", "").fillna("").astype(str)
    df["text"] = df["text"].map(_normalize_text)

    # keep only non-empty word-like rows after cleanup
    words = df[(df["text"] != "") & (df["conf"].fillna(-1) >= min_confidence)].copy()
    words = words.reset_index(drop=True)
    words["word_id"] = words.index.astype(int)

    words["left"] = words["left"].fillna(0).astype(int)
    words["top"] = words["top"].fillna(0).astype(int)
    words["width"] = words["width"].fillna(0).astype(int)
    words["height"] = words["height"].fillna(0).astype(int)
    words["right"] = words["left"] + words["width"]
    words["bottom"] = words["top"] + words["height"]

    safe_w = max(img_w, 1)
    safe_h = max(img_h, 1)
    words["norm_left"] = words["left"] / safe_w
    words["norm_top"] = words["top"] / safe_h
    words["norm_width"] = words["width"] / safe_w
    words["norm_height"] = words["height"] / safe_h

    key_cols = ["page_num", "block_num", "par_num", "line_num", "left", "top", "word_num", "word_id"]
    for c in ["page_num", "block_num", "par_num", "line_num", "word_num"]:
        if c in words.columns:
            words[c] = words[c].fillna(0).astype(int)

    words = words.sort_values(key_cols).reset_index(drop=True)
    words["word_id"] = words.index.astype(int)
    return words


def _reconstruct_lines(words_df: pd.DataFrame) -> pd.DataFrame:
    if words_df.empty:
        return pd.DataFrame(columns=[
            "line_id",
            "text",
            "left",
            "top",
            "width",
            "height",
            "right",
            "bottom",
            "page_num",
            "block_num",
            "par_num",
            "line_num",
            "word_ids",
        ])

    group_cols = ["page_num", "block_num", "par_num", "line_num"]
    records: List[Dict[str, Any]] = []

    grouped = words_df.groupby(group_cols, dropna=False, sort=True)
    for idx, (keys, g) in enumerate(grouped):
        g = g.sort_values(["left", "top", "word_id"])
        text = " ".join(g["text"].tolist()).strip()
        left = int(g["left"].min())
        top = int(g["top"].min())
        right = int(g["right"].max())
        bottom = int(g["bottom"].max())

        records.append(
            {
                "line_id": idx,
                "text": text,
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
                "right": right,
                "bottom": bottom,
                "page_num": int(keys[0]),
                "block_num": int(keys[1]),
                "par_num": int(keys[2]),
                "line_num": int(keys[3]),
                "word_ids": g["word_id"].astype(int).tolist(),
            }
        )

    return pd.DataFrame.from_records(records)


def _reconstruct_blocks(lines_df: pd.DataFrame) -> pd.DataFrame:
    if lines_df.empty:
        return pd.DataFrame(columns=[
            "block_id",
            "text",
            "left",
            "top",
            "width",
            "height",
            "right",
            "bottom",
            "page_num",
            "block_num",
        ])

    group_cols = ["page_num", "block_num"]
    records: List[Dict[str, Any]] = []

    grouped = lines_df.groupby(group_cols, dropna=False, sort=True)
    for idx, (keys, g) in enumerate(grouped):
        g = g.sort_values(["top", "left", "line_id"])
        text = "\n".join([t for t in g["text"].tolist() if t]).strip()
        left = int(g["left"].min())
        top = int(g["top"].min())
        right = int(g["right"].max())
        bottom = int(g["bottom"].max())

        records.append(
            {
                "block_id": idx,
                "text": text,
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
                "right": right,
                "bottom": bottom,
                "page_num": int(keys[0]),
                "block_num": int(keys[1]),
            }
        )

    return pd.DataFrame.from_records(records)


def _words_to_records(words_df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = [
        "word_id",
        "text",
        "conf",
        "left",
        "top",
        "width",
        "height",
        "right",
        "bottom",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "norm_left",
        "norm_top",
        "norm_width",
        "norm_height",
    ]
    if words_df.empty:
        return []
    return words_df[cols].to_dict(orient="records")


def _lines_to_records(lines_df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = [
        "line_id",
        "text",
        "left",
        "top",
        "width",
        "height",
        "right",
        "bottom",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_ids",
    ]
    if lines_df.empty:
        return []
    return lines_df[cols].to_dict(orient="records")


def _blocks_to_records(blocks_df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = [
        "block_id",
        "text",
        "left",
        "top",
        "width",
        "height",
        "right",
        "bottom",
        "page_num",
        "block_num",
    ]
    if blocks_df.empty:
        return []
    return blocks_df[cols].to_dict(orient="records")


def _quality_score(avg_conf: float, nonempty_word_ratio: float, num_words: int) -> float:
    # bounded heuristic in [0,1]
    conf_part = np.clip(avg_conf / 100.0, 0.0, 1.0)
    density_part = np.clip(nonempty_word_ratio, 0.0, 1.0)
    volume_part = np.clip(np.log1p(max(num_words, 0)) / np.log1p(200.0), 0.0, 1.0)
    return float(0.55 * conf_part + 0.25 * density_part + 0.20 * volume_part)


def _draw_diagnostics(image_gray_or_bin: np.ndarray, words_df: pd.DataFrame, lines_df: pd.DataFrame, blocks_df: pd.DataFrame, save_path: Path) -> None:
    if image_gray_or_bin.ndim == 2:
        canvas = cv2.cvtColor(image_gray_or_bin, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image_gray_or_bin.copy()

    for _, row in blocks_df.iterrows():
        cv2.rectangle(canvas, (int(row.left), int(row.top)), (int(row.right), int(row.bottom)), (255, 0, 0), 2)
    for _, row in lines_df.iterrows():
        cv2.rectangle(canvas, (int(row.left), int(row.top)), (int(row.right), int(row.bottom)), (0, 165, 255), 1)
    for _, row in words_df.iterrows():
        cv2.rectangle(canvas, (int(row.left), int(row.top)), (int(row.right), int(row.bottom)), (0, 255, 0), 1)

    ensure_dir(save_path.parent)
    cv2.imwrite(str(save_path), canvas)


def _run_tesseract_dataframe(image: np.ndarray, cfg: OCRConfig) -> pd.DataFrame:
    tesseract_cfg = _build_ocr_config_string(cfg)
    data = pytesseract.image_to_data(
        image,
        lang=cfg.lang,
        config=tesseract_cfg,
        output_type=Output.DATAFRAME,
    )
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    return data


def _run_tesseract_string(image: np.ndarray, cfg: OCRConfig) -> str:
    tesseract_cfg = _build_ocr_config_string(cfg)
    text = pytesseract.image_to_string(image, lang=cfg.lang, config=tesseract_cfg)
    return _normalize_text(text)


def _full_text_from_lines(lines_df: pd.DataFrame) -> str:
    if lines_df.empty:
        return ""
    sorted_lines = lines_df.sort_values(["page_num", "block_num", "par_num", "line_num", "top", "left"])
    return "\n".join([t for t in sorted_lines["text"].tolist() if t]).strip()


def ocr_document(
    doc_id: str,
    source_path: str | Path,
    cfg: Optional[OCRConfig] = None,
    force: bool = False,
    save_diagnostics: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run OCR for one document with cache-first behavior.

    Cache contract:
    - raw table:   data/interim/ocr/raw/{doc_id}.csv
    - parsed json: data/interim/ocr/parsed/{doc_id}.json
    - full text:   data/interim/ocr/text/{doc_id}.txt
    """
    cfg = cfg or OCRConfig()
    if cfg.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

    if not check_tesseract_installation(cfg.tesseract_cmd, verbose=False):
        raise RuntimeError("Tesseract is not installed or not accessible.")

    paths = _cache_paths(doc_id, cfg)
    diag_enabled = cfg.diagnostics_enabled if save_diagnostics is None else save_diagnostics

    if paths["parsed"].exists() and not force:
        cached = read_json(paths["parsed"])
        cached["cache_hit"] = True
        return cached

    start = time.perf_counter()
    src = Path(source_path)

    try:
        loaded = load_image_robust(src)
        pre_img, prep_meta = preprocess_for_ocr(
            loaded.image,
            mode=cfg.preprocess_mode,
            resize_max_dim=cfg.resize_max_dim,
            enable_grayscale=cfg.enable_grayscale,
            enable_denoise=cfg.enable_denoise,
            enable_deskew=cfg.enable_deskew,
        )

        raw_df = _run_tesseract_dataframe(pre_img, cfg)
        ensure_dir(paths["raw"].parent)
        raw_df.to_csv(paths["raw"], index=False)

        words_df = _prepare_words_df(raw_df, loaded.width, loaded.height, cfg.min_confidence)
        lines_df = _reconstruct_lines(words_df)
        blocks_df = _reconstruct_blocks(lines_df)

        full_text = _full_text_from_lines(lines_df)
        if not full_text:
            # fallback plain text extraction for edge cases.
            full_text = _run_tesseract_string(pre_img, cfg)
        ensure_dir(paths["text"].parent)
        paths["text"].write_text(full_text, encoding="utf-8")

        avg_conf = float(words_df["conf"].mean()) if not words_df.empty else 0.0
        nonempty_ratio = float((raw_df.get("text", "").fillna("").astype(str).str.strip() != "").mean()) if not raw_df.empty else 0.0

        runtime_sec = float(time.perf_counter() - start)
        stats = {
            "num_words": int(words_df.shape[0]),
            "num_lines": int(lines_df.shape[0]),
            "num_blocks": int(blocks_df.shape[0]),
            "avg_word_conf": avg_conf,
            "nonempty_word_ratio": nonempty_ratio,
            "ocr_quality_score": _quality_score(avg_conf, nonempty_ratio, int(words_df.shape[0])),
            "runtime_sec": runtime_sec,
        }

        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "source_path": str(src),
            "image_width": int(loaded.width),
            "image_height": int(loaded.height),
            "loaded_via": loaded.source,
            "preprocessing": prep_meta,
            "ocr_config": {
                "lang": cfg.lang,
                "psm": cfg.psm,
                "oem": cfg.oem,
                "extra_config": cfg.extra_config,
                "min_confidence": cfg.min_confidence,
            },
            "full_text": full_text,
            "words": _words_to_records(words_df),
            "lines": _lines_to_records(lines_df),
            "blocks": _blocks_to_records(blocks_df),
            "stats": stats,
            "cache_hit": False,
            "generated_at_utc": utc_now_iso(),
        }

        write_json(paths["parsed"], result)

        if diag_enabled:
            _draw_diagnostics(pre_img, words_df, lines_df, blocks_df, paths["diag"])

        return result

    except Exception as err:
        failure = {
            "doc_id": doc_id,
            "source_path": str(src),
            "error": str(err),
            "timestamp_utc": utc_now_iso(),
        }
        append_jsonl(cfg.failure_log_path, failure)
        raise


def ocr_batch(
    metadata_df: pd.DataFrame,
    cfg: Optional[OCRConfig] = None,
    force: bool = False,
    save_diagnostics: Optional[bool] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run OCR over a metadata table with caching and failure summary.

    Expected columns: doc_id, file_path (or source_path), optional split/class_name.
    """
    cfg = cfg or OCRConfig()
    if "doc_id" not in metadata_df.columns:
        raise ValueError("metadata_df must contain 'doc_id'")
    if "file_path" not in metadata_df.columns and "source_path" not in metadata_df.columns:
        raise ValueError("metadata_df must contain 'file_path' or 'source_path'")

    rows = metadata_df.to_dict(orient="records")
    iterator = tqdm(rows, desc="OCR batch", disable=not show_progress)

    summaries: List[Dict[str, Any]] = []

    for row in iterator:
        doc_id = str(row["doc_id"])
        source_path = row.get("file_path", row.get("source_path"))
        split = row.get("split")
        class_name = row.get("class_name")

        start = time.perf_counter()
        status = "ok"
        error_msg = None

        try:
            result = ocr_document(
                doc_id=doc_id,
                source_path=source_path,
                cfg=cfg,
                force=force,
                save_diagnostics=save_diagnostics,
            )
            cache_hit = bool(result.get("cache_hit", False))
            num_words = int(result.get("stats", {}).get("num_words", 0))
            avg_conf = float(result.get("stats", {}).get("avg_word_conf", 0.0))
        except Exception as err:
            status = "failed"
            error_msg = str(err)
            cache_hit = False
            num_words = 0
            avg_conf = 0.0

        runtime_sec = float(time.perf_counter() - start)
        summaries.append(
            {
                "doc_id": doc_id,
                "file_path": source_path,
                "split": split,
                "class_name": class_name,
                "status": status,
                "error": error_msg,
                "runtime_sec": runtime_sec,
                "num_words": num_words,
                "avg_conf": avg_conf,
                "cache_hit": cache_hit,
            }
        )

    return pd.DataFrame(summaries)


def load_ocr_result(doc_id: str, cfg: Optional[OCRConfig] = None) -> Dict[str, Any]:
    cfg = cfg or OCRConfig()
    paths = _cache_paths(doc_id, cfg)
    if not paths["parsed"].exists():
        raise FileNotFoundError(f"Parsed OCR cache missing for doc_id={doc_id}: {paths['parsed']}")
    return read_json(paths["parsed"])


def load_ocr_text(doc_id: str, cfg: Optional[OCRConfig] = None) -> str:
    cfg = cfg or OCRConfig()
    paths = _cache_paths(doc_id, cfg)
    if paths["text"].exists():
        return paths["text"].read_text(encoding="utf-8")
    result = load_ocr_result(doc_id, cfg)
    return str(result.get("full_text", ""))


def load_ocr_words(doc_id: str, cfg: Optional[OCRConfig] = None) -> pd.DataFrame:
    result = load_ocr_result(doc_id, cfg)
    return pd.DataFrame(result.get("words", []))


def load_ocr_lines(doc_id: str, cfg: Optional[OCRConfig] = None) -> pd.DataFrame:
    result = load_ocr_result(doc_id, cfg)
    return pd.DataFrame(result.get("lines", []))


def load_ocr_blocks(doc_id: str, cfg: Optional[OCRConfig] = None) -> pd.DataFrame:
    result = load_ocr_result(doc_id, cfg)
    return pd.DataFrame(result.get("blocks", []))
