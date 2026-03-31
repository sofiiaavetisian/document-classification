"""Layout and geometry feature engineering from shared OCR outputs."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import OCRConfig
from .ocr_engine import load_ocr_result

CURRENCY_RE = re.compile(r"^(?:[$€£¥₹])?\s*\d+[\d,\.]*$")
DATE_TOKEN_RE = re.compile(
    r"^(?:\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{4}[\-/]\d{1,2}[\-/]\d{1,2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})$"
)
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
URL_RE = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
PHONE_RE = re.compile(r"^(?:\+?\d[\d\s().-]{6,}\d)$")

ANCHORS = {
    "invoice": ["invoice", "total", "due date", "bill to", "amount due"],
    "budget": ["budget", "forecast", "planned", "allocated", "cost center", "variance"],
    "resume": ["education", "experience", "skills", "objective", "employment"],
    "email": ["subject", "from", "to", "cc", "dear", "regards"],
    "form": ["name", "address", "date", "signature", "checkbox", "application", "form"],
}


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _to_df(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(records)) if records is not None else pd.DataFrame()


def _prepare_words(words: pd.DataFrame, image_w: int, image_h: int) -> pd.DataFrame:
    if words.empty:
        return words

    out = words.copy()
    for c in ["left", "top", "width", "height", "right", "bottom", "conf", "norm_left", "norm_top", "norm_width", "norm_height"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "text" in out.columns:
        out["text"] = out["text"].fillna("").astype(str)
    else:
        out["text"] = ""

    # Recompute normalized coordinates when missing.
    if "norm_left" not in out.columns or out["norm_left"].isna().all():
        out["norm_left"] = _safe_div_series(out.get("left", 0), max(image_w, 1))
    if "norm_top" not in out.columns or out["norm_top"].isna().all():
        out["norm_top"] = _safe_div_series(out.get("top", 0), max(image_h, 1))
    if "norm_width" not in out.columns or out["norm_width"].isna().all():
        out["norm_width"] = _safe_div_series(out.get("width", 0), max(image_w, 1))
    if "norm_height" not in out.columns or out["norm_height"].isna().all():
        out["norm_height"] = _safe_div_series(out.get("height", 0), max(image_h, 1))

    return out


def _safe_div_series(series: Any, denom: float) -> pd.Series:
    s = pd.Series(series)
    if denom == 0:
        return pd.Series(np.zeros(len(s)))
    return pd.to_numeric(s, errors="coerce").fillna(0.0) / float(denom)


def _prepare_lines(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return lines
    out = lines.copy()
    for c in ["left", "top", "width", "height", "right", "bottom"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["text"] = out.get("text", "").fillna("").astype(str)
    return out


def _token_flags(tokens: List[str]) -> Dict[str, List[int]]:
    out = {
        "numeric": [],
        "alpha": [],
        "alnum": [],
        "upper": [],
        "punct_heavy": [],
        "short": [],
        "long": [],
        "currency": [],
        "date_like": [],
        "email_like": [],
        "url_like": [],
        "phone_like": [],
    }

    for tok in tokens:
        clean = tok.strip()
        if not clean:
            for k in out:
                out[k].append(0)
            continue

        punct_count = sum(1 for ch in clean if not ch.isalnum() and not ch.isspace())
        alpha_count = sum(1 for ch in clean if ch.isalpha())

        out["numeric"].append(int(clean.isdigit()))
        out["alpha"].append(int(clean.isalpha()))
        out["alnum"].append(int(clean.isalnum()))
        out["upper"].append(int(clean.isupper() and len(clean) > 1))
        out["punct_heavy"].append(int(_safe_div(punct_count, len(clean)) >= 0.3))
        out["short"].append(int(len(clean) <= 2))
        out["long"].append(int(len(clean) >= 12))
        out["currency"].append(int(bool(CURRENCY_RE.match(clean))))
        out["date_like"].append(int(bool(DATE_TOKEN_RE.match(clean))))
        out["email_like"].append(int(bool(EMAIL_RE.match(clean))))
        out["url_like"].append(int(bool(URL_RE.match(clean))))
        out["phone_like"].append(int(bool(PHONE_RE.match(clean))))

    return out


def _zone_density(words: pd.DataFrame, zone: str) -> float:
    if words.empty:
        return 0.0
    x = words["norm_left"].fillna(0.0)
    y = words["norm_top"].fillna(0.0)

    if zone == "top":
        mask = y < 0.33
    elif zone == "bottom":
        mask = y > 0.66
    elif zone == "left":
        mask = x < 0.5
    elif zone == "right":
        mask = x >= 0.5
    elif zone == "center":
        mask = (x >= 0.33) & (x <= 0.66) & (y >= 0.33) & (y <= 0.66)
    elif zone == "tl":
        mask = (x < 0.5) & (y < 0.5)
    elif zone == "tr":
        mask = (x >= 0.5) & (y < 0.5)
    elif zone == "bl":
        mask = (x < 0.5) & (y >= 0.5)
    elif zone == "br":
        mask = (x >= 0.5) & (y >= 0.5)
    else:
        mask = pd.Series(np.zeros(len(words), dtype=bool))

    return _safe_div(mask.sum(), len(words))


def _line_structure_features(lines_df: pd.DataFrame) -> Dict[str, float]:
    if lines_df.empty:
        return {
            "avg_line_length_chars": 0.0,
            "std_line_length_chars": 0.0,
            "avg_line_width": 0.0,
            "avg_line_height": 0.0,
            "left_edge_alignment_std": 0.0,
            "right_edge_alignment_std": 0.0,
            "center_alignment_std": 0.0,
            "short_line_ratio": 0.0,
            "long_line_ratio": 0.0,
            "uppercase_heavy_line_ratio": 0.0,
            "numeric_heavy_line_ratio": 0.0,
            "colon_ending_line_count": 0.0,
            "label_like_line_count": 0.0,
        }

    lengths = lines_df["text"].astype(str).str.len()
    widths = pd.to_numeric(lines_df.get("width", 0), errors="coerce").fillna(0)
    heights = pd.to_numeric(lines_df.get("height", 0), errors="coerce").fillna(0)
    lefts = pd.to_numeric(lines_df.get("left", 0), errors="coerce").fillna(0)
    rights = pd.to_numeric(lines_df.get("right", 0), errors="coerce").fillna(0)
    centers = (lefts + rights) / 2.0

    texts = lines_df["text"].astype(str)
    upper_ratio = texts.map(lambda t: _safe_div(sum(c.isupper() for c in t), max(len(t), 1)))
    digit_ratio = texts.map(lambda t: _safe_div(sum(c.isdigit() for c in t), max(len(t), 1)))

    colon_end = texts.str.endswith(":").sum()
    label_like = texts.str.contains(r"^[A-Za-z][A-Za-z\s]{0,30}:", regex=True).sum()

    return {
        "avg_line_length_chars": float(lengths.mean()),
        "std_line_length_chars": float(lengths.std(ddof=0)),
        "avg_line_width": float(widths.mean()),
        "avg_line_height": float(heights.mean()),
        "left_edge_alignment_std": float(lefts.std(ddof=0)),
        "right_edge_alignment_std": float(rights.std(ddof=0)),
        "center_alignment_std": float(centers.std(ddof=0)),
        "short_line_ratio": float((lengths <= 15).mean()),
        "long_line_ratio": float((lengths >= 80).mean()),
        "uppercase_heavy_line_ratio": float((upper_ratio >= 0.6).mean()),
        "numeric_heavy_line_ratio": float((digit_ratio >= 0.4).mean()),
        "colon_ending_line_count": float(colon_end),
        "label_like_line_count": float(label_like),
    }


def _table_form_heuristics(words_df: pd.DataFrame, lines_df: pd.DataFrame) -> Dict[str, float]:
    if words_df.empty:
        return {
            "aligned_numeric_column_count_estimate": 0.0,
            "table_like_line_ratio": 0.0,
            "repeated_x_alignment_count": 0.0,
            "repeated_colon_label_ratio": 0.0,
            "blank_space_ratio_approx": 1.0,
            "short_label_count": 0.0,
            "isolated_field_like_count": 0.0,
            "row_consistency_score": 0.0,
            "header_block_prominence": 0.0,
            "footer_block_prominence": 0.0,
        }

    # Numeric column alignment heuristic.
    num_words = words_df[words_df["text"].str.contains(r"\d", regex=True, na=False)].copy()
    bins = np.round(num_words["norm_left"].fillna(0) * 20).astype(int)
    bin_counts = bins.value_counts()
    aligned_numeric_cols = float((bin_counts >= 3).sum())

    # Repeated x-alignment for all words.
    xbins = np.round(words_df["norm_left"].fillna(0) * 20).astype(int)
    repeated_x = float((xbins.value_counts() >= 5).sum())

    # Per-line table-like signal.
    if lines_df.empty:
        table_like_ratio = 0.0
        colon_label_ratio = 0.0
        short_label_count = 0.0
        row_consistency_score = 0.0
    else:
        line_texts = lines_df["text"].astype(str)
        num_heavy = line_texts.map(lambda t: _safe_div(sum(ch.isdigit() for ch in t), max(len(t), 1)) > 0.25)
        has_multi_spaces = line_texts.str.contains(r"\s{2,}", regex=True)
        table_like_ratio = float((num_heavy | has_multi_spaces).mean())

        colon_lines = line_texts.str.contains(":", regex=False)
        colon_label_ratio = float(colon_lines.mean())

        short_label_count = float(
            line_texts.str.match(r"^[A-Za-z][A-Za-z\s]{0,25}:$", na=False).sum()
        )

        # Row consistency from words-per-line variability.
        if "word_ids" in lines_df.columns:
            wp_line = lines_df["word_ids"].map(lambda x: len(x) if isinstance(x, list) else 0)
        else:
            wp_line = line_texts.str.split().map(len)

        mean_wp = float(wp_line.mean())
        std_wp = float(wp_line.std(ddof=0))
        row_consistency_score = _safe_div(mean_wp, (mean_wp + std_wp + 1e-6))

    # Blank-space approximation via occupied word box area.
    occupied = (words_df["norm_width"].fillna(0) * words_df["norm_height"].fillna(0)).sum()
    blank_space = float(np.clip(1.0 - occupied, 0.0, 1.0))

    # Field-like words: short alpha tokens near left margin.
    isolated_field_like_count = float(
        (
            (words_df["text"].str.len() <= 15)
            & (words_df["text"].str.contains(r"^[A-Za-z][A-Za-z\s]*:?$", regex=True, na=False))
            & (words_df["norm_left"].fillna(1.0) < 0.35)
        ).sum()
    )

    header_prom = _zone_density(words_df, "top")
    footer_prom = _zone_density(words_df, "bottom")

    return {
        "aligned_numeric_column_count_estimate": aligned_numeric_cols,
        "table_like_line_ratio": table_like_ratio,
        "repeated_x_alignment_count": repeated_x,
        "repeated_colon_label_ratio": colon_label_ratio,
        "blank_space_ratio_approx": blank_space,
        "short_label_count": short_label_count,
        "isolated_field_like_count": isolated_field_like_count,
        "row_consistency_score": row_consistency_score,
        "header_block_prominence": header_prom,
        "footer_block_prominence": footer_prom,
    }


def _class_anchor_counts(full_text: str) -> Dict[str, float]:
    text = full_text.lower()
    feats: Dict[str, float] = {}
    for cls, terms in ANCHORS.items():
        count = 0
        for t in terms:
            count += text.count(t)
        feats[f"{cls}_anchor_count"] = float(count)
    return feats


def extract_layout_features_for_doc(
    doc_id: str,
    ocr_result: Dict[str, Any],
    image_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract one-row layout feature dict for a document.

    Parameters
    ----------
    doc_id : str
        Document id.
    ocr_result : dict
        Canonical OCR result from `src.ocr_engine.load_ocr_result`.
    image_meta : dict, optional
        Optional fallback metadata (width/height etc.).
    """
    image_meta = image_meta or {}
    image_w = int(ocr_result.get("image_width") or image_meta.get("width") or 1)
    image_h = int(ocr_result.get("image_height") or image_meta.get("height") or 1)

    words_df = _prepare_words(_to_df(ocr_result.get("words", [])), image_w, image_h)
    lines_df = _prepare_lines(_to_df(ocr_result.get("lines", [])))
    blocks_df = _to_df(ocr_result.get("blocks", []))

    tokens = words_df["text"].astype(str).tolist() if not words_df.empty else []
    token_flags = _token_flags(tokens)
    n_tokens = len(tokens)

    basic = {
        "doc_id": doc_id,
        "num_words": float(len(words_df)),
        "num_nonempty_words": float(sum(1 for t in tokens if t.strip())),
        "num_lines": float(len(lines_df)),
        "num_blocks": float(len(blocks_df)),
        "avg_word_confidence": float(pd.to_numeric(words_df.get("conf", 0), errors="coerce").fillna(0).mean()) if not words_df.empty else 0.0,
        "median_word_confidence": float(pd.to_numeric(words_df.get("conf", 0), errors="coerce").fillna(0).median()) if not words_df.empty else 0.0,
        "low_confidence_word_ratio": float((pd.to_numeric(words_df.get("conf", 0), errors="coerce").fillna(0) < 50).mean()) if not words_df.empty else 0.0,
        "avg_words_per_line": _safe_div(len(words_df), max(len(lines_df), 1)),
        "avg_chars_per_word": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "avg_chars_per_line": float(lines_df["text"].astype(str).str.len().mean()) if not lines_df.empty else 0.0,
    }

    token_ratio = {
        "numeric_token_ratio": _safe_div(sum(token_flags["numeric"]), n_tokens),
        "alphabetic_token_ratio": _safe_div(sum(token_flags["alpha"]), n_tokens),
        "alphanumeric_token_ratio": _safe_div(sum(token_flags["alnum"]), n_tokens),
        "uppercase_token_ratio": _safe_div(sum(token_flags["upper"]), n_tokens),
        "punctuation_heavy_token_ratio": _safe_div(sum(token_flags["punct_heavy"]), n_tokens),
        "short_token_ratio": _safe_div(sum(token_flags["short"]), n_tokens),
        "long_token_ratio": _safe_div(sum(token_flags["long"]), n_tokens),
        "currency_token_count": float(sum(token_flags["currency"])),
        "currency_token_ratio": _safe_div(sum(token_flags["currency"]), n_tokens),
        "date_like_token_count": float(sum(token_flags["date_like"])),
        "date_like_token_ratio": _safe_div(sum(token_flags["date_like"]), n_tokens),
        "email_like_token_count": float(sum(token_flags["email_like"])),
        "url_like_token_count": float(sum(token_flags["url_like"])),
        "phone_like_token_count": float(sum(token_flags["phone_like"])),
    }

    zonal = {
        "top_region_text_density": _zone_density(words_df, "top"),
        "bottom_region_text_density": _zone_density(words_df, "bottom"),
        "left_region_text_density": _zone_density(words_df, "left"),
        "right_region_text_density": _zone_density(words_df, "right"),
        "center_region_text_density": _zone_density(words_df, "center"),
        "top_left_density": _zone_density(words_df, "tl"),
        "top_right_density": _zone_density(words_df, "tr"),
        "bottom_left_density": _zone_density(words_df, "bl"),
        "bottom_right_density": _zone_density(words_df, "br"),
    }

    line_feats = _line_structure_features(lines_df)
    table_feats = _table_form_heuristics(words_df, lines_df)

    image_feats = {
        "image_width": float(image_w),
        "image_height": float(image_h),
        "aspect_ratio": _safe_div(image_w, max(image_h, 1)),
        "area": float(image_w * image_h),
        "portrait_vs_landscape_flag": float(1 if image_h >= image_w else 0),
    }

    anchors = _class_anchor_counts(str(ocr_result.get("full_text", "")))

    out: Dict[str, Any] = {}
    out.update(basic)
    out.update(token_ratio)
    out.update(zonal)
    out.update(line_feats)
    out.update(table_feats)
    out.update(image_feats)
    out.update(anchors)
    return out


def build_layout_feature_table(
    metadata_df: pd.DataFrame,
    ocr_loader_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    cfg: Optional[OCRConfig] = None,
    show_progress: bool = True,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Build one-row-per-document layout feature table.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Must include `doc_id`. `width` and `height` are optional fallback metadata.
    ocr_loader_fn : callable, optional
        Function like `load_ocr_result(doc_id, cfg=...)`. If omitted, defaults to shared loader.
    cfg : OCRConfig, optional
        Passed to default loader.
    save_path : path, optional
        If provided, saves resulting feature table as CSV.
    """
    if "doc_id" not in metadata_df.columns:
        raise ValueError("metadata_df must contain 'doc_id'")

    if ocr_loader_fn is None:
        def _loader(doc_id: str) -> Dict[str, Any]:
            return load_ocr_result(doc_id, cfg=cfg)
        ocr_loader_fn = _loader

    rows: List[Dict[str, Any]] = []
    iterator = tqdm(metadata_df.to_dict(orient="records"), desc="Layout features", disable=not show_progress)

    for row in iterator:
        doc_id = str(row["doc_id"])
        image_meta = {"width": row.get("width"), "height": row.get("height")}
        try:
            ocr = ocr_loader_fn(doc_id)
            feats = extract_layout_features_for_doc(doc_id, ocr, image_meta=image_meta)
        except Exception:
            # robust fallback row for unreadable/missing OCR
            feats = {"doc_id": doc_id}
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Ensure stable order and numeric typing for non-id columns.
    if "doc_id" in df.columns:
        ordered_cols = ["doc_id"] + [c for c in df.columns if c != "doc_id"]
        df = df[ordered_cols]

    for c in df.columns:
        if c != "doc_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    return df
