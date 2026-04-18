"""
hybrid_field_extractor.py
=========================
Production-grade hybrid extractor for invoice fields.

Design
------
1. Run one-stage and two-stage extraction in parallel.
2. Merge field-by-field with fixed routing policy:
   - INVOICE_NUMBER: two-stage preferred, fallback one-stage
   - INVOICE_DATE: one-stage preferred
   - DUE_DATE: one-stage preferred
   - ISSUER_NAME: two-stage preferred, guarded by garbage detector
   - RECIPIENT_NAME: one-stage preferred
   - TOTAL_AMOUNT: two-stage preferred with currency-symbol preservation

This module is intentionally backend-ready and independent of notebook code
except for reusing stable functions from src/extraction_improvements.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from extraction_improvements import (
    FIELD_THRESHOLDS,
    clean_invoice_number,
    extract_with_confidence_gating,
    ocr_image,
    sort_reading_order,
    validate_and_correct_fields,
)


FIELD_ORDER = [
    "INVOICE_NUMBER",
    "INVOICE_DATE",
    "DUE_DATE",
    "ISSUER_NAME",
    "RECIPIENT_NAME",
    "TOTAL_AMOUNT",
]

FIELD_PADDING = {
    "INVOICE_NUMBER": 10,
    "INVOICE_DATE": 20,
    "DUE_DATE": 20,
    "ISSUER_NAME": 25,
    "RECIPIENT_NAME": 25,
    "TOTAL_AMOUNT": 15,
}

_ADDRESS_WORD_RE = re.compile(
    r"\b(?:street|st\.|avenue|ave|road|rd\.|drive|dr\.|lane|ln\.|court|ct\.|square|sq\.)\b",
    re.IGNORECASE,
)
_STATE_ZIP_RE = re.compile(r",\s*[A-Z]{2}\s+\d{5}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\-\(\)\s]{7,}")
_LABEL_GARBAGE_RE = re.compile(
    r"\b(?:invoice|bill\s+to|ship\s+to|due\s+date|total|amount|purchase\s+order|p\.o\.?\s*#?)\b",
    re.IGNORECASE,
)
_CURRENCY_SYMBOL_RE = re.compile(r"[€$£¥₹₩]")
_DATE_LIKE_RE = re.compile(
    r"^(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}-[A-Za-z]{3}-\d{2,4})$"
)
_INVOICE_SHORT_CODE_RE = re.compile(r"\b[A-Z]{2,}-\d{3,}\b")
_RECIPIENT_LABEL_NOISE_RE = re.compile(
    r"\b(?:invoice|bill\s+to|ship\s+to|to|from)\b\s*#?:?",
    re.IGNORECASE,
)


class HybridInvoiceFieldExtractor:
    """Hybrid one-stage/two-stage extractor with deterministic field routing."""

    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        id2label: Dict[int, str],
        cleaner,
        *,
        ocr_engine: str = "doctr",
        max_length: int = 512,
        stage1_confidence_threshold: float = 0.70,
        field_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.id2label = id2label
        self.cleaner = cleaner
        self.ocr_engine = self._resolve_ocr_engine(ocr_engine)
        self.max_length = max_length
        self.stage1_confidence_threshold = stage1_confidence_threshold
        self.field_thresholds = dict(field_thresholds or FIELD_THRESHOLDS)
        # Single shared model instance is used across worker threads.
        self._model_lock = threading.Lock()

    def extract(self, image: Image.Image) -> Dict[str, str]:
        """Run one-stage + two-stage in parallel and merge by field policy."""
        image = image.convert("RGB")
        words, bboxes = self._ocr_and_sort(image)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_one = pool.submit(self._run_one_stage, image, words, bboxes)
            fut_two = pool.submit(self._run_two_stage, image, words, bboxes)
            one_stage = fut_one.result()
            two_stage = fut_two.result()

        merged = self._merge(one_stage, two_stage)
        return self._post_merge_corrections(merged, one_stage, two_stage, words)

    def _ocr_and_sort(self, image: Image.Image) -> Tuple[List[str], List[List[int]]]:
        words, bboxes = ocr_image(image, engine=self.ocr_engine)
        return sort_reading_order(words, bboxes)

    def _run_one_stage(
        self,
        image: Image.Image,
        words: List[str],
        bboxes: List[List[int]],
    ) -> Dict[str, str]:
        with self._model_lock:
            result = extract_with_confidence_gating(
                image,
                words,
                bboxes,
                self.model,
                self.processor,
                self.device,
                self.id2label,
                cleaner=self.cleaner,
                model_threshold=self.field_thresholds,
                ocr_text=" ".join(words),
                max_length=self.max_length,
            )
        conf = result.pop("confidence_scores", {})
        validated = validate_and_correct_fields(result, conf, words)
        return {
            "invoice_number": validated.get("invoice_number", "") or "",
            "invoice_date": validated.get("invoice_date", "") or "",
            "due_date": validated.get("due_date", "") or "",
            "issuer_name": validated.get("issuer_name", "") or "",
            "recipient_name": validated.get("recipient_name", "") or "",
            "total_amount": validated.get("total_amount", "") or "",
        }

    def _run_two_stage(
        self,
        image: Image.Image,
        words: List[str],
        bboxes: List[List[int]],
    ) -> Dict[str, str]:
        stage1 = self._get_field_bboxes(image, words, bboxes)

        # Spatial sanity: issuer and recipient occupying same area => issuer unreliable.
        issuer_bbox = stage1["ISSUER_NAME"]["bbox"]
        recipient_bbox = stage1["RECIPIENT_NAME"]["bbox"]
        if issuer_bbox is not None and recipient_bbox is not None:
            if self._compute_iou(issuer_bbox, recipient_bbox) > 0.5:
                stage1["ISSUER_NAME"]["bbox"] = None

        # Spatial sanity: due_date near invoice_number tends to be invoice fragment.
        inv_bbox = stage1["INVOICE_NUMBER"]["bbox"]
        due_bbox = stage1["DUE_DATE"]["bbox"]
        if inv_bbox is not None and due_bbox is not None:
            if self._bboxes_close(inv_bbox, due_bbox, threshold=50):
                stage1["DUE_DATE"]["bbox"] = None

        out: Dict[str, str] = {}
        for field in FIELD_ORDER:
            val = self._extract_field_from_crop(image, stage1[field]["bbox"], field)
            out[field.lower()] = val

        return {
            "invoice_number": out.get("invoice_number", "") or "",
            "invoice_date": out.get("invoice_date", "") or "",
            "due_date": out.get("due_date", "") or "",
            "issuer_name": out.get("issuer_name", "") or "",
            "recipient_name": out.get("recipient_name", "") or "",
            "total_amount": out.get("total_amount", "") or "",
        }

    def _get_field_bboxes(
        self,
        image: Image.Image,
        words: List[str],
        bboxes: List[List[int]],
    ) -> Dict[str, Dict[str, Any]]:
        encoding = self.processor(
            image,
            words,
            boxes=bboxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        with self._model_lock:
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in encoding.items()})

        logits = outputs.logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)

        token_pred_ids = logits.argmax(-1).cpu().tolist()
        token_max_probs = probs.max(-1).values.cpu().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        word_preds: Dict[int, Tuple[int, float]] = {}
        for ti, wi in enumerate(word_ids):
            if wi is not None and wi not in word_preds:
                word_preds[wi] = (token_pred_ids[ti], token_max_probs[ti])

        field_data: Dict[str, Dict[str, Any]] = {
            f: {"boxes": [], "confs": [], "tokens": []} for f in FIELD_ORDER
        }

        for wi in sorted(word_preds):
            pred_id, conf = word_preds[wi]
            label = self.id2label[pred_id]
            if label == "O":
                continue

            field = label[2:]
            if field not in field_data or conf < self.stage1_confidence_threshold:
                continue
            if wi >= len(bboxes):
                continue

            field_data[field]["boxes"].append(bboxes[wi])
            field_data[field]["confs"].append(conf)
            field_data[field]["tokens"].append(words[wi])

        result: Dict[str, Dict[str, Any]] = {}
        for field in FIELD_ORDER:
            boxes_f = field_data[field]["boxes"]
            if not boxes_f:
                result[field] = {"bbox": None, "confidence": 0.0, "tokens": []}
                continue

            merged_bbox = [
                min(b[0] for b in boxes_f),
                min(b[1] for b in boxes_f),
                max(b[2] for b in boxes_f),
                max(b[3] for b in boxes_f),
            ]
            avg_conf = float(sum(field_data[field]["confs"]) / len(field_data[field]["confs"]))
            result[field] = {
                "bbox": merged_bbox,
                "confidence": round(avg_conf, 4),
                "tokens": field_data[field]["tokens"],
            }

        return result

    def _extract_field_from_crop(
        self,
        image: Image.Image,
        bbox_normalized: Optional[List[int]],
        field_name: str,
        min_crop_size: int = 10,
    ) -> str:
        if bbox_normalized is None:
            return ""

        x0_n, y0_n, x1_n, y1_n = bbox_normalized
        pad = FIELD_PADDING.get(field_name, 15)

        x0_n = max(0, x0_n - pad)
        y0_n = max(0, y0_n - pad)
        x1_n = min(1000, x1_n + pad)
        y1_n = min(1000, y1_n + pad)

        w, h = image.size
        x0 = int(x0_n / 1000 * w)
        y0 = int(y0_n / 1000 * h)
        x1 = int(x1_n / 1000 * w)
        y1 = int(y1_n / 1000 * h)

        if (x1 - x0) < min_crop_size or (y1 - y0) < min_crop_size:
            return ""

        crop = image.crop((x0, y0, x1, y1))
        try:
            crop_words, _ = ocr_image(crop, engine=self.ocr_engine)
        except Exception:
            return ""

        if not crop_words or crop_words == ["empty"]:
            return ""

        crop_text = " ".join(crop_words)
        cleaned = self.cleaner.clean({field_name: crop_text}, ocr_words=crop_words)
        value = cleaned.get(field_name.lower(), "").strip()
        return value or crop_text

    def _merge(self, one_stage: Dict[str, str], two_stage: Dict[str, str]) -> Dict[str, str]:
        merged: Dict[str, str] = {}

        # INVOICE_NUMBER: two-stage preferred.
        merged["invoice_number"] = self._pick(two_stage.get("invoice_number", ""), one_stage.get("invoice_number", ""))

        # INVOICE_DATE and DUE_DATE: always one-stage by policy.
        merged["invoice_date"] = one_stage.get("invoice_date", "") or ""
        merged["due_date"] = one_stage.get("due_date", "") or ""

        # ISSUER_NAME: two-stage preferred, but reject garbage and fall back.
        ts_issuer = two_stage.get("issuer_name", "") or ""
        os_issuer = one_stage.get("issuer_name", "") or ""
        if self._is_empty(ts_issuer) or self._is_issuer_garbage(ts_issuer):
            merged["issuer_name"] = os_issuer
        else:
            merged["issuer_name"] = ts_issuer

        # RECIPIENT_NAME: one-stage (policy says one-stage or either).
        merged["recipient_name"] = one_stage.get("recipient_name", "") or ""

        # TOTAL_AMOUNT: two-stage preferred + currency symbol preservation.
        ts_amount = two_stage.get("total_amount", "") or ""
        os_amount = one_stage.get("total_amount", "") or ""
        merged_amount = self._pick(ts_amount, os_amount)
        merged["total_amount"] = self._preserve_currency_symbol(merged_amount, ts_amount, os_amount)

        return merged

    def _post_merge_corrections(
        self,
        merged: Dict[str, str],
        one_stage: Dict[str, str],
        two_stage: Dict[str, str],
        ocr_words: List[str],
    ) -> Dict[str, str]:
        """
        Final safety net for known real-world failure modes:
        1) invoice_number drifting to a date string
        2) recipient_name carrying label noise / duplicated tail
        """
        out = dict(merged)

        # Recipient cleanup: strip invoice labels/codes and deduplicate
        out["recipient_name"] = self._clean_recipient_name(out.get("recipient_name", ""))

        # Invoice number cleanup using shared src rule engine.
        ocr_text = " ".join(ocr_words or [])
        inv_clean = clean_invoice_number(
            out.get("invoice_number", "") or "",
            {
                "raw_text": ocr_text,
                "labelled_fields": {
                    "invoice_number": out.get("invoice_number", ""),
                    "recipient_name": out.get("recipient_name", ""),
                    "one_stage_invoice_number": one_stage.get("invoice_number", ""),
                    "two_stage_invoice_number": two_stage.get("invoice_number", ""),
                },
                "ocr_candidates": self._collect_invoice_candidates(out, one_stage, two_stage, ocr_text),
            },
        )

        # Guard against date leakage into invoice_number.
        if self._looks_like_date(inv_clean) and self._looks_like_date(out.get("invoice_date", "")):
            if self._normalize_date_like(inv_clean) == self._normalize_date_like(out.get("invoice_date", "")):
                inv_clean = "—"

        # If invoice number still empty, try extracting a short code from recipient.
        if self._is_empty(inv_clean):
            m = _INVOICE_SHORT_CODE_RE.search(out.get("recipient_name", "") or "")
            if m:
                inv_clean = m.group(0)

        out["invoice_number"] = inv_clean if not self._is_empty(inv_clean) else "—"
        return out

    @staticmethod
    def _pick(preferred: str, fallback: str) -> str:
        return preferred if not HybridInvoiceFieldExtractor._is_empty(preferred) else (fallback or "")

    @staticmethod
    def _is_empty(val: Any) -> bool:
        if val is None:
            return True
        s = str(val).strip()
        return s == "" or s == "—"

    @staticmethod
    def _compute_iou(bbox_a: List[int], bbox_b: List[int]) -> float:
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _bboxes_close(bbox_a: List[int], bbox_b: List[int], threshold: int = 50) -> bool:
        ca_x = (bbox_a[0] + bbox_a[2]) / 2
        ca_y = (bbox_a[1] + bbox_a[3]) / 2
        cb_x = (bbox_b[0] + bbox_b[2]) / 2
        cb_y = (bbox_b[1] + bbox_b[3]) / 2
        return abs(ca_x - cb_x) < threshold and abs(ca_y - cb_y) < threshold

    @staticmethod
    def _is_issuer_garbage(value: str) -> bool:
        s = value.strip()
        if s == "":
            return True

        lower = s.lower()
        if "@" in s or "http" in lower or "www." in lower:
            return True
        if _LABEL_GARBAGE_RE.search(s):
            return True
        if _PHONE_RE.search(s):
            return True
        if _STATE_ZIP_RE.search(s):
            return True
        if _ADDRESS_WORD_RE.search(s):
            return True

        if len(s) > 80:
            return True
        if len(s.split()) > 10:
            return True

        digits = sum(ch.isdigit() for ch in s)
        alpha = sum(ch.isalpha() for ch in s)
        if digits >= 4:
            return True
        if alpha == 0:
            return True
        if (alpha / max(len(s), 1)) < 0.40:
            return True

        return False

    @staticmethod
    def _extract_currency_symbol(value: str) -> str:
        m = _CURRENCY_SYMBOL_RE.search(value or "")
        return m.group(0) if m else ""

    @staticmethod
    def _looks_like_date(value: str) -> bool:
        s = (value or "").strip()
        return bool(_DATE_LIKE_RE.match(s))

    @staticmethod
    def _normalize_date_like(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z]", "", (value or "").strip()).lower()

    @staticmethod
    def _dedup_name(value: str) -> str:
        tokens = (value or "").split()
        half = len(tokens) // 2
        if half > 0 and tokens[:half] == tokens[half:half * 2]:
            return " ".join(tokens[:half])
        # Handle tail-repeat pattern: "John Smith John" -> "John Smith"
        if len(tokens) >= 3 and tokens[0].lower() == tokens[-1].lower():
            return " ".join(tokens[:-1])
        return value

    def _clean_recipient_name(self, value: str) -> str:
        s = (value or "").strip()
        if self._is_empty(s):
            return ""

        # Remove label words and trailing punctuation markers.
        s = _RECIPIENT_LABEL_NOISE_RE.sub(" ", s)
        # Remove invoice-like code tokens from recipient line.
        s = _INVOICE_SHORT_CODE_RE.sub(" ", s)
        s = re.sub(r"[#:/]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip(" ,.-")
        s = self._dedup_name(s)

        # Prefer a clean title-cased name chunk when present.
        chunks = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", s)
        if chunks:
            s = chunks[-1].strip()

        return s or value

    @staticmethod
    def _collect_invoice_candidates(
        merged: Dict[str, str],
        one_stage: Dict[str, str],
        two_stage: Dict[str, str],
        ocr_text: str,
    ) -> List[str]:
        cands: List[str] = []
        for v in (
            merged.get("invoice_number", ""),
            one_stage.get("invoice_number", ""),
            two_stage.get("invoice_number", ""),
            merged.get("recipient_name", ""),
            one_stage.get("recipient_name", ""),
            two_stage.get("recipient_name", ""),
        ):
            if v and str(v).strip():
                cands.append(str(v).strip())
        cands.extend(re.findall(r"\b[A-Z0-9][A-Z0-9/\-]{2,}\b", ocr_text))
        return cands

    @staticmethod
    def _numeric_core(value: str) -> str:
        return re.sub(r"[^0-9.,]", "", value or "")

    @staticmethod
    def _normalized_numeric_for_compare(value: str) -> str:
        num = HybridInvoiceFieldExtractor._numeric_core(value)
        if num.count(",") > 0 and num.count(".") == 0:
            num = num.replace(",", ".")
        num = num.replace(",", "")
        return num

    @staticmethod
    def _preserve_currency_symbol(merged_amount: str, two_stage_amount: str, one_stage_amount: str) -> str:
        """
        Preserve one-stage currency symbol when two-stage keeps numeric value
        but drops symbol (e.g. '$556.90' -> '556.90').
        """
        if HybridInvoiceFieldExtractor._is_empty(merged_amount):
            return ""

        merged = merged_amount.strip()
        sym_merged = HybridInvoiceFieldExtractor._extract_currency_symbol(merged)
        if sym_merged:
            return merged

        sym_one = HybridInvoiceFieldExtractor._extract_currency_symbol(one_stage_amount or "")
        if not sym_one:
            return merged

        # Apply symbol only when numbers are compatible.
        if not HybridInvoiceFieldExtractor._is_empty(two_stage_amount):
            a = HybridInvoiceFieldExtractor._normalized_numeric_for_compare(two_stage_amount)
            b = HybridInvoiceFieldExtractor._normalized_numeric_for_compare(one_stage_amount)
            if a and b and a == b:
                return f"{sym_one}{merged}"

        if HybridInvoiceFieldExtractor._is_empty(two_stage_amount):
            # If merged came from one-stage already, keep original one-stage formatting.
            return one_stage_amount.strip()

        return merged

    @staticmethod
    def _resolve_ocr_engine(requested_engine: str) -> str:
        """
        Use DocTR when available; otherwise downgrade to Tesseract silently.
        This avoids repeated runtime warnings on every OCR call.
        """
        if requested_engine != "doctr":
            return requested_engine
        try:
            import doctr  # type: ignore  # noqa: F401
            return "doctr"
        except Exception:
            return "tesseract"
