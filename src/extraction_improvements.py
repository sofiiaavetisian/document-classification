"""
extraction_improvements.py
===========================
Standalone module containing 4 targeted improvements to the LayoutLMv3
invoice field extraction pipeline.

Improvements
------------
1. Reading order sorting        — sort OCR tokens top-to-bottom, left-to-right
                                  before feeding to LayoutLMv3.  Directly fixes
                                  the swapped-dates failure observed in testing.

2. Confidence-gated extraction  — run the regex fallback when LayoutLMv3 softmax
                                  confidence is below a threshold.  Fixes missed
                                  fields and low-quality predictions.

3. DocTR OCR engine             — drop-in replacement for Tesseract.  Produces
                                  cleaner word segmentation on document images,
                                  especially for merged tokens (e.g. 'to:Nicole').

4. Entity-level validation      — business rule checks (date swap, bad amounts,
                                  label mis-predictions, address leakage) with
                                  OCR-stream recovery where possible.

Usage
-----
    import sys; sys.path.insert(0, '/path/to/document-classification/src')
    from extraction_improvements import (
        sort_reading_order,
        ocr_image, ocr_image_tesseract, ocr_image_doctr,
        get_raw_predictions_with_confidence,
        extract_with_confidence_gating,
        validate_and_correct_fields,
        process_invoice,
    )

Constraints honoured
--------------------
* TOKENIZERS_PARALLELISM=false set before any transformers import.
* use_fast=True for LayoutLMv3Processor — avoids macOS deadlock.
* local_files_only=True for all HuggingFace model loads.
* num_workers=0 on all DataLoaders (callers' responsibility).
* No generative AI — LayoutLMv3 is a discriminative token classifier only.
* DocTR gracefully falls back to Tesseract if the package is not installed.
* Does NOT import from any notebook file.
"""

import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Must be set before any transformers / tokenizers import
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Reading Order Sorting
# ─────────────────────────────────────────────────────────────────────────────

ROW_THRESHOLD: int = 15  # normalised bbox units ([0, 1000] space) ≈ 1.5% of height

# Per-field confidence thresholds used by default for confidence gating.
FIELD_THRESHOLDS: Dict[str, float] = {
    'INVOICE_NUMBER': 0.70,
    'INVOICE_DATE':   0.70,
    'DUE_DATE':       0.70,
    'ISSUER_NAME':    0.95,
    'RECIPIENT_NAME': 0.50,
    'TOTAL_AMOUNT':   0.70,
}


def sort_reading_order(
    words: List[str],
    boxes: List[List[int]],
) -> Tuple[List[str], List[List[int]]]:
    """
    Sort OCR tokens into natural reading order: top-to-bottom, left-to-right.

    Uses row clustering: tokens whose y0 coordinates are within ROW_THRESHOLD
    (15 units in [0, 1000] normalised space) of each other are considered to be
    on the same row and sorted left-to-right within that row.  Rows are then
    sorted top-to-bottom by their centroid y0.

    This must be applied to OCR word/bbox output BEFORE feeding to LayoutLMv3.
    Tesseract does not always return tokens in strict top-to-bottom order on
    complex invoice layouts, causing the model's spatial attention to become
    confused — most visibly as swapped dates (both assigned to DUE_DATE).

    Parameters
    ----------
    words : list of str
        OCR word tokens.
    boxes : list of list of int
        Normalised bounding boxes [x0, y0, x1, y1] in [0, 1000] space.
        Must be the same length as ``words``.

    Returns
    -------
    sorted_words : list of str
    sorted_boxes : list of list of int
        Same formats as inputs, reordered into reading order.

    Notes
    -----
    ROW_THRESHOLD = 15 groups tokens within 1.5 % of page height into the same
    row.  This is appropriate for typical invoice line heights.  Increase it for
    documents with very large line spacing; decrease for dense tables.
    """
    if not words:
        return list(words), list(boxes)

    # Pair each token with its original index for diagnostics
    indexed = list(enumerate(zip(words, boxes)))

    # Sort by y0 so the greedy row-extension always processes tokens top-down
    indexed_by_y = sorted(indexed, key=lambda t: t[1][1][1])

    rows: List[List[tuple]] = []  # each row: [(orig_idx, (word, box)), ...]
    for orig_idx, (word, box) in indexed_by_y:
        y0 = box[1]
        placed = False
        for row in rows:
            row_y = sum(item[1][1][1] for item in row) / len(row)
            if abs(y0 - row_y) <= ROW_THRESHOLD:
                row.append((orig_idx, (word, box)))
                placed = True
                break
        if not placed:
            rows.append([(orig_idx, (word, box))])

    # Sort rows top-to-bottom; tokens within each row left-to-right
    rows.sort(key=lambda row: sum(item[1][1][1] for item in row) / len(row))
    for row in rows:
        row.sort(key=lambda t: t[1][1][0])  # sort by x0

    sorted_words: List[str]        = []
    sorted_boxes: List[List[int]]  = []
    for row in rows:
        for _, (word, box) in row:
            sorted_words.append(word)
            sorted_boxes.append(box)

    return sorted_words, sorted_boxes


# ─────────────────────────────────────────────────────────────────────────────
# 2.  OCR Engines
# ─────────────────────────────────────────────────────────────────────────────

def ocr_image_tesseract(image: Image.Image) -> Tuple[List[str], List[List[int]]]:
    """
    Run Tesseract OCR on a PIL image.

    This is the existing OCR function from notebook 13, extracted as an
    importable function so that it can be used as a named engine alongside
    DocTR and compared in benchmarks.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (RGB recommended).

    Returns
    -------
    words : list of str
        OCR word tokens, confidence-filtered (conf >= 10).
    boxes : list of [x0, y0, x1, y1]
        Bounding boxes normalised to [0, 1000] integer space.

    Notes
    -----
    Known limitation: Tesseract occasionally merges tokens that should be
    separate, e.g. ``to:Nicole`` instead of ``to:`` + ``Nicole``.  Use
    ``ocr_image_doctr()`` to avoid this problem.
    """
    import pytesseract  # type: ignore

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: List[str]        = []
    boxes: List[List[int]]  = []
    w, h = image.size

    for i, text in enumerate(data['text']):
        text = str(text).strip()
        if not text or int(data['conf'][i]) < 10:
            continue
        left, top     = data['left'][i], data['top'][i]
        width, height = data['width'][i], data['height'][i]
        x0 = int(max(0, min(1000, round(left / w * 1000))))
        y0 = int(max(0, min(1000, round(top  / h * 1000))))
        x1 = int(max(0, min(1000, round((left + width)  / w * 1000))))
        y1 = int(max(0, min(1000, round((top  + height) / h * 1000))))
        if x1 <= x0:
            x1 = min(1000, x0 + 1)
        if y1 <= y0:
            y1 = min(1000, y0 + 1)
        words.append(text)
        boxes.append([x0, y0, x1, y1])

    return (words or ['empty']), (boxes or [[0, 0, 1, 1]])


def ocr_image_doctr(image: Image.Image) -> Tuple[List[str], List[List[int]]]:
    """
    Run DocTR OCR on a PIL image.

    Output format is identical to ``ocr_image_tesseract()`` — this is a
    drop-in replacement for the existing ``ocr_image()`` function in notebook
    13.  The rest of the pipeline (sort_reading_order, get_raw_predictions_
    with_confidence, etc.) works without any other changes.

    DocTR is a modern deep-learning OCR engine that produces significantly
    better word segmentation on document images, especially for cases where
    Tesseract merges a label and its value (e.g. ``to:Nicole``, ``Date:2024``).

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (RGB recommended).

    Returns
    -------
    words : list of str
        OCR word tokens, typically better-segmented than Tesseract.
    boxes : list of [x0, y0, x1, y1]
        Bounding boxes normalised to [0, 1000] integer space.

    Notes
    -----
    Requires ``python-doctr``:  ``pip install python-doctr``

    If DocTR is not available (import fails), automatically falls back to
    Tesseract with a RuntimeWarning.  The pipeline will never crash because of
    a missing dependency.
    """
    try:
        import importlib
        import numpy as np
        import sys
        from doctr.models import ocr_predictor  # type: ignore
    except ImportError:
        warnings.warn(
            "python-doctr is not installed — falling back to Tesseract.  "
            "Install with:  pip install python-doctr",
            RuntimeWarning,
            stacklevel=2,
        )
        return ocr_image_tesseract(image)

    # Avoid clashes with a stale/local module named `validators` that can
    # break DocTR internals (expects the external package with validators.url).
    cached_validators = sys.modules.get('validators')
    if cached_validators is not None and not hasattr(cached_validators, 'url'):
        del sys.modules['validators']
    try:
        imported_validators = importlib.import_module('validators')
        if not hasattr(imported_validators, 'url'):
            raise AttributeError("validators.url is unavailable")
    except Exception as exc:
        warnings.warn(
            "DocTR dependency resolution failed for package `validators` "
            f"({exc!r}) — falling back to Tesseract.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ocr_image_tesseract(image)

    img_array = np.array(image.convert('RGB'))

    # Build the predictor once per call; for repeated calls the caller should
    # pass a pre-built predictor to avoid repeated model instantiation.
    predictor = ocr_predictor(pretrained=True)
    result    = predictor([img_array])

    words: List[str]        = []
    boxes: List[List[int]]  = []

    for page in result.pages:
        page_h, page_w = page.dimensions
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text = word.value.strip()
                    if not text:
                        continue
                    # DocTR geometry: ((rx0, ry0), (rx1, ry1)) in [0, 1]
                    (rx0, ry0), (rx1, ry1) = word.geometry
                    x0 = int(max(0, min(1000, round(rx0 * 1000))))
                    y0 = int(max(0, min(1000, round(ry0 * 1000))))
                    x1 = int(max(0, min(1000, round(rx1 * 1000))))
                    y1 = int(max(0, min(1000, round(ry1 * 1000))))
                    if x1 <= x0:
                        x1 = min(1000, x0 + 1)
                    if y1 <= y0:
                        y1 = min(1000, y0 + 1)
                    words.append(text)
                    boxes.append([x0, y0, x1, y1])

    return (words or ['empty']), (boxes or [[0, 0, 1, 1]])


def ocr_image(
    image: Image.Image,
    engine: str = 'doctr',
) -> Tuple[List[str], List[List[int]]]:
    """
    Auto-routing OCR function.

    Routes to the specified engine.  If 'doctr' is requested but python-doctr
    is not installed, silently falls back to Tesseract.

    Parameters
    ----------
    image : PIL.Image.Image
    engine : {'doctr', 'tesseract'}
        OCR engine to use.  Default: 'doctr'.

    Returns
    -------
    words : list of str
    boxes : list of [x0, y0, x1, y1]
    """
    if engine == 'tesseract':
        return ocr_image_tesseract(image)
    return ocr_image_doctr(image)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Confidence-Gated Extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_raw_predictions_with_confidence(
    image: Image.Image,
    words: List[str],
    bboxes: List[List[int]],
    model,
    processor,
    device,
    id2label: Dict[int, str],
    max_length: int = 512,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Run LayoutLMv3 and return both raw field strings AND per-field confidence.

    Extends ``get_raw_predictions()`` from notebook 13 by also computing the
    mean softmax confidence across all tokens assigned to each predicted field.
    This confidence score is used by ``extract_with_confidence_gating()`` to
    decide whether to trust the model or fall back to the InvoiceCleaner regex.

    Parameters
    ----------
    image : PIL.Image.Image
    words : list of str
        OCR word tokens.  Apply ``sort_reading_order()`` before calling.
    bboxes : list of [x0, y0, x1, y1]
        Normalised bounding boxes.
    model : LayoutLMv3ForTokenClassification
        Loaded model, already moved to ``device`` and set to eval mode.
    processor : LayoutLMv3Processor
        Loaded processor (use_fast=True).
    device : torch.device
    id2label : dict {int: str}
        Label id → BIO label string.
    max_length : int
        Processor truncation length.  Default: 512.

    Returns
    -------
    raw_fields : dict {FIELD_NAME: raw_string}
        Uppercase keys, no cleaning — identical format to get_raw_predictions().
    confidences : dict {FIELD_NAME: float}
        Mean softmax confidence (0.0–1.0) across all tokens assigned to that
        field.  Fields the model never predicted have no entry (use .get(k, 0.0)).
        Higher value = model is more certain.
    """
    import torch.nn.functional as F

    encoding = processor(
        image, words, boxes=bboxes,
        truncation=True, padding='max_length',
        max_length=max_length, return_tensors='pt',
    )

    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in encoding.items()})

    logits = outputs.logits.squeeze(0)          # (seq_len, num_labels)
    probs  = F.softmax(logits, dim=-1)           # (seq_len, num_labels)

    token_pred_ids: List[int]   = logits.argmax(-1).cpu().tolist()
    token_max_probs: List[float] = probs.max(-1).values.cpu().tolist()
    word_ids = encoding.word_ids(batch_index=0)

    # Map subword tokens → word level (first subword per word, same as nb13)
    word_preds: Dict[int, int]   = {}
    word_confs: Dict[int, float] = {}
    for ti, wi in enumerate(word_ids):
        if wi is not None and wi not in word_preds:
            word_preds[wi] = token_pred_ids[ti]
            word_confs[wi] = token_max_probs[ti]

    sorted_word_ids   = sorted(word_preds)
    aligned_words     = [words[i]      for i in sorted_word_ids]
    aligned_pred_ids  = [word_preds[i] for i in sorted_word_ids]
    aligned_confs     = [word_confs[i] for i in sorted_word_ids]

    # Group BIO tokens into field strings and accumulate per-field confidences
    raw_fields: Dict[str, str]              = {}
    field_conf_lists: Dict[str, List[float]] = {}
    current_field:  Optional[str]           = None
    current_tokens: List[str]               = []
    current_confs_acc: List[float]          = []

    def _flush() -> None:
        nonlocal current_field, current_tokens, current_confs_acc
        if current_field:
            text = ' '.join(current_tokens).strip()
            if text:
                raw_fields[current_field] = text
                field_conf_lists.setdefault(current_field, []).extend(current_confs_acc)
        current_field, current_tokens, current_confs_acc = None, [], []

    for label_id, word, conf in zip(aligned_pred_ids, aligned_words, aligned_confs):
        label = id2label[label_id]
        if label == 'O':
            _flush()
        elif label.startswith('B-'):
            _flush()
            current_field       = label[2:]
            current_tokens      = [word]
            current_confs_acc   = [conf]
        elif label.startswith('I-'):
            fn = label[2:]
            if current_field == fn:
                current_tokens.append(word)
                current_confs_acc.append(conf)
            elif current_field is None and fn in raw_fields:
                raw_fields[fn] += ' ' + word
                field_conf_lists.setdefault(fn, []).append(conf)
            elif current_field is None:
                current_field     = fn
                current_tokens    = [word]
                current_confs_acc = [conf]
            else:
                _flush()
                current_field     = fn
                current_tokens    = [word]
                current_confs_acc = [conf]
    _flush()

    confidences: Dict[str, float] = {
        field: float(sum(c_list) / len(c_list))
        for field, c_list in field_conf_lists.items()
        if c_list
    }

    return raw_fields, confidences


def extract_with_confidence_gating(
    image: Image.Image,
    words: List[str],
    bboxes: List[List[int]],
    model,
    processor,
    device,
    id2label: Dict[int, str],
    cleaner,
    model_threshold: Union[float, Dict[str, float]] = FIELD_THRESHOLDS,
    ocr_text: str = '',
    max_length: int = 512,
) -> Dict[str, str]:
    """
    Confidence-gated extraction: LayoutLMv3 for high-confidence fields,
    InvoiceCleaner regex fallback for low-confidence fields.

    Decision logic per field
    ------------------------
    * field confidence >= model_threshold  →  use LayoutLMv3 prediction,
                                              clean with InvoiceCleaner
    * field confidence < model_threshold   →  pass empty string to InvoiceCleaner
                                              so its OCR-word fallbacks activate

    This means InvoiceCleaner's regex fallbacks (bill-to keyword scan, date
    keyword scan, swapped-date arbitration) are triggered automatically for any
    field the model is uncertain about.

    Parameters
    ----------
    image : PIL.Image.Image
    words : list of str
        OCR tokens.  Apply ``sort_reading_order()`` before calling.
    bboxes : list of [x0, y0, x1, y1]
    model, processor, device, id2label
        Loaded LayoutLMv3 components.
    cleaner : InvoiceCleaner
        Post-processing cleaner instance from ``src/invoice_cleaner.py``.
    model_threshold : float | dict[str, float]
        Confidence threshold(s). If float, one threshold is used for all
        fields. If dict, per-field thresholds are used (keys = FIELD names).
    ocr_text : str
        Full OCR text joined as a single string.  If empty, built from words.
    max_length : int

    Returns
    -------
    dict
        Lowercase keys: ``invoice_number``, ``invoice_date``, ``due_date``,
        ``issuer_name``, ``recipient_name``, ``total_amount``.
        Plus ``'confidence_scores'`` key with per-field float values.
    """
    raw_fields, confidences = get_raw_predictions_with_confidence(
        image, words, bboxes, model, processor, device, id2label, max_length
    )

    ALL_FIELDS = [
        'INVOICE_NUMBER', 'INVOICE_DATE', 'DUE_DATE',
        'ISSUER_NAME', 'RECIPIENT_NAME', 'TOTAL_AMOUNT',
    ]

    # Support either one global threshold or per-field thresholds.
    if isinstance(model_threshold, dict):
        thresholds: Dict[str, float] = {
            field: float(model_threshold.get(field, FIELD_THRESHOLDS[field]))
            for field in ALL_FIELDS
        }
    else:
        thresholds = {field: float(model_threshold) for field in ALL_FIELDS}

    # Gate: keep model prediction only when confidence is above threshold
    gated_raw: Dict[str, str] = {}
    for field in ALL_FIELDS:
        conf = confidences.get(field, 0.0)
        if conf >= thresholds[field] and field in raw_fields:
            gated_raw[field] = raw_fields[field]
        else:
            # Empty string triggers InvoiceCleaner's OCR-word fallback paths
            gated_raw[field] = ''

    # Run InvoiceCleaner — fallbacks activate for any empty field
    result = cleaner.clean(gated_raw, ocr_words=words)

    # Fix 1: if model has zero confidence for issuer, do not trust fallback.
    if confidences.get('ISSUER_NAME', 0.0) == 0.0:
        result['issuer_name'] = ''

    # Attach per-field confidence scores (0.0 for absent predictions)
    result['confidence_scores'] = {
        field: round(confidences.get(field, 0.0), 4)
        for field in ALL_FIELDS
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Entity-Level Cross-Field Validation
# ─────────────────────────────────────────────────────────────────────────────

_DATE_SCAN = re.compile(
    r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'
    r'|\b\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}\b'
    r'|\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b'
    r'|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
    re.IGNORECASE,
)

_AMOUNT_SCAN = re.compile(
    r'(?:[$€£¥₹₩]\s*\d[\d,.\s]*\d'
    r'|\d[\d,.\s]*\d\s*(?:USD|EUR|GBP|CAD|AUD|CHF)'
    r'|\d{1,3}(?:[,]\d{3})*\.\d{2})',
    re.IGNORECASE,
)

_DUE_LABEL_KW    = frozenset({'due', 'payment', 'pay', 'payable', 'expiry', 'deadline'})
_INVOICE_DATE_KW = frozenset({'invoice', 'issued', 'issue', 'created', 'date'})

_LABEL_ONLY_WORDS = frozenset({
    'TO', 'FROM', 'DATE', 'TOTAL', 'AMOUNT', 'INVOICE', 'NUMBER', 'DUE',
    'BILL', 'PAY', 'RECEIPT', 'ORDER', 'ISSUED', 'BALANCE', 'GRAND',
    'SUB', 'TAX', 'NET', 'ITEM', 'QTY', 'UNIT', 'PRICE', 'DESCRIPTION',
    'NAME', 'COMPANY', 'CLIENT', 'VENDOR', 'SUPPLIER', 'BUYER',
})
_REJECT_INVOICE_PREFIXES = re.compile(
    r'^(Address|View|Invoice|Date|Total|Bill|From|To)[:\s]',
    re.IGNORECASE,
)
_STRICT_RECOVERED_INVOICE_RE = re.compile(
    r'^(?:[A-Z0-9]{1,6}[-/]?)?[A-Z0-9]{2,}(?:[-/][A-Z0-9]{2,})*$',
    re.IGNORECASE,
)
_LABEL_ONLY_INVOICE_VALUES = frozenset({'INVOICE', 'ID', 'BUYER', 'REF'})

_INVOICE_LABEL_PATTERNS = (
    re.compile(
        r'\binvoice\s*(?:#|no\.?|number)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\-]{1,})\b',
        re.IGNORECASE,
    ),
)
_PO_LABEL_PATTERNS = (
    re.compile(
        r'\b(?:p\.?\s*o\.?\s*#?|purchase\s+order)\s*(?:#|no\.?|number)?\s*[:\-]?\s*'
        r'([A-Z0-9][A-Z0-9/\-]{1,})\b',
        re.IGNORECASE,
    ),
)
_ADDRESS_NUMBER_PATTERN = re.compile(
    r'\b(\d{3,4})\b[^\n]{0,40}\b(?:street|st\.|ave|avenue|road|drive|lane|court|square)\b',
    re.IGNORECASE,
)


def _norm_invoice_candidate(value: str) -> str:
    return value.strip().strip(',:;#()[]{}')


def _is_rejected_label_only_invoice(value: str) -> bool:
    cleaned = value.strip().upper()
    return cleaned in _LABEL_ONLY_INVOICE_VALUES and not any(ch.isdigit() for ch in cleaned)


def _invoice_rank(candidate: str, explicitly_labelled_invoice: bool) -> int:
    cand = candidate.strip()
    upper = cand.upper()

    # 1) INV/XX-XX/XXX or INV-XXXX-X style (structured with separators)
    if re.match(r'^(?:INV(?:OICE)?[/-]?)?[A-Z0-9]+(?:[-/][A-Z0-9]+){1,}$', upper):
        if any(ch.isdigit() for ch in upper):
            return 0

    # 2) INVXXXXXXXX (INV + 8 digits)
    if re.match(r'^INV\d{8}$', upper):
        return 1

    # 3) US-001 style short references, valid when explicitly invoice-labelled
    if explicitly_labelled_invoice and re.match(r'^[A-Z]{2,}-\d{3}$', upper):
        return 2

    # 4) Plain numeric strings are lowest-priority valid option
    if re.match(r'^\d+$', upper):
        return 3

    # Fallback (alphanumeric with digits) if none of the above
    if any(ch.isdigit() for ch in upper):
        return 4
    return 99


def clean_invoice_number(value: str, doc_context: dict) -> str:
    """
    Clean and select the primary invoice number from candidate sources.

    `doc_context` may include:
      - `raw_text`: full OCR text
      - `labelled_fields`: dict with label -> extracted value
      - `ocr_candidates`: list[str] of additional OCR candidates
    """
    raw_text = str(doc_context.get('raw_text', '') or '')
    labelled_fields = doc_context.get('labelled_fields', {}) or {}
    ocr_candidates = doc_context.get('ocr_candidates', []) or []

    invoice_labelled: List[str] = []
    po_labelled: List[str] = []
    generic_candidates: List[str] = []

    def _append_unique(bucket: List[str], candidate: str) -> None:
        c = _norm_invoice_candidate(candidate)
        if c and c not in bucket:
            bucket.append(c)

    # Candidate from current extracted value
    if value:
        _append_unique(generic_candidates, str(value))

    # Labelled fields from upstream parser (if available)
    for label, v in labelled_fields.items():
        if not v:
            continue
        label_l = str(label).lower()
        val_s = _norm_invoice_candidate(str(v))
        if not val_s:
            continue
        if 'invoice' in label_l:
            _append_unique(invoice_labelled, val_s)
        elif 'p.o' in label_l or 'po' in label_l or 'purchase order' in label_l:
            _append_unique(po_labelled, val_s)
        else:
            _append_unique(generic_candidates, val_s)

    # Scan raw OCR text for label/value pairs
    for pat in _INVOICE_LABEL_PATTERNS:
        for m in pat.finditer(raw_text):
            _append_unique(invoice_labelled, m.group(1))
    for pat in _PO_LABEL_PATTERNS:
        for m in pat.finditer(raw_text):
            _append_unique(po_labelled, m.group(1))

    for c in ocr_candidates:
        _append_unique(generic_candidates, str(c))

    # Build candidate pool
    if invoice_labelled:
        # Rule 1: if invoice-labelled candidate(s) exist, prefer those and never
        # return P.O. numbers when both are present.
        pool = list(invoice_labelled)
    else:
        pool = list(dict.fromkeys(generic_candidates + po_labelled))

    # Rule 4: reject label words with no digits.
    pool = [
        c for c in pool
        if (not _is_rejected_label_only_invoice(c)) and any(ch.isdigit() for ch in c)
    ]

    # Rule 2: reject bare 3-4 digit address fragments (e.g., "1912 Harvest Lane").
    address_nums = {m.group(1) for m in _ADDRESS_NUMBER_PATTERN.finditer(raw_text)}
    filtered: List[str] = []
    for cand in pool:
        if re.match(r'^\d{3,4}$', cand) and cand in address_nums:
            continue
        filtered.append(cand)
    pool = filtered

    if not pool:
        return '—'

    # Rule 3 + Rule 5: ranked selection with invoice-labelled awareness.
    ranked = sorted(
        pool,
        key=lambda c: (
            _invoice_rank(c, explicitly_labelled_invoice=(c in invoice_labelled)),
            len(c),
        ),
    )

    best = ranked[0]
    if _is_rejected_label_only_invoice(best):
        return '—'
    return best or '—'


def validate_and_correct_fields(
    fields: Dict[str, str],
    confidences: Dict[str, float],
    ocr_words: List[str],
) -> Dict[str, str]:
    """
    Apply business rule validation to extracted invoice fields.

    Detects violations that are identifiable without any additional model and
    corrects them using the OCR word stream where possible.

    Rules applied
    -------------
    1. DATE SWAP DETECTION
       If ``invoice_date == due_date``, find two distinct dates in the OCR
       stream and re-assign using label keyword proximity (words near 'due' →
       due_date; words near 'invoice'/'issued'/'date' → invoice_date).

    2. AMOUNT SANITY CHECK
       If ``total_amount`` is not parseable as a number, or looks like a date
       fragment (e.g. 24.09, value < 50.0), reject and re-run regex on OCR.

    3. INVOICE NUMBER SANITY CHECK
       If ``invoice_number`` is a common label word with no digits (TO, DATE,
       TOTAL, etc.), reject as label mis-prediction.  Invoice numbers must
       contain at least one digit.

    4. NAME FIELD SANITY CHECK
       If ``issuer_name`` or ``recipient_name`` contains '@', 'http', or is
       longer than 60 characters, it is an address/contact block — truncate
       to the first name-like segment.

    5. MISSING FIELD RECOVERY
       For any field that is empty or '—' after all previous steps, attempt
       OCR-stream regex recovery before giving up.

    Parameters
    ----------
    fields : dict
        Extracted fields with lowercase keys (output of
        ``extract_with_confidence_gating()`` or ``cleaner.clean()``).
    confidences : dict
        Per-field confidence scores.  Currently used for logging only.
    ocr_words : list of str
        Raw OCR word tokens for the image.

    Returns
    -------
    dict
        Same structure as ``fields`` with corrections applied, plus a
        ``'validation_notes'`` key (list of str) listing which rules fired.
    """
    result: Dict[str, str] = {
        k: v for k, v in fields.items()
        if k != 'validation_notes'
    }
    notes: List[str] = []
    ocr_text   = ' '.join(ocr_words)
    words_lower = [w.lower().strip(':.,-') for w in ocr_words]

    # ── Rule 1: Date swap detection ───────────────────────────────────────
    inv_date = result.get('invoice_date', '').strip()
    due_date = result.get('due_date', '').strip()
    if inv_date and due_date and inv_date == due_date:
        notes.append(
            f'RULE1: invoice_date == due_date ("{inv_date}") — '
            f'attempting re-assignment from OCR stream'
        )
        # Collect all distinct dates with their positions in the OCR stream
        date_positions: List[Tuple[int, str]] = []
        for i, word in enumerate(ocr_words):
            m = _DATE_SCAN.search(word)
            if m:
                candidate = m.group(0).strip()
                if not any(d[1] == candidate for d in date_positions):
                    date_positions.append((i, candidate))
        # Also catch multi-word date patterns spanning tokens
        for m in _DATE_SCAN.finditer(ocr_text):
            candidate = m.group(0).strip()
            if not any(d[1] == candidate for d in date_positions):
                date_positions.append((-1, candidate))

        unique_dates = list(dict.fromkeys(d[1] for d in date_positions))
        if len(unique_dates) >= 2:
            reassigned: Dict[str, str] = {}
            for pos, date in date_positions:
                if pos < 0:
                    continue
                ctx = ' '.join(words_lower[max(0, pos - 5):pos + 2])
                is_due = any(kw in ctx for kw in _DUE_LABEL_KW)
                is_inv = any(kw in ctx for kw in _INVOICE_DATE_KW)
                if is_due and 'due_date' not in reassigned:
                    reassigned['due_date'] = date
                if is_inv and 'invoice_date' not in reassigned:
                    reassigned['invoice_date'] = date

            if 'invoice_date' in reassigned and 'due_date' in reassigned:
                result['invoice_date'] = reassigned['invoice_date']
                result['due_date']     = reassigned['due_date']
                notes.append(
                    f"  → keyword re-assign: invoice_date={reassigned['invoice_date']}, "
                    f"due_date={reassigned['due_date']}"
                )
            else:
                # Positional fallback: earlier date → invoice_date, later → due_date
                result['invoice_date'] = unique_dates[0]
                result['due_date']     = unique_dates[1]
                notes.append(
                    f'  → positional fallback: invoice_date={unique_dates[0]}, '
                    f'due_date={unique_dates[1]}'
                )

    # ── Rule 2: Amount sanity check ───────────────────────────────────────
    total = result.get('total_amount', '').strip()
    if total and total != '—':
        num_str   = re.sub(r'[^\d.,]', '', total)
        amount_ok = False
        if num_str:
            try:
                val = float(num_str.replace(',', ''))
                # Fix 2: reject only strict DD.MM-like fragments (e.g. 24.09).
                if re.match(r'^\d{1,2}\.\d{2}$', num_str) and val <= 31.12:
                    notes.append(
                        f'RULE2: total_amount "{total}" looks like a date '
                        f'fragment (value {val:.2f} <= 31.12) — rejecting'
                    )
                else:
                    amount_ok = True
            except ValueError:
                notes.append(
                    f'RULE2: total_amount "{total}" not parseable as number — rejecting'
                )
        else:
            notes.append(
                f'RULE2: total_amount "{total}" contains no digits — rejecting'
            )

        if not amount_ok:
            result['total_amount'] = ''
            m = _AMOUNT_SCAN.search(ocr_text)
            if m:
                candidate = m.group(0).strip()
                try:
                    val = float(re.sub(r'[^\d.,]', '', candidate).replace(',', ''))
                    if val >= 1.0:
                        result['total_amount'] = candidate
                        notes.append(f'  → OCR recovery: "{candidate}"')
                except ValueError:
                    pass

    # ── Rule 3: Invoice number cleanup and ranking ────────────────────────
    inv_before = result.get('invoice_number', '').strip()
    inv_after = clean_invoice_number(
        inv_before,
        {
            'raw_text': ocr_text,
            'labelled_fields': {
                'invoice_number': inv_before,
                'due_date': result.get('due_date', ''),
                'invoice_date': result.get('invoice_date', ''),
            },
            'ocr_candidates': re.findall(r'\b[A-Z0-9][A-Z0-9/\-]{2,}\b', ocr_text),
        },
    )
    result['invoice_number'] = inv_after
    if (inv_before or '—') != inv_after:
        notes.append(
            f'RULE3: normalized invoice_number "{inv_before or "—"}" '
            f'→ "{inv_after}"'
        )

    # Fix 5: due_date should not look like invoice-number fragments (e.g. 9-3/22)
    due_candidate = result.get('due_date', '').strip()
    if due_candidate and re.match(r'^\d+-\d+/\d+$', due_candidate):
        notes.append(
            f'RULEX: due_date "{due_candidate}" looks like invoice-number '
            f'fragment — clearing'
        )
        result['due_date'] = ''

    # ── Rule 4: Name field sanity check ───────────────────────────────────
    for name_field in ('issuer_name', 'recipient_name'):
        name_val = result.get(name_field, '').strip()
        if not name_val or name_val == '—':
            continue

        if '@' in name_val or 'http' in name_val.lower() or 'www.' in name_val.lower():
            notes.append(
                f'RULE4: {name_field} contains contact info ("@"/URL) — truncating'
            )
            tokens = name_val.split()
            clean_toks: List[str] = []
            for tok in tokens:
                if '@' in tok or tok.lower().startswith('http') or tok.lower().startswith('www'):
                    break
                clean_toks.append(tok)
            result[name_field] = ' '.join(clean_toks[:6]).strip(',.:-() ')
            notes.append(f'  → truncated to: "{result[name_field]}"')

        elif len(name_val) > 60:
            notes.append(
                f'RULE4: {name_field} is {len(name_val)} chars (> 60) — '
                f'truncating to first name-like segment'
            )
            tokens   = name_val.split()
            truncated = ' '.join(tokens[:6]).strip(',.:-() ')
            result[name_field] = truncated
            notes.append(f'  → truncated to: "{truncated}"')

    # ── Rule 5: Missing field recovery ───────────────────────────────────
    if not result.get('invoice_date', ''):
        for i, w in enumerate(words_lower):
            if w in _INVOICE_DATE_KW:
                ctx = ' '.join(ocr_words[i:min(i + 6, len(ocr_words))])
                m   = _DATE_SCAN.search(ctx)
                if m:
                    result['invoice_date'] = m.group(0).strip()
                    notes.append(
                        f'RULE5: recovered invoice_date="{result["invoice_date"]}" from OCR'
                    )
                    break

    if not result.get('due_date', ''):
        for i, w in enumerate(words_lower):
            if w in _DUE_LABEL_KW:
                ctx = ' '.join(ocr_words[i:min(i + 6, len(ocr_words))])
                m   = _DATE_SCAN.search(ctx)
                if m:
                    candidate = m.group(0).strip()
                    if candidate != result.get('invoice_date', ''):
                        result['due_date'] = candidate
                        notes.append(
                            f'RULE5: recovered due_date="{result["due_date"]}" from OCR'
                        )
                        break

    if not result.get('total_amount', ''):
        m = re.search(
            r'(?:total|balance\s+due|amount\s+due|grand\s+total)'
            r'\s*:?\s*([$€£¥₹₩]?\s*\d[\d,.\s]*\d)',
            ocr_text, re.IGNORECASE
        )
        if m:
            candidate = m.group(1).strip()
            try:
                val = float(re.sub(r'[^\d.,]', '', candidate).replace(',', ''))
                if val >= 1.0:
                    result['total_amount'] = candidate
                    notes.append(
                        f'RULE5: recovered total_amount="{candidate}" from OCR'
                    )
            except ValueError:
                pass

    result['validation_notes'] = notes  # type: ignore[assignment]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_invoice(
    image_or_path,
    extractor_model,
    extractor_processor,
    device,
    id2label: Dict[int, str],
    cleaner,
    classifier_model=None,
    classifier_processor=None,
    ocr_engine: str = 'doctr',
    confidence_threshold: Union[float, Dict[str, float]] = FIELD_THRESHOLDS,
    max_length: int = 512,
) -> Dict[str, str]:
    """
    Full improved invoice field extraction pipeline.

    Applies all 4 improvements in sequence:

        1. Load image (handles jpg/png/webp/avif/pdf, PIL.Image directly)
        2. OCR via DocTR (or Tesseract fallback)
        3. Reading order sorting
        4. Confidence-gated LayoutLMv3 extraction
        5. Entity-level cross-field validation

    For native text PDFs (>= 100 chars of embedded text), the first page is
    still rasterised and processed through LayoutLMv3 for field extraction — the
    native text is used only to provide richer OCR word stream to InvoiceCleaner
    fallbacks.

    Parameters
    ----------
    image_or_path : str | pathlib.Path | PIL.Image.Image
        The invoice to process.  Accepts file paths (jpg, png, webp, avif, pdf)
        or a PIL Image object directly.
    extractor_model : LayoutLMv3ForTokenClassification
        Loaded model, already on ``device``, eval mode.
    extractor_processor : LayoutLMv3Processor
        Loaded processor (use_fast=True, local_files_only=True).
    device : torch.device
    id2label : dict {int: str}
    cleaner : InvoiceCleaner
    classifier_model : optional
        Document classifier (e.g. DiT) for type verification.  If None, the
        pipeline skips type checking and proceeds directly to extraction.
    classifier_processor : optional
        Processor for ``classifier_model``.
    ocr_engine : {'doctr', 'tesseract'}
        OCR engine.  Default: 'doctr'.
    confidence_threshold : float | dict[str, float]
        Global threshold or per-field thresholds used by confidence gating.
    max_length : int
        Processor max token length.  Default: 512.

    Returns
    -------
    dict with keys:
        invoice_number, invoice_date, due_date,
        issuer_name, recipient_name, total_amount,
        confidence_scores, validation_notes, source_mode
    """
    # ── Load image ────────────────────────────────────────────────────────
    if isinstance(image_or_path, Image.Image):
        image       = image_or_path.convert('RGB')
        source_mode = 'image_direct'
    else:
        path        = Path(image_or_path)
        image       = _load_image_from_path(path)
        source_mode = 'native_pdf' if path.suffix.lower() == '.pdf' else 'image_ocr'

    # ── OCR ───────────────────────────────────────────────────────────────
    words, bboxes = ocr_image(image, engine=ocr_engine)

    # ── Reading order sort (Improvement 1) ───────────────────────────────
    words, bboxes = sort_reading_order(words, bboxes)

    # ── Confidence-gated extraction (Improvements 2 + 3) ─────────────────
    result = extract_with_confidence_gating(
        image, words, bboxes,
        extractor_model, extractor_processor, device, id2label,
        cleaner,
        model_threshold=confidence_threshold,
        ocr_text=' '.join(words),
        max_length=max_length,
    )

    # ── Validation and correction (Improvement 4) ─────────────────────────
    confidence_scores = result.pop('confidence_scores', {})
    result = validate_and_correct_fields(result, confidence_scores, words)
    result['confidence_scores'] = confidence_scores
    result['source_mode']       = source_mode

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_image_from_path(path: Path) -> Image.Image:
    """
    Load an image or PDF page from disk.

    Supports jpg, png, webp, avif (via Pillow or ImageMagick), and pdf
    (via pdf2image / pymupdf).  Raises RuntimeError with installation
    instructions if a required dependency is missing.
    """
    suffix = path.suffix.lower()

    if suffix == '.pdf':
        # Try pymupdf first (lighter)
        try:
            import fitz  # type: ignore
            import io
            doc = fitz.open(str(path))
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = doc[0].get_pixmap(matrix=mat)
            doc.close()
            return Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')
        except ImportError:
            pass
        # Fall back to pdf2image
        try:
            from pdf2image import convert_from_path  # type: ignore
            pages = convert_from_path(str(path), dpi=200, first_page=1, last_page=1)
            if pages:
                return pages[0].convert('RGB')
        except ImportError:
            pass
        raise RuntimeError(
            f'Cannot rasterise {path.name}.  '
            'Install pymupdf:  pip install pymupdf  '
            'or pdf2image:  pip install pdf2image'
        )

    if suffix == '.avif':
        try:
            return Image.open(path).convert('RGB')
        except Exception:
            pass
        import subprocess, tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.check_call(
            ['convert', str(path), tmp_path], stderr=subprocess.DEVNULL
        )
        img = Image.open(tmp_path).convert('RGB')
        _os.unlink(tmp_path)
        return img

    return Image.open(path).convert('RGB')
