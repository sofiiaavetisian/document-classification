"""
invoice_extraction.py

Main extraction engine for invoice documents.

Orchestrates zone detection, field-specific rules, and validators to extract
structured information from a single invoice document's OCR output.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running this file directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from invoice_rules import (
    INVOICE_NUMBER_ANCHORS,
    INVOICE_DATE_ANCHORS,
    DUE_DATE_ANCHORS,
    ISSUER_ANCHORS,
    RECIPIENT_ANCHORS,
    TOTAL_ANCHORS,
    SUBTOTAL_ANCHORS,
    find_anchor_line,
    find_all_anchor_lines,
    extract_value_after_anchor,
    score_anchor_proximity,
    find_dates_in_text,
    find_amounts_in_text,
    infer_due_date_from_terms,
    INVOICE_NUMBER_RE,
    DATE_RE,
)
from validators import (
    clean_ocr_text,
    validate_date,
    validate_amount,
    validate_invoice_number,
    validate_name,
    issuer_differs_from_recipient,
    compute_field_confidence,
)
from zones import (
    detect_zones,
    get_lines_in_zone,
    lines_to_text,
    get_top_lines,
    ZONE_SELLER,
    ZONE_BUYER,
    ZONE_METADATA,
    ZONE_TOTALS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lines_from_ocr_result(ocr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull the lines list from a cached OCR result dict."""
    return ocr_result.get("lines", [])


def _text_of(line: Dict[str, Any]) -> str:
    return clean_ocr_text(str(line.get("text", "")))


def _extract_invoice_number(
    lines: List[Dict[str, Any]],
    zone_indices: List[int],
    full_text: str,
) -> Optional[str]:
    """
    Look for an invoice number in the metadata zone first, then fall back
    to searching the full text with the invoice number regex.
    """
    # Search metadata zone lines for an anchor keyword on the same line.
    zone_lines = [lines[i] for i in zone_indices if i < len(lines)]
    for line in zone_lines:
        text = _text_of(line)
        value = extract_value_after_anchor(text, INVOICE_NUMBER_ANCHORS)
        if value:
            # Take only the first token after the anchor so we don't grab the whole line.
            candidate = value.split()[0] if value.split() else value
            validated = validate_invoice_number(candidate)
            if validated:
                return validated

    # Fall back: scan all lines for an anchor and grab the value after it.
    all_texts = [_text_of(l) for l in lines]
    anchor_idx = find_anchor_line(all_texts, INVOICE_NUMBER_ANCHORS)
    if anchor_idx is not None:
        value = extract_value_after_anchor(all_texts[anchor_idx], INVOICE_NUMBER_ANCHORS)
        if value:
            candidate = value.split()[0] if value.split() else value
            validated = validate_invoice_number(candidate)
            if validated:
                return validated

        # The value might be on the next line rather than the same line.
        if anchor_idx + 1 < len(all_texts):
            next_line = all_texts[anchor_idx + 1].strip()
            validated = validate_invoice_number(next_line)
            if validated:
                return validated

    # Last resort: regex scan over the full text.
    for match in INVOICE_NUMBER_RE.finditer(full_text):
        validated = validate_invoice_number(match.group(0))
        if validated:
            return validated

    return None


def _extract_date_near_anchors(
    lines: List[Dict[str, Any]],
    anchors: List[str],
    full_text: str,
) -> Optional[str]:
    """
    Generic date extractor that looks for a date near a set of anchor keywords.
    Tries same-line extraction first, then the next line, then falls back to
    scanning the full text.
    """
    all_texts = [_text_of(l) for l in lines]

    # Try to find the value on the same line as the anchor.
    anchor_idx = find_anchor_line(all_texts, anchors)
    if anchor_idx is not None:
        value = extract_value_after_anchor(all_texts[anchor_idx], anchors)
        if value:
            dates = find_dates_in_text(value)
            if dates:
                validated = validate_date(dates[0])
                if validated:
                    return validated

        # Check dates on the same line even without a clean after-anchor value.
        dates = find_dates_in_text(all_texts[anchor_idx])
        if dates:
            validated = validate_date(dates[0])
            if validated:
                return validated

        # Check the next line.
        if anchor_idx + 1 < len(all_texts):
            dates = find_dates_in_text(all_texts[anchor_idx + 1])
            if dates:
                validated = validate_date(dates[0])
                if validated:
                    return validated

    # Scan the full text as a last resort.
    dates = find_dates_in_text(full_text)
    for d in dates:
        validated = validate_date(d)
        if validated:
            return validated

    return None


def _extract_invoice_date(
    lines: List[Dict[str, Any]],
    full_text: str,
) -> Optional[str]:
    return _extract_date_near_anchors(lines, INVOICE_DATE_ANCHORS, full_text)


def _extract_due_date(
    lines: List[Dict[str, Any]],
    full_text: str,
    invoice_date_iso: Optional[str],
) -> tuple[Optional[str], bool]:
    """
    Returns (due_date_iso, was_inferred).
    Tries explicit anchors first. If not found and an invoice date is available,
    tries to infer from payment terms like "Net 30".
    """
    explicit = _extract_date_near_anchors(lines, DUE_DATE_ANCHORS, full_text)
    if explicit:
        return explicit, False

    if invoice_date_iso:
        inferred, flag = infer_due_date_from_terms(invoice_date_iso, full_text)
        if inferred:
            return inferred, flag

    return None, False


def _extract_issuer_name(
    lines: List[Dict[str, Any]],
    seller_zone_indices: List[int],
) -> Optional[str]:
    """
    The issuer is typically one of the first prominent lines in the seller zone.
    We try the top lines of the seller zone and return the first one that
    passes name validation.
    """
    zone_lines = [lines[i] for i in seller_zone_indices if i < len(lines)]

    # Sort by vertical position so we get the topmost lines first.
    zone_lines_sorted = sorted(zone_lines, key=lambda l: float(l.get("top", 0)))

    for line in zone_lines_sorted[:6]:
        text = _text_of(line)
        if not text:
            continue
        # Skip lines that start with an issuer anchor keyword themselves.
        lower = text.lower()
        if any(a in lower for a in ISSUER_ANCHORS):
            # The anchor label itself is not the name; look at the value after it.
            value = extract_value_after_anchor(text, ISSUER_ANCHORS)
            if value:
                validated = validate_name(value)
                if validated:
                    return validated
            continue
        validated = validate_name(text)
        if validated:
            return validated

    # Fall back to the very top lines of the full page.
    all_sorted = sorted(lines, key=lambda l: float(l.get("top", 0)))
    for line in all_sorted[:5]:
        text = _text_of(line)
        validated = validate_name(text)
        if validated:
            return validated

    return None


def _extract_recipient_name(
    lines: List[Dict[str, Any]],
    buyer_zone_indices: List[int],
) -> Optional[str]:
    """
    The recipient name typically appears just after a "bill to" style anchor.
    We find the anchor line and return the next non-empty, non-address line.
    """
    all_texts = [_text_of(l) for l in lines]
    anchor_idx = find_anchor_line(all_texts, RECIPIENT_ANCHORS)

    if anchor_idx is not None:
        # Check if the name is on the same line as the anchor.
        value = extract_value_after_anchor(all_texts[anchor_idx], RECIPIENT_ANCHORS)
        if value:
            validated = validate_name(value)
            if validated:
                return validated

        # Otherwise look at the lines immediately after the anchor.
        for i in range(anchor_idx + 1, min(anchor_idx + 5, len(all_texts))):
            text = all_texts[i].strip()
            if not text:
                continue
            validated = validate_name(text)
            if validated:
                return validated

    # Fall back to buyer zone lines if no anchor was found.
    zone_lines = [lines[i] for i in buyer_zone_indices if i < len(lines)]
    zone_sorted = sorted(zone_lines, key=lambda l: float(l.get("top", 0)))
    for line in zone_sorted:
        text = _text_of(line)
        lower = text.lower()
        # Skip the anchor line itself.
        if any(a in lower for a in RECIPIENT_ANCHORS):
            continue
        validated = validate_name(text)
        if validated:
            return validated

    return None


def _extract_total_amount(
    lines: List[Dict[str, Any]],
    totals_zone_indices: List[int],
    full_text: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (amount_string, currency_symbol).
    Prefers candidates near total anchors in the totals zone.
    Avoids subtotal/tax lines when a grand total line is present.
    """
    all_texts = [_text_of(l) for l in lines]

    # Score each candidate amount by anchor proximity and zone membership.
    best_amount: Optional[str] = None
    best_symbol: Optional[str] = None
    best_score: float = -1.0

    # Search total anchor lines first.
    anchor_indices = find_all_anchor_lines(all_texts, TOTAL_ANCHORS)

    for anchor_idx in anchor_indices:
        anchor_text = all_texts[anchor_idx].lower()

        # Skip lines that are clearly subtotals or taxes.
        if any(sub in anchor_text for sub in SUBTOTAL_ANCHORS):
            continue

        # Look for an amount on the same line as the anchor.
        candidates = find_amounts_in_text(all_texts[anchor_idx])
        for raw in candidates:
            result = validate_amount(raw)
            if result is None:
                continue
            amount_str, symbol = result

            # Lines in the totals zone get a small bonus.
            zone_bonus = 0.2 if anchor_idx in totals_zone_indices else 0.0

            # Prefer "grand total" and "amount due" over plain "total".
            priority_anchors = ["grand total", "amount due", "balance due", "total due"]
            priority_bonus = 0.3 if any(p in anchor_text for p in priority_anchors) else 0.0

            score = 1.0 + zone_bonus + priority_bonus
            if score > best_score:
                best_score = score
                best_amount = amount_str
                best_symbol = symbol

        # If nothing on the same line, check the next line.
        if best_amount is None and anchor_idx + 1 < len(all_texts):
            candidates = find_amounts_in_text(all_texts[anchor_idx + 1])
            for raw in candidates:
                result = validate_amount(raw)
                if result is None:
                    continue
                amount_str, symbol = result
                score = score_anchor_proximity(anchor_idx + 1, anchor_idx)
                if score > best_score:
                    best_score = score
                    best_amount = amount_str
                    best_symbol = symbol

    # Last resort: largest plausible amount in the totals zone.
    if best_amount is None and totals_zone_indices:
        zone_text = lines_to_text(get_lines_in_zone(lines, totals_zone_indices))
        candidates = find_amounts_in_text(zone_text)
        for raw in candidates:
            result = validate_amount(raw)
            if result is None:
                continue
            amount_str, symbol = result
            try:
                val = float(amount_str)
                if best_amount is None or val > float(best_amount):
                    best_amount = amount_str
                    best_symbol = symbol
            except ValueError:
                continue

    return best_amount, best_symbol


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_invoice_fields(
    ocr_result: Dict[str, Any],
    doc_id: str = "",
    predicted_class: str = "invoice",
    predicted_confidence: Optional[float] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Extract all required fields from a single invoice document's OCR result.

    Parameters
    ----------
    ocr_result : dict
        The cached OCR result dict from src.ocr_engine (load_ocr_result).
    doc_id : str
        Document identifier, included in the output for traceability.
    predicted_class : str
        The classifier's predicted class label, included in the output.
    predicted_confidence : float, optional
        Classifier confidence score, included if available.
    debug : bool
        If True, include a rule_trace field in the output with intermediate values.

    Returns
    -------
    dict
        Extraction result with all required fields and confidence score.
    """
    lines = _lines_from_ocr_result(ocr_result)
    full_text = str(ocr_result.get("full_text", ""))
    image_width = int(ocr_result.get("image_width", 1))
    image_height = int(ocr_result.get("image_height", 1))

    trace: Dict[str, Any] = {}

    # Detect spatial zones so field extractors can prefer zone-appropriate lines.
    page_zones = detect_zones(lines, image_width, image_height)

    if debug:
        from zones import zone_summary
        trace["zone_summary"] = zone_summary(page_zones)

    # Extract each field in order of dependency (date before due_date).
    invoice_number = _extract_invoice_number(lines, page_zones.metadata, full_text)
    invoice_date   = _extract_invoice_date(lines, full_text)
    due_date, due_date_inferred = _extract_due_date(lines, full_text, invoice_date)
    issuer_name    = _extract_issuer_name(lines, page_zones.seller)
    recipient_name = _extract_recipient_name(lines, page_zones.buyer)
    total_amount, currency_symbol = _extract_total_amount(
        lines, page_zones.totals, full_text
    )

    if debug:
        trace["invoice_number_raw"] = invoice_number
        trace["invoice_date_raw"]   = invoice_date
        trace["due_date_raw"]       = due_date
        trace["issuer_raw"]         = issuer_name
        trace["recipient_raw"]      = recipient_name
        trace["total_raw"]          = total_amount

    # Sanity check: warn if issuer and recipient resolved to the same string.
    if not issuer_differs_from_recipient(issuer_name, recipient_name):
        if debug:
            trace["warning"] = "issuer and recipient resolved to the same value"
        recipient_name = None

    extracted = {
        "invoice_number": invoice_number,
        "invoice_date":   invoice_date,
        "due_date":       due_date,
        "issuer_name":    issuer_name,
        "recipient_name": recipient_name,
        "total_amount":   total_amount,
    }

    confidence = compute_field_confidence(extracted)

    result: Dict[str, Any] = {
        "doc_id":                    doc_id,
        "predicted_class":           predicted_class,
        "predicted_class_confidence": predicted_confidence,
        "invoice_number":            invoice_number,
        "invoice_date":              invoice_date,
        "due_date":                  due_date,
        "due_date_inferred":         due_date_inferred,
        "issuer_name":               issuer_name,
        "recipient_name":            recipient_name,
        "total_amount":              total_amount,
        "currency_symbol":           currency_symbol,
        "extraction_confidence":     confidence,
    }

    if debug:
        result["rule_trace"] = trace

    return result


def extract_batch(
    doc_ids: List[str],
    predicted_class: str = "invoice",
    cfg=None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run extraction on a list of doc_ids using cached OCR results.

    Parameters
    ----------
    doc_ids : list of str
        Document IDs whose OCR caches exist under data/interim/ocr/parsed/.
    predicted_class : str
        Label to attach to all results.
    cfg : OCRConfig, optional
        Config object passed to the OCR loader. Uses defaults if None.
    debug : bool
        If True, include rule traces in the output.

    Returns
    -------
    list of dicts
        One extraction result dict per document.
    """
    # Import here to avoid circular imports when testing modules individually.
    from ocr_engine import load_ocr_result

    results = []
    for doc_id in doc_ids:
        try:
            ocr_result = load_ocr_result(doc_id, cfg=cfg)
            result = extract_invoice_fields(
                ocr_result,
                doc_id=doc_id,
                predicted_class=predicted_class,
                debug=debug,
            )
        except FileNotFoundError:
            result = {
                "doc_id": doc_id,
                "predicted_class": predicted_class,
                "error": "OCR cache not found",
                "invoice_number": None,
                "invoice_date": None,
                "due_date": None,
                "due_date_inferred": False,
                "issuer_name": None,
                "recipient_name": None,
                "total_amount": None,
                "currency_symbol": None,
                "extraction_confidence": 0.0,
            }
        except Exception as e:
            result = {
                "doc_id": doc_id,
                "predicted_class": predicted_class,
                "error": str(e),
                "invoice_number": None,
                "invoice_date": None,
                "due_date": None,
                "due_date_inferred": False,
                "issuer_name": None,
                "recipient_name": None,
                "total_amount": None,
                "currency_symbol": None,
                "extraction_confidence": 0.0,
            }
        results.append(result)

    return results