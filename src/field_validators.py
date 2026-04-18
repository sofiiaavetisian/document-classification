"""
validators.py

Validation and cleaning helpers for extracted invoice fields.
Each function takes a raw string candidate and returns either a cleaned
value or None if the candidate fails validation.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from dateutil import parser as dateutil_parser


# Compiled once at import time so they are not recompiled on every call.

_PHONE_RE = re.compile(
    r"^[\+\(]?[\d\s\-\.\(\)]{7,}\d$"
)

_CURRENCY_RE = re.compile(
    r"^[$€£¥₹]?\s*[\d]{1,3}(?:[,\.\s]\d{3})*(?:[.,]\d{1,2})?\s*[$€£¥₹]?$"
)

_OCR_GARBAGE_RE = re.compile(r"[|\\`~^]")

_PURE_DIGITS_RE = re.compile(r"^\d+$")

# Lines that look like postal addresses rather than company or person names.
_ADDRESS_LINE_RE = re.compile(
    r"\b(street|st\.|avenue|ave\.|road|rd\.|boulevard|blvd|suite|ste\.|"
    r"floor|fl\.|po\s+box|p\.o\.|zip|postal|\d{5})\b",
    re.IGNORECASE,
)


def clean_ocr_text(text: str) -> str:
    """Remove common OCR artifacts and normalize whitespace."""
    if not text:
        return ""
    text = _OCR_GARBAGE_RE.sub(" ", text)
    text = " ".join(text.split())
    return text.strip()


def validate_date(raw: str) -> Optional[str]:
    """
    Try to parse a date string in any common format.
    Returns ISO format (YYYY-MM-DD) on success, None otherwise.
    """
    if not raw:
        return None
    cleaned = clean_ocr_text(raw)
    if not cleaned:
        return None
    try:
        # dayfirst=False prefers MM/DD/YYYY when the format is ambiguous,
        # which is the more common style in US invoices.
        dt = dateutil_parser.parse(cleaned, dayfirst=False, fuzzy=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def validate_amount(raw: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Validate and normalize a currency amount string.
    Returns (normalized_amount, currency_symbol) or None if invalid.
    Example: "$ 1,234.56" returns ("1234.56", "$").
    """
    if not raw:
        return None

    cleaned = clean_ocr_text(raw)

    symbol_match = re.search(r"[$€£¥₹]", cleaned)
    symbol = symbol_match.group(0) if symbol_match else None

    numeric_str = re.sub(r"[$€£¥₹\s]", "", cleaned)
    numeric_str = numeric_str.replace(",", "")

    if not _CURRENCY_RE.match(cleaned):
        return None

    try:
        value = float(numeric_str)
        if value <= 0 or value > 999_999_999:
            return None
        return (f"{value:.2f}", symbol)
    except ValueError:
        return None


def is_phone_number(text: str) -> bool:
    """Return True if text looks like a phone number."""
    if not text:
        return False
    cleaned = clean_ocr_text(text)
    digits_only = re.sub(r"\D", "", cleaned)
    if len(digits_only) >= 7 and _PHONE_RE.match(cleaned):
        return True
    # 10 or 11 digit pure number strings are almost always phone numbers.
    if _PURE_DIGITS_RE.match(digits_only) and len(digits_only) in (10, 11):
        return True
    return False


def is_date_like(text: str) -> bool:
    """Return True if text parses successfully as a date."""
    return validate_date(text) is not None


def validate_invoice_number(raw: str) -> Optional[str]:
    """
    Validate a candidate invoice number string.
    Rejects phone numbers, pure dates, and strings with no alphanumeric content.
    Returns the cleaned string if valid, None otherwise.
    """
    if not raw:
        return None

    cleaned = clean_ocr_text(raw)

    if len(cleaned) < 2 or len(cleaned) > 40:
        return None

    if not re.search(r"[A-Za-z0-9]", cleaned):
        return None

    if is_phone_number(cleaned):
        return None

    if is_date_like(cleaned):
        return None

    return cleaned


def validate_name(raw: str) -> Optional[str]:
    """
    Validate a candidate name string for issuer or recipient.
    Rejects address lines, purely numeric strings, and very short strings.
    Returns the cleaned string if valid, None otherwise.
    """
    if not raw:
        return None

    cleaned = clean_ocr_text(raw)

    if len(cleaned) < 2:
        return None

    if _PURE_DIGITS_RE.match(re.sub(r"\s", "", cleaned)):
        return None

    if _ADDRESS_LINE_RE.search(cleaned):
        return None

    return cleaned


def issuer_differs_from_recipient(issuer: Optional[str], recipient: Optional[str]) -> bool:
    """
    Return True if issuer and recipient are meaningfully different.
    If they normalize to the same string, extraction likely went wrong.
    """
    if not issuer or not recipient:
        return True

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    return _norm(issuer) != _norm(recipient)


def compute_field_confidence(extracted: dict) -> float:
    """
    Compute an overall extraction confidence score between 0 and 1.
    Each field contributes a fixed weight if it has a non-empty value.
    """
    field_weights = {
        "invoice_number": 0.20,
        "invoice_date":   0.20,
        "due_date":       0.10,
        "issuer_name":    0.20,
        "recipient_name": 0.20,
        "total_amount":   0.10,
    }

    score = 0.0
    for field, weight in field_weights.items():
        value = extracted.get(field)
        if value and str(value).strip():
            score += weight

    return round(score, 3)