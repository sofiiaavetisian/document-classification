"""
invoice_rules.py

Regex patterns and anchor keyword lists for locating invoice fields in OCR text.
These are used by the extraction engine to find candidate values near known anchors.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Anchor keyword lists
# Each list contains lowercase strings that signal a nearby field value.
# ---------------------------------------------------------------------------

INVOICE_NUMBER_ANCHORS = [
    "invoice number", "invoice no", "invoice #", "invoice no.",
    "inv number", "inv no", "inv #", "inv no.", "inv.",
    "reference number", "ref no", "ref #", "document number",
]

INVOICE_DATE_ANCHORS = [
    "invoice date", "date issued", "issue date", "date of issue",
    "billing date", "bill date", "date",
]

DUE_DATE_ANCHORS = [
    "due date", "payment due", "payment due date", "payable by",
    "pay by", "due by", "due on", "payment date",
]

ISSUER_ANCHORS = [
    "from", "bill from", "seller", "vendor", "supplier",
    "service provider", "issued by",
]

RECIPIENT_ANCHORS = [
    "bill to", "billed to", "invoice to", "sold to",
    "ship to", "client", "customer", "pay to",
]

TOTAL_ANCHORS = [
    "grand total", "amount due", "balance due", "total due",
    "total amount", "amount payable", "net total",
    "total payable", "invoice total", "total",
]

# Anchors that indicate a subtotal or partial amount we want to avoid
# when a proper total is present.
SUBTOTAL_ANCHORS = [
    "subtotal", "sub total", "sub-total",
    "tax", "vat", "gst", "hst", "discount",
    "shipping", "freight", "handling",
]

PAYMENT_TERMS_ANCHORS = [
    "net 30", "net 60", "net 15", "net 7",
    "net30", "net60", "payment terms", "terms",
]


# ---------------------------------------------------------------------------
# Regex patterns for field values
# ---------------------------------------------------------------------------

# Invoice number: alphanumeric with optional separators, 2-30 chars.
INVOICE_NUMBER_RE = re.compile(
    r"\b([A-Z]{0,4}[-/]?\d{3,10}(?:[-/][A-Z0-9]{1,8})?)\b",
    re.IGNORECASE,
)

# Matches many common date formats used in invoices.
DATE_RE = re.compile(
    r"""
    \b(
        # MM/DD/YYYY or DD/MM/YYYY or MM-DD-YYYY etc.
        \d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}
        |
        # YYYY-MM-DD (ISO)
        \d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}
        |
        # Month name formats: January 15, 2024 or 15 Jan 2024
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
           Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|
           Dec(?:ember)?)
        [\s\.\,]{0,2}\d{1,2}[\s\.\,]{0,2}\d{2,4}
        |
        \d{1,2}[\s\.\,]{0,2}
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
           Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|
           Dec(?:ember)?)
        [\s\.\,]{0,2}\d{2,4}
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Currency amount: requires a currency symbol OR at least 4 digits with a decimal.
# This prevents matching bare short numbers like "202" or "4".
AMOUNT_RE = re.compile(
    r"""
    (?:
        # option 1: has a currency symbol
        [$€£¥₹]\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?
        |
        # option 2: no symbol but must have thousands separator or decimal to be plausible
        \d{1,3}(?:,\d{3})+(?:\.\d{1,2})?
        |
        # option 3: plain decimal number that is at least 4 digits total
        \d{4,}(?:\.\d{1,2})?
        |
        # option 4: trailing currency symbol
        \d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?\s*[$€£¥₹]
    )
    """,
    re.VERBOSE,
)

# Detects payment term strings like "Net 30" for due date inference.
PAYMENT_TERMS_RE = re.compile(
    r"\bnet\s*(\d+)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Anchor search helpers
# ---------------------------------------------------------------------------

def find_anchor_line(lines: List[str], anchors: List[str]) -> Optional[int]:
    """
    Search a list of text lines for the first line containing any anchor keyword.
    Returns the line index or None if no anchor is found.
    """
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for anchor in anchors:
            if anchor in line_lower:
                return i
    return None


def find_all_anchor_lines(lines: List[str], anchors: List[str]) -> List[int]:
    """
    Return indices of all lines that contain any of the given anchor keywords.
    Useful when a field anchor may appear multiple times.
    """
    hits = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for anchor in anchors:
            if anchor in line_lower:
                hits.append(i)
                break
    return hits


def extract_value_after_anchor(line: str, anchors: List[str]) -> Optional[str]:
    """
    If a line contains an anchor keyword, return the text that appears after it
    on the same line. Returns None if no anchor is found or nothing follows it.

    Example: "Invoice No: INV-001" with anchor "invoice no" returns "INV-001".
    """
    line_lower = line.lower()
    for anchor in anchors:
        idx = line_lower.find(anchor)
        if idx != -1:
            after = line[idx + len(anchor):].strip()
            after = after.lstrip(":# \t")
            if after:
                return after.strip()
    return None


def score_anchor_proximity(line_index: int, anchor_index: int, max_distance: int = 5) -> float:
    """
    Return a proximity score between 0 and 1 based on how close a candidate
    line is to an anchor line. Closer is better. Returns 0 if beyond max_distance.
    """
    distance = abs(line_index - anchor_index)
    if distance > max_distance:
        return 0.0
    return 1.0 - (distance / max_distance)


def find_dates_in_text(text: str) -> List[str]:
    """Return all date-like strings found anywhere in a block of text."""
    return [m.group(0).strip() for m in DATE_RE.finditer(text)]


def find_amounts_in_text(text: str) -> List[str]:
    """Return all currency amount strings found anywhere in a block of text."""
    return [m.group(0).strip() for m in AMOUNT_RE.finditer(text)
            if m.group(0).strip()]


def infer_due_date_from_terms(invoice_date_iso: str, text: str) -> Tuple[Optional[str], bool]:
    """
    Try to infer a due date from payment terms like "Net 30" when no explicit
    due date anchor is found.

    Returns (inferred_iso_date, True) if successful, (None, False) otherwise.
    The boolean flag indicates the date was inferred rather than extracted directly.
    """
    match = PAYMENT_TERMS_RE.search(text)
    if not match:
        return None, False

    days = int(match.group(1))

    try:
        from dateutil import parser as dateutil_parser
        from datetime import timedelta
        base = dateutil_parser.parse(invoice_date_iso)
        due = base + timedelta(days=days)
        return due.strftime("%Y-%m-%d"), True
    except Exception:
        return None, False