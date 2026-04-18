"""
pdf_extractor.py
================
Utilities for native-text PDF handling.

is_native_pdf()          — check whether a PDF has extractable text
_extract_fields_from_text() — regex-based field extraction from PDF text

This module is intentionally standalone (no ML imports) so it loads fast
and is safe to import before the GPU models are initialised.

The _extract_fields_from_text() function is copied verbatim from
notebook 13 cell 14, which was debugged and validated against 5 real-world
invoice PDFs with the following layouts:

  invoice-0-4.pdf  — INVOICE # BPXINV-00550, DATE: 23.05.2021, TO: recipient
  invoice-1-3.pdf  — Invoice No. 1213, bare date 16.12.2021 before label
  invoice-2-1.pdf  — INVOICE 0012820, ORDER DATE / ORDER NUMBER / DUE DATE table
  invoice-3-0.pdf  — Invoice 4235, Date / To / Ship To 3-column header
  invoice-7-0.pdf  — Invoice number: NP#00183, Date: / Due-date: labels

Note: several of these invoices produce imperfect results even with the regex
extractor — especially those that mix columns, use non-standard date formats,
or have address blocks that bleed into name fields. The InvoiceCleaner (used
for the LayoutLMv3 path) does not run here because we are extracting from
plain text, not from an OCR token stream.
"""

import re
from pathlib import Path
from typing import Dict

import fitz  # pymupdf

_PDF_TEXT_THRESHOLD = 100

# ── Date pattern ────────────────────────────────────────────────────────────
# Copied exactly from notebook 13 so the same formats are recognised.
_DATE = (
    r'\d{1,2}\.\d{1,2}\.\d{4}'
    r'|\d{4}-\d{2}-\d{2}'
    r'|\d{1,2}/\d{1,2}/\d{2,4}'
    r'|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
    r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
    r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}'
    r'|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
    r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
    r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}'
)
_DATE_RE = re.compile(_DATE, re.IGNORECASE)


def is_native_pdf(path: str) -> bool:
    """
    Return True if the PDF at *path* has >= _PDF_TEXT_THRESHOLD extractable
    characters across all pages (i.e. it is not a scanned image-only PDF).
    """
    doc = fitz.open(str(path))
    total_chars = sum(len(page.get_text()) for page in doc)
    doc.close()
    return total_chars >= _PDF_TEXT_THRESHOLD


def extract_text_from_pdf(path: str) -> str:
    """Return the full text of all pages joined by newlines."""
    doc = fitz.open(str(path))
    text = '\n'.join(page.get_text() for page in doc)
    doc.close()
    return text


def _extract_fields_from_text(text: str) -> Dict[str, str]:
    """
    Regex extraction from native PDF text.
    Handles 5 date formats, 3-column table layouts, labelled and unlabelled fields.

    Copied verbatim from notebook 13, cell 14. Do not edit without re-validating
    against the five test PDFs listed in the module docstring.

    Returns
    -------
    dict with lowercase keys (same key set as InvoiceCleaner.clean):
        invoice_number, invoice_date, due_date,
        issuer_name, recipient_name, total_amount
    Each value is a non-empty string or the key is absent from the dict.
    """
    fields: Dict[str, str] = {}
    lines = [l.strip() for l in text.split('\n')]

    # ── Invoice number ─────────────────────────────────────────────────────
    for pat in [
        r'invoice\s*(?:no\.?|number|#|num\.?)\s*:?\s+([A-Z0-9#][A-Z0-9#\-/]+)',
        r'INVOICE\s+#\s+([A-Z0-9][A-Z0-9\-]+)',
        r'^INVOICE\s+(\d{4,})',
        r'^invoice\s+(?:no\.?\s+)?(\d{3,})',
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip()
            if any(c.isdigit() for c in val):
                fields['invoice_number'] = val
                break

    # ── Invoice date ───────────────────────────────────────────────────────
    for pat in [
        r'(?:invoice\s+)?date\s*:?\s*\n?\s*(' + _DATE + r')',
        r'(?:issued?|created)\s*:?\s*(' + _DATE + r')',
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            fields['invoice_date'] = m.group(1).strip()
            break

    if 'invoice_date' not in fields:
        m = re.search(r'^(' + _DATE + r')\s*\n\s*invoice', text, re.IGNORECASE | re.MULTILINE)
        if m:
            fields['invoice_date'] = m.group(1).strip()

    # 3-column table: Date / To / Ship To headers then values on next lines
    if 'invoice_date' not in fields:
        for i, line in enumerate(lines):
            if line.lower() == 'date' and i + 3 < len(lines):
                if lines[i + 1].lower() in ('to', 'ship to') or lines[i + 2].lower() in ('to', 'ship to'):
                    for j in range(i + 2, min(i + 5, len(lines))):
                        if _DATE_RE.match(lines[j]):
                            fields['invoice_date'] = lines[j].strip()
                            if j + 1 < len(lines):
                                candidate = lines[j + 1].strip()
                                if (candidate
                                        and 'same as' not in candidate.lower()
                                        and candidate.lower() not in ('to', 'ship to', 'none')
                                        and not re.match(r'^\d+\s+[A-Z]', candidate)
                                        and len(candidate) < 60):
                                    fields['recipient_name'] = candidate
                            break
                break

    # ── Due date ───────────────────────────────────────────────────────────
    for pat in [
        r'due[\s\-]date\s*:?\s*(' + _DATE + r')',
        r'payment\s+due\s*:?\s*(' + _DATE + r')',
        r'total\s+due\s+by\s*[^\n]*?(' + _DATE + r')',
        r'pay(?:able)?\s+(?:by|on)\s*:?\s*(' + _DATE + r')',
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip()
            if val != fields.get('invoice_date', ''):
                fields['due_date'] = val
                break

    if 'due_date' not in fields:
        m = re.search(
            r'order\s+date\s*\n+order\s+number\s*\n+due\s+date\s*\n+'
            r'(' + _DATE + r')\s*\n+\S+\s*\n+(' + _DATE + r')',
            text, re.IGNORECASE | re.MULTILINE
        )
        if m:
            if 'invoice_date' not in fields:
                fields['invoice_date'] = m.group(1).strip()
            fields['due_date'] = m.group(2).strip()

    # ── Issuer name ────────────────────────────────────────────────────────
    # Scan the first 20 lines for the first non-label, non-address, non-numeric line.
    # This heuristic works well when the issuer name is the first real text line.
    # It will fail for invoices that start with a logo description or a long header.
    SKIP = re.compile(
        r'^(?:tel|fax|phone|email|web|http|www|invoice|date|lorem|we\s|'
        r'order|ship|bill|to\b|from\b|due|total|sub)',
        re.IGNORECASE
    )
    PHONE = re.compile(r'^\+?\d[\d\s\-().]{5,}$')
    for line in lines[:20]:
        if not line or SKIP.match(line):
            continue
        if PHONE.match(line):
            continue
        if '@' in line or line.lower().startswith('http'):
            continue
        if re.match(r'^\d+\s+[A-Za-z]', line):
            continue
        if re.match(r'^\d', line):
            continue
        if len(line) > 60:
            continue
        fields['issuer_name'] = line
        break

    # ── Recipient name ─────────────────────────────────────────────────────
    if 'recipient_name' not in fields:
        for pat in [
            r'bill\s+to\s*:?\s*\n\s*(.+)',
            r'billed\s+to\s*:?\s*\n\s*(.+)',
            r'ship\s+to\s*:?\s*\n\s*([A-Z][^\n]{1,50})',
            r'^to\s*\n\s*([A-Z][^\n]{1,50})',
        ]:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                candidate = m.group(1).strip()
                if (candidate
                        and not re.match(r'^\d+\s+[A-Z]', candidate)
                        and '@' not in candidate
                        and 'same as' not in candidate.lower()
                        and candidate.lower() not in ('ship to', 'bill to', 'to', 'none')
                        and len(candidate) < 60):
                    fields['recipient_name'] = candidate
                    break

    # ── Total amount ───────────────────────────────────────────────────────
    _CUR = r'(?:\s*(?:EUR|USD|GBP|CAD|AUD|CHF))?'
    for pat in [
        r'total\s+due\s+by[^\n]*\n\s*(\d[\d,\.]+)' + _CUR,
        r'total\s+due\s*:?\s*\n?\s*(\d[\d,\.]+)' + _CUR,
        r'amount\s+due\s*:?\s*\n?\s*(\d[\d,\.]+)' + _CUR,
        r'grand\s+total\s*:?\s*\$?\s*(\d[\d,\.]+)' + _CUR,
        r'\btotal\b\s*\**\s*\$?\s*(\d[\d,\.]+)' + _CUR,
        r'balance\s+due\s*:?\s*(\d[\d,\.]+)' + _CUR,
        r'total\s+due\s*\n\s*(\d[\d,\.]+)' + _CUR,
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            amount = m.group(1).strip().rstrip('.,')
            try:
                v = float(amount.replace(',', ''))
                if v < 1:
                    continue
            except ValueError:
                continue
            # Reject small decimals that look like date fragments (e.g. 24.09)
            if re.match(r'^\d{1,2}\.\d{2}$', amount) and float(amount) < 50:
                continue
            context = text[max(0, m.start() - 20):m.end() + 20]
            cur_m = re.search(r'\b(EUR|USD|GBP|CAD|AUD)\b|[$€£]', context, re.IGNORECASE)
            currency = ''
            if cur_m:
                c = cur_m.group(0).upper()
                currency = {'$': 'USD', '€': 'EUR', '£': 'GBP'}.get(c, c)
            fields['total_amount'] = f'{amount} {currency}'.strip() if currency else amount
            break

    return fields
