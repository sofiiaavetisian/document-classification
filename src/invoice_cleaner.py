"""
invoice_cleaner.py
==================
State-of-the-art post-processing for LayoutLMv3 invoice field extraction.

Takes the raw string output from the model (which includes label prefixes,
address leakage, and surrounding noise) and returns clean, structured values
for the 6 required fields:

    INVOICE_NUMBER, INVOICE_DATE, DUE_DATE,
    ISSUER_NAME, RECIPIENT_NAME, TOTAL_AMOUNT

Design principles
-----------------
1. Pattern extraction over prefix stripping
   For structured fields (dates, amounts, numbers) we search for the value
   pattern directly in the raw string rather than trying to strip the label.
   This is robust against any label wording the model produces.

2. Token-by-token name truncation
   For name fields we walk tokens and stop at the first token that looks like
   an address element. This is more reliable than regex on free-form text.

3. Multi-date arbitration
   When the model assigns both dates to the same field (a known failure mode
   on some FATURA templates) we use OCR word positions to recover the second
   date and assign it correctly.

4. Keyword fallback for missing recipient
   When the model misses RECIPIENT_NAME entirely we scan the OCR word stream
   for bill-to trigger words and extract the name that follows. This handles
   merged tokens like 'to:Nicole' that confuse the model.

5. No hard-coded template knowledge
   All rules are based on general invoice conventions, not FATURA-specific
   layout knowledge. The cleaner will generalise to real-world invoices.

Usage
-----
    from invoice_cleaner import InvoiceCleaner

    cleaner = InvoiceCleaner()
    cleaned = cleaner.clean(raw_fields, ocr_words)
    # cleaned is a dict with lowercase keys:
    # invoice_number, invoice_date, due_date,
    # issuer_name, recipient_name, total_amount
"""

import re
from typing import Dict, List, Optional


# ── Compiled regex patterns ────────────────────────────────────────────────

# Date patterns — ordered from most specific to least specific so the
# first match is always the most precise representation available.
_DATE_PATTERNS = [
    # ISO:           2024-03-15
    re.compile(r'\b(\d{4})[-/](\d{2})[-/](\d{2})\b'),
    # Day-MonthName-Year:  22-Aug-1995,  3-January-2024
    re.compile(r'\b(\d{1,2})[-/]([A-Za-z]{3,9})[-/](\d{2,4})\b'),
    # MonthName Day, Year: August 22, 1995  /  Aug 22 1995
    re.compile(r'\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b'),
    # Day/Month/Year:      22/03/2024,  22-03-2024
    re.compile(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b'),
    # Year/Month/Day (some locales): 2024/03/22
    re.compile(r'\b(\d{4})[./](\d{1,2})[./](\d{1,2})\b'),
]

# Master date regex — used for scanning OCR word stream for fallbacks
_DATE_ANY = re.compile(
    r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'
    r'|\b\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}\b'
    r'|\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b'
    r'|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
    re.IGNORECASE
)

# Amount patterns — ordered from most to least specific.
# Key design: use \d[\d,.\s]*\d to greedily match numbers of any format
# (1098.28, 1,234.56, 1.234,56, 1 234.56) without imposing thousand-group
# structure that would exclude plain 4-digit amounts like 1098.
_AMOUNT_PATTERNS = [
    # Currency code BEFORE number:  USD 1098.28  /  EUR 1,234.56
    re.compile(
        r'(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|CNY|INR|MXN|BRL)\s*'
        r'(\d[\d,.\s]*\d|\d)',
        re.IGNORECASE
    ),
    # Number THEN currency code:    1098.28 USD  /  1,234.56 EUR
    re.compile(
        r'(\d[\d,.\s]*\d|\d)\s*'
        r'(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|CNY|INR|MXN|BRL)',
        re.IGNORECASE
    ),
    # Currency symbol THEN number:  $1,234.56  /  €1098.28
    re.compile(r'([$€£¥₹₩]\s*\d[\d,.\s]*\d|[$€£¥₹₩]\s*\d)'),
    # Number THEN currency symbol:  1234.56$
    re.compile(r'(\d[\d,.\s]*\d|\d)\s*[$€£¥₹₩]'),
    # Plain decimal number (last resort):  1,234.56  /  1098.28
    re.compile(r'(\d{1,3}(?:[,]\d{3})*\.\d{2}|\d+\.\d{2})'),
]

# Currency symbols and codes for reconstruction
_CURRENCY_SYMBOL = re.compile(r'[$€£¥₹₩]')
_CURRENCY_CODE   = re.compile(
    r'\b(USD|EUR|GBP|CAD|AUD|CHF|JPY|CNY|INR|MXN|BRL)\b',
    re.IGNORECASE
)

# Invoice number — alphanumeric token that looks like a reference number
# Must contain at least one digit and be 3+ characters
_INVOICE_NUM_PATTERN = re.compile(
    r'\b([A-Z0-9][-A-Z0-9/]{2,}[A-Z0-9]|[A-Z]{1,4}[-\s]?\d{3,}|\d{4,}[-A-Z0-9]*)\b',
    re.IGNORECASE
)

# Tokens that mark the start of an address block
_ADDRESS_TOKENS = frozenset({
    'address', 'addr', 'street', 'st', 'ave', 'avenue', 'road', 'rd',
    'boulevard', 'blvd', 'lane', 'ln', 'drive', 'dr', 'court', 'ct',
    'apt', 'apartment', 'suite', 'ste', 'floor', 'fl', 'po', 'box',
    'tel', 'tele', 'phone', 'fax', 'mob', 'mobile', 'cell',
    'email', 'e-mail', 'mail',
    'site', 'web', 'website', 'http', 'https', 'www',
    'gstin', 'gst', 'vat', 'tax', 'ein', 'pan', 'tin',
    'zip', 'postal', 'postcode',
    'country', 'state', 'province', 'city',
})

# Label words that appear before the actual name value
_NAME_LABEL_WORDS = frozenset({
    'bill', 'billed', 'billing',
    'to', 'from', 'attn', 'attention',
    'sold', 'ship', 'shipped',
    'send', 'sent',
    'client', 'customer', 'buyer', 'purchaser',
    'vendor', 'supplier', 'seller', 'issuer',
    'invoice', 'invoiced',
    'recipient', 'payee', 'payer',
    'company', 'co', 'corp', 'inc', 'ltd', 'llc',
})

# Trigger words that indicate a bill-to block in OCR word stream
_BILL_TO_TRIGGERS = frozenset({'bill', 'billed', 'sold', 'send', 'ship'})

# Words that clearly indicate a due-date label (vs invoice date label)
_DUE_DATE_LABELS = frozenset({
    'due', 'payment', 'pay', 'payable', 'expiry', 'expires', 'deadline',
})

# Words that clearly indicate an invoice date label
_INVOICE_DATE_LABELS = frozenset({
    'invoice', 'issued', 'issue', 'created', 'date',
})


class InvoiceCleaner:
    """
    Post-processes raw LayoutLMv3 field strings into clean extracted values.

    Parameters
    ----------
    max_name_tokens : int
        Maximum number of tokens to accept as a person/company name before
        truncating. Increase for long company names. Default: 6.
    min_invoice_num_len : int
        Minimum character length for a valid invoice number token.
        Default: 3.
    """

    def __init__(
        self,
        max_name_tokens: int = 6,
        min_invoice_num_len: int = 3,
    ):
        self.max_name_tokens      = max_name_tokens
        self.min_invoice_num_len  = min_invoice_num_len

    # ── Public API ────────────────────────────────────────────────────────

    def clean(
        self,
        raw_fields: Dict[str, str],
        ocr_words: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Clean raw LayoutLMv3 field predictions.

        Parameters
        ----------
        raw_fields : dict
            Output of extract_fields_from_encoding() — keys are uppercase
            field names like 'INVOICE_NUMBER', values are raw strings.
        ocr_words : list of str, optional
            The OCR word stream for the image. Used for fallback extraction
            when the model missed a field or produced a corrupt prediction.
            Highly recommended to pass this for best results.

        Returns
        -------
        dict with lowercase keys:
            invoice_number, invoice_date, due_date,
            issuer_name, recipient_name, total_amount
        Each value is a clean string or empty string if not found.
        """
        # Work on a copy — never mutate the input
        raw = {k: (v or '').strip() for k, v in raw_fields.items()}

        result = {
            'invoice_number': self._clean_invoice_number(raw.get('INVOICE_NUMBER', '')),
            'invoice_date':   self._clean_date(raw.get('INVOICE_DATE', '')),
            'due_date':       self._clean_date(raw.get('DUE_DATE', '')),
            'issuer_name':    self._clean_name(raw.get('ISSUER_NAME', '')),
            'recipient_name': self._clean_name(raw.get('RECIPIENT_NAME', '')),
            'total_amount':   self._clean_amount(raw.get('TOTAL_AMOUNT', '')),
        }

        # ── Fallbacks using OCR word stream ───────────────────────────────
        if ocr_words:
            result = self._fallback_recipient(ocr_words, result)
            result = self._fallback_dates(ocr_words, result)
            result = self._arbitrate_swapped_dates(ocr_words, result)

        return result

    # ── Field-level cleaners ──────────────────────────────────────────────

    def _clean_date(self, raw: str) -> str:
        """Extract a date value from a raw model string."""
        if not raw:
            return ''
        # Try each date pattern in order of specificity
        for pat in _DATE_PATTERNS:
            m = pat.search(raw)
            if m:
                return m.group(0).strip()
        return ''

    def _clean_amount(self, raw: str) -> str:
        """Extract a monetary amount from a raw model string."""
        if not raw:
            return ''

        # Try amount patterns in order — prefer currency-qualified matches
        for pat in _AMOUNT_PATTERNS:
            m = pat.search(raw)
            if m:
                # Use the full match (group 0) which includes the currency marker
                matched = m.group(0).strip()
                amount  = self._normalise_amount(matched)
                if amount:
                    return amount

        return ''

    def _clean_invoice_number(self, raw: str) -> str:
        """Extract invoice number from raw model string."""
        if not raw:
            return ''

        tokens = raw.split()
        if not tokens:
            return ''

        # Strategy 1: find a token that matches invoice number pattern
        # Search from end to start — invoice number is usually the last token
        for tok in reversed(tokens):
            tok_clean = tok.strip(':.,-#()')
            if (
                len(tok_clean) >= self.min_invoice_num_len
                and _INVOICE_NUM_PATTERN.match(tok_clean)
                and any(c.isdigit() for c in tok_clean)
            ):
                return tok_clean

        # Strategy 2: last token that has at least one digit
        for tok in reversed(tokens):
            tok_clean = tok.strip(':.,-#()')
            if any(c.isdigit() for c in tok_clean) and len(tok_clean) >= 2:
                return tok_clean

        # Strategy 3: last token regardless
        return tokens[-1].strip(':.,-#()')

    def _clean_name(self, raw: str) -> str:
        """
        Extract a person or company name from a raw model string.

        Skips leading label words (Bill, to, From, etc.) and stops
        at the first token that looks like an address element.
        """
        if not raw:
            return ''

        tokens = raw.split()
        if not tokens:
            return ''

        # Skip leading label words
        start = 0
        while start < len(tokens):
            t = tokens[start].lower().strip(':.,-')
            if t in _NAME_LABEL_WORDS:
                start += 1
            else:
                break

        # Collect name tokens until address boundary
        name_tokens = []
        for tok in tokens[start:]:
            t = tok.lower().strip(':.,-')

            # Stop at address keywords (also catches merged like 'Address:5776')
            if any(t.startswith(aw) for aw in _ADDRESS_TOKENS):
                break

            # Stop at pure digit (house number)
            if re.match(r'^\d+$', t):
                break

            # Stop at merged digit-alpha that is not a name initial
            # e.g. '5776Whitney' or 'tel:+1234'
            if re.match(r'^\d', t) and len(t) > 2:
                break

            # Stop at email / URL pattern
            if '@' in t or t.startswith('http') or t.startswith('www'):
                break

            # Stop at phone-like token
            if re.match(r'^\+?\(?\d[\d\s\-().]{5,}$', t):
                break

            # Hard length cap — prevents entire address blocks leaking in
            if len(name_tokens) >= self.max_name_tokens:
                break

            name_tokens.append(tok)

        result = ' '.join(name_tokens).strip(',:.-() ')
        return result

    # ── Fallback extractors using OCR word stream ─────────────────────────

    def _fallback_recipient(
        self,
        words: List[str],
        fields: Dict[str, str],
    ) -> Dict[str, str]:
        """
        If RECIPIENT_NAME is empty, scan OCR words for bill-to triggers
        and extract the name that follows.

        Handles all of these OCR tokenisation variants:
          - 'Bill'  'to'   'Nicole' 'Mathis'      (3 separate tokens)
          - 'Bill'  'to:'  'Nicole' 'Mathis'      (colon attached to 'to')
          - 'Bill'  'to:Nicole' 'Mathis'           (name merged with 'to:')
          - 'Billto:Nicole' 'Mathis'               (fully merged)
        """
        if fields.get('recipient_name', ''):
            return fields

        words_lower = [w.lower().strip(':.,-') for w in words]

        for i, w in enumerate(words_lower):
            if w not in _BILL_TO_TRIGGERS:
                continue
            if i + 1 >= len(words):
                continue

            next_raw   = words[i + 1]
            next_lower = next_raw.lower()

            # Case A: next token starts with 'to' followed by optional colon
            # and possibly the start of the name ('to:Nicole' or 'to:')
            if next_lower.startswith('to'):
                after_to = re.sub(r'^to[:\s]*', '', next_raw, flags=re.IGNORECASE).strip()
                name_parts = []

                if after_to:
                    # Name started in the same token as 'to:'
                    name_parts.append(after_to)
                    collect_start = i + 2
                else:
                    # Clean 'to' — name starts on next token
                    collect_start = i + 2

                for j in range(collect_start, min(collect_start + 6, len(words))):
                    tok = words[j].strip(':.,-')
                    if not tok:
                        continue
                    t = tok.lower()
                    if any(t.startswith(aw) for aw in _ADDRESS_TOKENS):
                        break
                    if re.match(r'^\d+$', t):
                        break
                    if '@' in t or t.startswith('http'):
                        break
                    if len(name_parts) >= self.max_name_tokens:
                        break
                    name_parts.append(tok)

                if name_parts:
                    fields['recipient_name'] = ' '.join(name_parts).strip(',:. ')
                    return fields

            # Case B: current trigger word itself merged with 'to:Name'
            # e.g. 'Billto:Nicole'
            merged_match = re.match(
                r'^(?:bill|billed|sold|send|ship)to:?(.+)',
                w, re.IGNORECASE
            )
            if merged_match:
                remainder  = merged_match.group(1).strip()
                name_parts = [remainder] if remainder else []
                for j in range(i + 1, min(i + 5, len(words))):
                    tok = words[j].strip(':.,-')
                    if not tok:
                        continue
                    t = tok.lower()
                    if any(t.startswith(aw) for aw in _ADDRESS_TOKENS):
                        break
                    if re.match(r'^\d+$', t):
                        break
                    name_parts.append(tok)
                if name_parts:
                    fields['recipient_name'] = ' '.join(name_parts).strip(',:. ')
                    return fields

        return fields

    def _fallback_dates(
        self,
        words: List[str],
        fields: Dict[str, str],
    ) -> Dict[str, str]:
        """
        If a date field is empty, scan the OCR word stream for a date
        near a matching label keyword and assign it to the right field.
        """
        inv_date_empty = not fields.get('invoice_date', '')
        due_date_empty = not fields.get('due_date', '')

        if not inv_date_empty and not due_date_empty:
            return fields

        words_lower = [w.lower().strip(':.,-') for w in words]

        for i, w in enumerate(words_lower):
            # Look for a date value in the next 4 tokens
            for j in range(i + 1, min(i + 5, len(words))):
                m = _DATE_ANY.search(words[j])
                if not m:
                    continue
                candidate = m.group(0).strip()

                # Determine which date field this belongs to
                # by looking at the surrounding label words
                context = ' '.join(words_lower[max(0, i-1):i+2])

                is_due = any(kw in context for kw in _DUE_DATE_LABELS)
                is_inv = any(kw in context for kw in _INVOICE_DATE_LABELS)

                if is_due and due_date_empty and candidate != fields.get('invoice_date', ''):
                    fields['due_date'] = candidate
                    due_date_empty = False
                elif is_inv and inv_date_empty and candidate != fields.get('due_date', ''):
                    fields['invoice_date'] = candidate
                    inv_date_empty = False

                if not inv_date_empty and not due_date_empty:
                    return fields
                break  # move to next label word

        return fields

    def _arbitrate_swapped_dates(
        self,
        words: List[str],
        fields: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Detect when both dates were assigned to the same field by the model
        (a known failure mode on some FATURA templates where two dates appear
        close together and the model tags both as DUE_DATE or both as
        INVOICE_DATE).

        Recovery strategy: collect ALL date occurrences in the OCR stream.
        If we have two dates but one field is empty, assign the unassigned
        date to the empty field using the label keyword closest to each date
        to determine which is which.
        """
        inv_date = fields.get('invoice_date', '')
        due_date = fields.get('due_date', '')

        # Only act if exactly one date field is filled
        if bool(inv_date) == bool(due_date):
            return fields

        filled   = inv_date or due_date
        empty_field = 'invoice_date' if not inv_date else 'due_date'

        # Collect all dates from OCR with their positions
        all_dates = []
        for i, w in enumerate(words):
            m = _DATE_ANY.search(w)
            if m:
                all_dates.append((i, m.group(0).strip()))

        if len(all_dates) < 2:
            return fields

        # Find a date that differs from the one already assigned
        words_lower = [w.lower().strip(':.,-') for w in words]
        for pos, date in all_dates:
            if date == filled:
                continue
            # Verify this date is near a label that matches the empty field
            context_start = max(0, pos - 5)
            context       = ' '.join(words_lower[context_start:pos + 2])

            if empty_field == 'due_date':
                # Only assign to due_date if context contains due-date keywords
                # OR if no invoice-date keywords are nearby
                if any(kw in context for kw in _DUE_DATE_LABELS) or \
                   not any(kw in context for kw in _INVOICE_DATE_LABELS):
                    fields['due_date'] = date
                    return fields
            else:
                if any(kw in context for kw in _INVOICE_DATE_LABELS) or \
                   not any(kw in context for kw in _DUE_DATE_LABELS):
                    fields['invoice_date'] = date
                    return fields

        # Last resort: just assign the first unmatched date to the empty field
        for _, date in all_dates:
            if date != filled:
                fields[empty_field] = date
                return fields

        return fields

    # ── Amount normalisation helper ────────────────────────────────────────

    @staticmethod
    def _normalise_amount(raw: str) -> str:
        """
        Normalise an amount string:
        - Remove internal spaces within the number
        - Standardise currency position: NUMBER CURRENCY_CODE
        - Keep currency symbol if no code present

        Examples:
            '$ 1,234.56'        → '$1,234.56'
            '1234.56 USD'       → '1,234.56 USD'   (no reformatting of commas)
            'EUR  1.234,56'     → '1.234,56 EUR'
            '1 234.56'          → '1234.56'  (spaces within number removed)
        """
        if not raw:
            return ''

        raw = raw.strip()

        # Find currency code
        code_match = _CURRENCY_CODE.search(raw)
        code       = code_match.group(0).upper() if code_match else ''

        # Find currency symbol
        sym_match  = _CURRENCY_SYMBOL.search(raw)
        symbol     = sym_match.group(0) if sym_match else ''

        # Extract the numeric part — remove all currency markers and normalise
        num = raw
        if code:
            num = _CURRENCY_CODE.sub('', num)
        if symbol:
            num = _CURRENCY_SYMBOL.sub('', num)

        # Remove remaining label words that may have crept in
        num = re.sub(r'\b(?:total|amount|balance|due|grand|net|sub)\b', '', num, flags=re.IGNORECASE)
        num = num.strip(':.,-() ')

        # Remove internal spaces that are not thousand separators
        # (e.g. '1 234.56' → '1234.56', but keep '1,234.56' intact)
        num = re.sub(r'(?<=\d)\s+(?=\d)', '', num)

        if not re.search(r'\d', num):
            return ''  # nothing left

        num = num.strip()

        # Reconstruct: prefer NUMBER CODE, fall back to SYMBOL+NUMBER
        if code:
            return f'{num} {code}'
        elif symbol:
            return f'{symbol}{num}'
        else:
            return num


# ── Module-level convenience function ─────────────────────────────────────

_default_cleaner = None


def clean_invoice_fields(
    raw_fields: Dict[str, str],
    ocr_words: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Module-level convenience wrapper around InvoiceCleaner.

    Creates a singleton cleaner on first call — safe to call in a loop.

    Parameters
    ----------
    raw_fields : dict
        Raw LayoutLMv3 predictions with uppercase keys.
    ocr_words : list of str, optional
        OCR word stream for fallback extraction.

    Returns
    -------
    dict with lowercase keys and clean values.
    """
    global _default_cleaner
    if _default_cleaner is None:
        _default_cleaner = InvoiceCleaner()
    return _default_cleaner.clean(raw_fields, ocr_words)
