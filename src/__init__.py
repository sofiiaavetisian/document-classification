"""Shared package exports."""

from .config import OCRConfig, ProjectConfig, load_ocr_config
from .ocr_engine import (
    check_tesseract_installation,
    load_ocr_blocks,
    load_ocr_lines,
    load_ocr_result,
    load_ocr_text,
    load_ocr_words,
    ocr_batch,
    ocr_document,
)

__all__ = [
    "OCRConfig",
    "ProjectConfig",
    "load_ocr_config",
    "check_tesseract_installation",
    "ocr_document",
    "ocr_batch",
    "load_ocr_result",
    "load_ocr_text",
    "load_ocr_words",
    "load_ocr_lines",
    "load_ocr_blocks",
    "clean_ocr_text",
    "validate_date",
    "validate_amount",
    "validate_invoice_number",
    "validate_name",
    "compute_field_confidence",
    "find_anchor_line",
    "find_dates_in_text",
    "find_amounts_in_text",
    "extract_value_after_anchor",
    "detect_zones",
    "PageZones",
    "zone_summary",
    "extract_invoice_fields",
    "extract_batch",
]


from .validators import (
    clean_ocr_text,
    validate_date,
    validate_amount,
    validate_invoice_number,
    validate_name,
    compute_field_confidence,
)
from .invoice_rules import (
    find_anchor_line,
    find_dates_in_text,
    find_amounts_in_text,
    extract_value_after_anchor,
)
from .zones import detect_zones, PageZones, zone_summary
from .invoice_extraction import extract_invoice_fields, extract_batch