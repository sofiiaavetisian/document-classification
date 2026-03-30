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
]
