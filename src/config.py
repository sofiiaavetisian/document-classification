"""Configuration objects for OCR pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class OCRConfig:
    """Runtime configuration for OCR pipeline.

    Fields are intentionally explicit so notebooks and scripts can override
    behavior in a controlled, auditable way.
    """

    tesseract_cmd: Optional[str] = None
    lang: str = "eng"
    psm: int = 6
    oem: int = 3
    extra_config: str = ""

    preprocess_mode: str = "adaptive"  # none|gray|otsu|adaptive|denoise|adaptive_denoise
    enable_grayscale: bool = True
    enable_denoise: bool = True
    enable_deskew: bool = False
    resize_max_dim: int = 1800

    min_confidence: float = 0.0
    diagnostics_enabled: bool = True

    cache_dir: str = "data/interim/ocr"
    diagnostics_dir: str = "outputs/ocr_diagnostics"
    failure_log_path: str = "data/interim/ocr/logs/ocr_failures.jsonl"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectConfig:
    """Project-level config wrapper with OCR defaults."""

    project_name: str = "document_classification_project"
    random_seed: int = 42
    ocr: OCRConfig = field(default_factory=OCRConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        ocr_data = data.get("ocr", {}) or {}
        ocr = OCRConfig(**ocr_data)

        return ProjectConfig(
            project_name=data.get("project_name", "document_classification_project"),
            random_seed=int(data.get("random_seed", 42)),
            ocr=ocr,
        )


def load_ocr_config(config_path: str | Path = "configs/config.yaml") -> OCRConfig:
    """Load OCR config from YAML. Falls back to defaults if section missing."""
    return ProjectConfig.from_yaml(config_path).ocr
