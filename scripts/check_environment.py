#!/usr/bin/env python3
"""Environment validator for the document-classification project.

Checks:
- Python package availability (required + optional).
- Tesseract binary availability and version.
- Kaggle API credential file presence.

Exits with non-zero code if required dependencies are missing.
"""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "cv2",
    "PIL",
    "pytesseract",
    "yaml",
    "tqdm",
    "joblib",
    "dateutil",
    "regex",
    "kaggle",
]

OPTIONAL_PACKAGES = [
    "rapidfuzz",
    "wordfreq",
    "xgboost",
    "lightgbm",
    "imblearn",
]


def check_imports(packages: List[str]) -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            status[pkg] = True
        except Exception:
            status[pkg] = False
    return status


def print_import_report(title: str, status: Dict[str, bool]) -> List[str]:
    print(f"\n{title}")
    missing: List[str] = []
    for pkg, ok in status.items():
        marker = "OK" if ok else "MISSING"
        print(f"- {pkg}: {marker}")
        if not ok:
            missing.append(pkg)
    return missing


def check_tesseract() -> bool:
    print("\nTesseract OCR Check")
    tesseract_path = shutil.which("tesseract")
    if tesseract_path is None:
        print("- tesseract: MISSING")
        print("- WARNING: OCR pipeline cannot run until Tesseract is installed.")
        print_install_instructions()
        return False

    try:
        proc = subprocess.run(
            ["tesseract", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = proc.stdout.splitlines()[0] if proc.stdout else "unknown version"
        print(f"- tesseract: OK ({first_line})")
        print(f"- binary: {tesseract_path}")
        return True
    except Exception as err:
        print("- tesseract: ERROR while reading version")
        print(f"- details: {err}")
        print_install_instructions()
        return False


def print_install_instructions() -> None:
    os_name = platform.system().lower()
    print("\nInstall instructions:")
    if "darwin" in os_name:
        print("- macOS (Homebrew): brew install tesseract")
        print("- Verify: tesseract --version")
    elif "linux" in os_name:
        print("- Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y tesseract-ocr")
        print("- Verify: tesseract --version")
    elif "windows" in os_name:
        print("- Windows: install Tesseract from UB Mannheim build")
        print("- Add install directory to PATH")
        print("- Verify in a new terminal: tesseract --version")
    else:
        print("- Install Tesseract for your OS and ensure 'tesseract' is in PATH")


def check_kaggle_credentials() -> bool:
    print("\nKaggle Credentials Check")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"- kaggle.json: MISSING at {kaggle_json}")
        print("- WARNING: dataset download from Kaggle will fail until credentials are configured.")
        print("- Setup:")
        print("  mkdir -p ~/.kaggle")
        print("  cp /path/to/kaggle.json ~/.kaggle/kaggle.json")
        print("  chmod 600 ~/.kaggle/kaggle.json")
        return False

    try:
        payload = json.loads(kaggle_json.read_text(encoding="utf-8"))
        has_user = bool(payload.get("username"))
        has_key = bool(payload.get("key"))
        ok = has_user and has_key
        marker = "OK" if ok else "INVALID"
        print(f"- kaggle.json: {marker} ({kaggle_json})")
        if not ok:
            print("- WARNING: kaggle.json exists but missing 'username' or 'key'.")
        return ok
    except Exception as err:
        print(f"- kaggle.json: ERROR parsing file ({err})")
        return False


def main() -> int:
    print("Environment Validation Report")
    print(f"- Python: {sys.version.split()[0]}")

    required_status = check_imports(REQUIRED_PACKAGES)
    optional_status = check_imports(OPTIONAL_PACKAGES)

    missing_required = print_import_report("Required Python Packages", required_status)
    missing_optional = print_import_report("Optional Python Packages", optional_status)

    tesseract_ok = check_tesseract()
    kaggle_ok = check_kaggle_credentials()

    print("\nSummary")
    print(f"- missing_required_packages: {len(missing_required)}")
    print(f"- missing_optional_packages: {len(missing_optional)}")
    print(f"- tesseract_ok: {tesseract_ok}")
    print(f"- kaggle_credentials_ok: {kaggle_ok}")

    if missing_required:
        print("\nAction: install required packages with:")
        print("- pip install -r requirements.txt")
    if missing_optional:
        print("\nOptional install:")
        print("- pip install -r requirements-optional.txt")

    if not tesseract_ok:
        print("\nAction: install Tesseract before OCR/model notebooks.")

    if not kaggle_ok:
        print("\nAction: configure ~/.kaggle/kaggle.json before dataset download notebook.")

    if missing_required or not tesseract_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
