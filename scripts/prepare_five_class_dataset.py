#!/usr/bin/env python3

"""
Prepare 5-class document dataset subset with train/val/test splits

This script:
1. Downloads Kaggle dataset
2. Inspects dataset structure 
3. Filters target classes: invoice, form, resume, email, budget
4. Builds metadata CSVs 
5. Creates reproducible train/val/test split
6. Creates organized subset folders 
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
TARGET_CLASSES = ["invoice", "form", "resume", "email", "budget"]
CLASS_ALIASES = {
    "invoice": ["invoice", "invoices"],
    "form": ["form", "forms"],
    "resume": ["resume", "resumes", "cv", "curriculum_vitae"],
    "email": ["email", "emails", "e-mail", "mail"],
    "budget": ["budget", "budgets"],
}
SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation", "dev"],
    "test": ["test", "testing"],
}


@dataclass
class Paths:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    five_subset_dir: Path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare five-class document dataset")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path, default=None, help="Existing dataset path. If omitted, uses data/raw")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle into data/raw")
    parser.add_argument("--dataset-slug", type=str, default="pdavpoojan/the-rvlcdip-dataset-test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--compute-hash", action="store_true")
    parser.add_argument("--max-files", type=int, default=None, help="Debug limit")
    return parser.parse_args()


def ensure_dirs(project_root: Path) -> Paths:
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    five_subset_dir = processed_dir / "five_class_subset"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    five_subset_dir.mkdir(parents=True, exist_ok=True)
    return Paths(project_root=project_root, raw_dir=raw_dir, processed_dir=processed_dir, five_subset_dir=five_subset_dir)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    logging.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def download_kaggle_dataset(dataset_slug: str, raw_dir: Path) -> None:
    zip_path = raw_dir / f"{dataset_slug.replace('/', '__')}.zip"
    cmd = [sys.executable, "-m", "kaggle.cli", "datasets", "download", "-d", dataset_slug, "-p", str(raw_dir)]
    run_cmd(cmd)

    # Try to find downloaded zip if name is different
    if not zip_path.exists():
        zips = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zips:
            raise FileNotFoundError("Kaggle download finished but no zip file found in data/raw")
        zip_path = zips[0]

    extract_dir = raw_dir / "kaggle_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["unzip", "-o", str(zip_path), "-d", str(extract_dir)])
    logging.info("Dataset extracted to: %s", extract_dir)


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def infer_split_from_parts(parts: List[str]) -> Optional[str]:
    norm = [normalize_token(p) for p in parts]
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in norm:
                return canonical
    return None


def infer_class_from_parts(parts: List[str]) -> Optional[str]:
    norm = [normalize_token(p) for p in parts]
    for canonical, aliases in CLASS_ALIASES.items():
        for alias in aliases:
            if normalize_token(alias) in norm:
                return canonical
    return None


def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def read_image_size(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        with Image.open(path) as img:
            w, h = img.size
        return w, h, None
    except (UnidentifiedImageError, OSError) as err:
        return None, None, str(err)


def discover_image_files(source_root: Path, max_files: Optional[int]) -> List[Path]:
    files: List[Path] = []
    for p in source_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
            if max_files is not None and len(files) >= max_files:
                break
    return files


def build_metadata(source_root: Path, files: List[Path], compute_hash: bool) -> pd.DataFrame:
    records = []
    errors = 0

    for idx, fp in enumerate(files):
        rel = fp.relative_to(source_root)
        parts = list(rel.parts)
        class_name = infer_class_from_parts(parts)
        split_original = infer_split_from_parts(parts)
        width, height, img_err = read_image_size(fp)

        if img_err is not None:
            errors += 1

        rec = {
            "doc_id": f"doc_{idx:08d}",
            "file_path": str(fp.resolve()),
            "relative_path": str(rel),
            "class_name": class_name,
            "split_original": split_original,
            "source_folder": str(fp.parent.relative_to(source_root)),
            "file_ext": fp.suffix.lower(),
            "width": width,
            "height": height,
            "read_error": img_err,
        }
        if compute_hash:
            rec["hash_md5"] = compute_md5(fp)
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    logging.info("Metadata created for %d files (image read errors: %d)", len(df), errors)
    return df


def create_splits(df: pd.DataFrame, seed: int, train_size: float, val_size: float, test_size: float) -> pd.DataFrame:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if df["split_original"].notna().any():
        known = df["split_original"].dropna().unique().tolist()
        known_set = set(known)
        if known_set.issubset({"train", "val", "test"}):
            out = df.copy()
            # Perfect case: official train/val/test all available.
            if known_set == {"train", "val", "test"}:
                logging.info("Using detected official split from folder names: %s", sorted(known))
                out["split"] = out["split_original"]
                return out

            # Common case: only train+test exist -> carve validation from train.
            if known_set == {"train", "test"}:
                logging.info(
                    "Detected train+test official folders; deriving validation split from train (seed=%d)",
                    seed,
                )
                train_idx = out[out["split_original"] == "train"].index
                tr_idx, va_idx = train_test_split(
                    train_idx,
                    test_size=val_size / (train_size + val_size),
                    random_state=seed,
                    stratify=out.loc[train_idx, "class_name"],
                )
                out.loc[tr_idx, "split"] = "train"
                out.loc[va_idx, "split"] = "val"
                out.loc[out["split_original"] == "test", "split"] = "test"
                return out

            # Any other partial split layout is not enough for direct usage.
            logging.warning(
                "Detected partial split folders %s. Falling back to new stratified train/val/test split.",
                sorted(known_set),
            )

    logging.info("No official split detected; creating stratified train/val/test splits")
    out = df.copy()
    train_val_idx, test_idx = train_test_split(
        out.index,
        test_size=test_size,
        random_state=seed,
        stratify=out["class_name"],
    )

    adjusted_val_ratio = val_size / (train_size + val_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=adjusted_val_ratio,
        random_state=seed,
        stratify=out.loc[train_val_idx, "class_name"],
    )

    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def write_split_csvs(df: pd.DataFrame, processed_dir: Path) -> None:
    df.to_csv(processed_dir / "metadata_five_classes.csv", index=False)
    df[df["split"] == "train"].to_csv(processed_dir / "train.csv", index=False)
    df[df["split"] == "val"].to_csv(processed_dir / "val.csv", index=False)
    df[df["split"] == "test"].to_csv(processed_dir / "test.csv", index=False)


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            logging.warning("Symlink failed for %s -> %s; falling back to copy", src, dst)
    shutil.copy2(src, dst)


def materialize_subset(df: pd.DataFrame, subset_root: Path, mode: str) -> None:
    for _, row in df.iterrows():
        src = Path(row["file_path"])
        split = row["split"]
        cls = row["class_name"]
        ext = src.suffix.lower()
        dst = subset_root / split / cls / f"{row['doc_id']}{ext}"
        safe_link_or_copy(src, dst, mode)


def save_reports(df_all: pd.DataFrame, df_five: pd.DataFrame, processed_dir: Path) -> None:
    # class balance by split
    pivot = (
        df_five.groupby(["split", "class_name"], as_index=False)
        .size()
        .pivot(index="split", columns="class_name", values="size")
        .fillna(0)
        .astype(int)
    )
    pivot.to_csv(processed_dir / "class_balance_by_split.csv")

    # duplicate report
    dup_report = {}
    if "hash_md5" in df_five.columns:
        dup = df_five[df_five.duplicated("hash_md5", keep=False)].sort_values("hash_md5")
        dup.to_csv(processed_dir / "duplicates_by_hash.csv", index=False)
        dup_report["hash_duplicates"] = int(dup.shape[0])
    else:
        dup_report["hash_duplicates"] = None

    name_dup = df_five[df_five.duplicated("relative_path", keep=False)]
    name_dup.to_csv(processed_dir / "duplicates_by_relative_path.csv", index=False)
    dup_report["relative_path_duplicates"] = int(name_dup.shape[0])

    summary = {
        "n_all_images": int(df_all.shape[0]),
        "n_selected_images": int(df_five.shape[0]),
        "selected_classes": TARGET_CLASSES,
        "split_counts": df_five["split"].value_counts().to_dict(),
        "class_counts": df_five["class_name"].value_counts().to_dict(),
        "duplicate_report": dup_report,
    }
    with (processed_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def infer_default_source(raw_dir: Path) -> Path:
    candidates = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not candidates:
        return raw_dir
    # Prefer kaggle_extracted if present
    for c in candidates:
        if c.name == "kaggle_extracted":
            return c
    # Otherwise largest directory by file count
    best = max(candidates, key=lambda d: sum(1 for _ in d.rglob("*") if _.is_file()))
    return best


def main() -> None:
    setup_logging()
    args = parse_args()

    paths = ensure_dirs(args.project_root)

    if args.download:
        download_kaggle_dataset(args.dataset_slug, paths.raw_dir)

    source_root = args.source_root if args.source_root else infer_default_source(paths.raw_dir)
    source_root = source_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    logging.info("Source root for discovery: %s", source_root)
    files = discover_image_files(source_root, args.max_files)
    if not files:
        raise RuntimeError(
            "No image files found. Place dataset under data/raw or pass --source-root."
        )

    df_all = build_metadata(source_root=source_root, files=files, compute_hash=args.compute_hash)
    df_all.to_csv(paths.processed_dir / "metadata_all.csv", index=False)

    df_five = df_all[df_all["class_name"].isin(TARGET_CLASSES)].copy()
    if df_five.empty:
        found = sorted(df_all["class_name"].dropna().unique().tolist())
        raise RuntimeError(
            "Could not map any files to the 5 target classes from folder names. "
            f"Detected mapped classes: {found}. "
            "Please pass --source-root to correct dataset folder or adjust CLASS_ALIASES."
        )

    missing = [c for c in TARGET_CLASSES if c not in set(df_five["class_name"].unique())]
    if missing:
        raise RuntimeError(
            f"Missing required classes in discovered data: {missing}. "
            "Stopping to avoid incomplete subset."
        )

    df_five = create_splits(
        df_five,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    write_split_csvs(df_five, paths.processed_dir)
    materialize_subset(df_five, paths.five_subset_dir, args.copy_mode)
    save_reports(df_all=df_all, df_five=df_five, processed_dir=paths.processed_dir)

    logging.info("Saved metadata and splits under %s", paths.processed_dir)
    logging.info("Materialized subset under %s", paths.five_subset_dir)


if __name__ == "__main__":
    main()
