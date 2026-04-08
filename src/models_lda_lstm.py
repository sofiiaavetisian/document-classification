"""Utilities for experimental LDA + LSTM document classification."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def normalize_ocr_text(text: str) -> str:
    """Lowercase + whitespace normalization."""
    if text is None:
        return ""
    return " ".join(str(text).lower().split())


def basic_tokenize(text: str) -> List[str]:
    """Simple alnum tokenization for OCR text."""
    return re.findall(r"[a-z0-9]+", normalize_ocr_text(text))


def build_lda_corpus_text(texts: Sequence[str]) -> List[str]:
    """Prepare plain-text corpus for CountVectorizer+LDA."""
    return [" ".join(basic_tokenize(t)) for t in texts]


def fit_lda_vectorizer(
    train_texts: Sequence[str],
    min_df: int = 3,
    max_df: float = 0.95,
    max_features: int = 30000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[CountVectorizer, np.ndarray]:
    """Fit CountVectorizer on train only, return sparse train term matrix."""
    vect = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=False,
        stop_words="english",
    )
    X_train = vect.fit_transform(build_lda_corpus_text(train_texts))
    return vect, X_train


def transform_lda_vectorizer(vectorizer: CountVectorizer, texts: Sequence[str]):
    """Transform texts with fitted vectorizer."""
    return vectorizer.transform(build_lda_corpus_text(texts))


def fit_lda_model(
    X_train_counts,
    n_topics: int,
    random_state: int = 42,
    max_iter: int = 20,
) -> LatentDirichletAllocation:
    """Fit LDA topic model on train count matrix only."""
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=max_iter,
    )
    lda.fit(X_train_counts)
    return lda


def transform_topics(lda_model: LatentDirichletAllocation, X_counts) -> np.ndarray:
    """Return topic distributions (rows sum to 1)."""
    return lda_model.transform(X_counts)


def label_to_index(labels: Sequence[str], y: Sequence[str]) -> np.ndarray:
    mapping = {lab: i for i, lab in enumerate(labels)}
    return np.array([mapping[str(v)] for v in y], dtype=np.int64)


def index_to_label(labels: Sequence[str], y_idx: Sequence[int]) -> List[str]:
    return [labels[int(i)] for i in y_idx]


def build_prediction_table(
    doc_ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    split: str,
    model_name: str,
    labels: Sequence[str],
    proba: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Project-style prediction table with optional confidence columns."""
    df = pd.DataFrame(
        {
            "doc_id": list(doc_ids),
            "true_label": list(y_true),
            "pred_label": list(y_pred),
            "split": split,
            "model_name": model_name,
        }
    )
    if proba is not None and proba.shape[1] == len(labels):
        for i, label in enumerate(labels):
            df[f"confidence_{label}"] = proba[:, i]
    return df


def save_json(obj: Dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_artifact(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)

