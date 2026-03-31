"""Modeling helpers for OCR text + layout architecture."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class TextVectorizerConfig:
    word_ngram_range: Tuple[int, int] = (1, 2)
    char_ngram_range: Tuple[int, int] = (3, 5)
    min_df_word: int = 2
    min_df_char: int = 2
    max_features_word: int = 60000
    max_features_char: int = 120000


def clean_ocr_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text).lower()
    # Preserve currency/date symbols but normalize whitespace.
    t = " ".join(t.split())
    return t


def fit_text_vectorizers(
    train_texts: Sequence[str],
    cfg: Optional[TextVectorizerConfig] = None,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    cfg = cfg or TextVectorizerConfig()

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=cfg.word_ngram_range,
        min_df=cfg.min_df_word,
        max_features=cfg.max_features_word,
        lowercase=False,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.char_ngram_range,
        min_df=cfg.min_df_char,
        max_features=cfg.max_features_char,
        lowercase=False,
        sublinear_tf=True,
    )

    train_clean = [clean_ocr_text(t) for t in train_texts]
    word_vectorizer.fit(train_clean)
    char_vectorizer.fit(train_clean)
    return word_vectorizer, char_vectorizer


def transform_text_features(
    texts: Sequence[str],
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
) -> sparse.csr_matrix:
    clean = [clean_ocr_text(t) for t in texts]
    xw = word_vectorizer.transform(clean)
    xc = char_vectorizer.transform(clean)
    return sparse.hstack([xw, xc], format="csr")


def fit_layout_preprocessor(train_layout_df: pd.DataFrame) -> Tuple[SimpleImputer, StandardScaler, List[str]]:
    feature_cols = [c for c in train_layout_df.columns if c != "doc_id"]
    X = train_layout_df[feature_cols]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_imp)
    return imputer, scaler, feature_cols


def transform_layout_features(
    layout_df: pd.DataFrame,
    feature_cols: List[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    X = layout_df[feature_cols]
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled


def get_layout_model(random_state: int = 42):
    """Choose layout-only model with graceful fallback.

    Preference: XGBoost (if installed) -> ExtraTrees -> RandomForest.
    """
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        try:
            return ExtraTreesClassifier(
                n_estimators=450,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
        except Exception:
            return RandomForestClassifier(
                n_estimators=450,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )


def fit_text_only_model(X_text_train: sparse.csr_matrix, y_train: Sequence[str], random_state: int = 42):
    model = LogisticRegression(
        max_iter=3000,
        solver="saga",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_text_train, y_train)
    return model


def fit_layout_only_model(X_layout_train: np.ndarray, y_train: Sequence[str], random_state: int = 42):
    model = get_layout_model(random_state=random_state)
    model.fit(X_layout_train, y_train)
    return model


def fit_text_layout_model(
    X_text_train: sparse.csr_matrix,
    X_layout_train_scaled: np.ndarray,
    y_train: Sequence[str],
    random_state: int = 42,
):
    X_layout_sparse = sparse.csr_matrix(X_layout_train_scaled)
    X_combo = sparse.hstack([X_text_train, X_layout_sparse], format="csr")

    model = LogisticRegression(
        max_iter=3500,
        solver="saga",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_combo, y_train)
    return model


def combine_text_layout_features(
    X_text: sparse.csr_matrix,
    X_layout_scaled: np.ndarray,
) -> sparse.csr_matrix:
    return sparse.hstack([X_text, sparse.csr_matrix(X_layout_scaled)], format="csr")


def _decision_to_proba(decision: np.ndarray) -> np.ndarray:
    if decision.ndim == 1:
        # binary fallback
        decision = np.vstack([-decision, decision]).T
    exp = np.exp(decision - decision.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def predict_labels_and_proba(model, X):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        proba = _decision_to_proba(model.decision_function(X))
    else:
        proba = None

    return y_pred, proba


def build_prediction_table(
    doc_ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    split: str,
    model_name: str,
    labels: Sequence[str],
    proba: Optional[np.ndarray] = None,
) -> pd.DataFrame:
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


def save_artifact(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)


def save_model_bundle(
    model,
    model_path: str | Path,
    word_vectorizer: Optional[TfidfVectorizer] = None,
    char_vectorizer: Optional[TfidfVectorizer] = None,
    imputer: Optional[SimpleImputer] = None,
    scaler: Optional[StandardScaler] = None,
    layout_feature_cols: Optional[List[str]] = None,
) -> None:
    save_artifact(model, model_path)

    base = Path(model_path).with_suffix("")
    if word_vectorizer is not None:
        save_artifact(word_vectorizer, base.parent / f"{base.name}_word_vectorizer.joblib")
    if char_vectorizer is not None:
        save_artifact(char_vectorizer, base.parent / f"{base.name}_char_vectorizer.joblib")
    if imputer is not None:
        save_artifact(imputer, base.parent / f"{base.name}_layout_imputer.joblib")
    if scaler is not None:
        save_artifact(scaler, base.parent / f"{base.name}_layout_scaler.joblib")
    if layout_feature_cols is not None:
        save_artifact(layout_feature_cols, base.parent / f"{base.name}_layout_columns.joblib")
