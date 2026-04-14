"""Hybrid late-fusion helpers for classical document classification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class TextBranchConfig:
    """Configuration for text branch vectorization and classifier."""

    word_ngram_range: Tuple[int, int] = (1, 2)
    char_ngram_range: Tuple[int, int] = (3, 5)
    min_df_word: int = 2
    min_df_char: int = 2
    max_features_word: int = 60000
    max_features_char: int = 100000
    c_value: float = 1.0
    random_state: int = 42


def clean_ocr_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = " ".join(str(text).lower().split())
    return cleaned


def _softmax(decision: np.ndarray) -> np.ndarray:
    if decision.ndim == 1:
        decision = np.vstack([-decision, decision]).T
    shifted = decision - decision.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _aligned_probabilities(
    raw_prob: np.ndarray,
    raw_classes: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray:
    out = np.zeros((raw_prob.shape[0], len(labels)), dtype=np.float64)
    class_to_idx = {c: i for i, c in enumerate(raw_classes)}
    for j, label in enumerate(labels):
        if label in class_to_idx:
            out[:, j] = raw_prob[:, class_to_idx[label]]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return out / row_sums


def save_artifact(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)


def fit_text_vectorizers(
    train_texts: Sequence[str],
    cfg: Optional[TextBranchConfig] = None,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    cfg = cfg or TextBranchConfig()
    train_clean = [clean_ocr_text(t) for t in train_texts]

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

    word_vectorizer.fit(train_clean)
    char_vectorizer.fit(train_clean)
    return word_vectorizer, char_vectorizer


def transform_text_features(
    texts: Sequence[str],
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
) -> sparse.csr_matrix:
    clean = [clean_ocr_text(t) for t in texts]
    x_word = word_vectorizer.transform(clean)
    x_char = char_vectorizer.transform(clean)
    return sparse.hstack([x_word, x_char], format="csr")


def fit_calibrated_linear_svm(
    X_train,
    y_train: Sequence[str],
    c_value: float = 1.0,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    base = LinearSVC(C=c_value, random_state=random_state)
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(X_train, y_train)
    return model


def fit_logistic_text_classifier(
    X_train,
    y_train: Sequence[str],
    c_value: float = 1.0,
    random_state: int = 42,
) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value,
        max_iter=3000,
        solver="saga",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def fit_layout_preprocessor(
    train_layout_df: pd.DataFrame,
    id_col: str = "doc_id",
) -> Tuple[SimpleImputer, StandardScaler, List[str]]:
    feature_cols = [c for c in train_layout_df.columns if c != id_col]
    X = train_layout_df[feature_cols]
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_imp)
    return imputer, scaler, feature_cols


def transform_dense_features(
    feat_df: pd.DataFrame,
    feature_cols: Sequence[str],
    imputer: SimpleImputer,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    X = feat_df[list(feature_cols)]
    X_imp = imputer.transform(X)
    if scaler is not None:
        return scaler.transform(X_imp)
    return X_imp


def get_tree_classifier(random_state: int = 42):
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.06,
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
                n_estimators=400,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception:
            return RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )


def predict_proba_robust(model, X) -> Tuple[np.ndarray, Sequence[str]]:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        classes = getattr(model, "classes_", [])
        return prob, classes
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        prob = _softmax(np.asarray(decision))
        classes = getattr(model, "classes_", [])
        return prob, classes
    raise ValueError("Model must implement predict_proba or decision_function")


def weighted_average_fusion(
    probabilities: Dict[str, np.ndarray],
    labels: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    if not probabilities:
        raise ValueError("No probability arrays provided for fusion.")

    branch_names = sorted(probabilities.keys())
    n = next(iter(probabilities.values())).shape[0]
    fused = np.zeros((n, len(labels)), dtype=np.float64)

    if weights is None:
        weights = {name: 1.0 for name in branch_names}

    weight_sum = float(sum(max(weights.get(name, 0.0), 0.0) for name in branch_names))
    if weight_sum <= 0.0:
        raise ValueError("Fusion weights must sum to a positive value.")

    for name in branch_names:
        w = max(weights.get(name, 0.0), 0.0)
        fused += w * probabilities[name]

    fused /= weight_sum
    fused = fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)
    return fused


def optimize_fusion_weights(
    val_probabilities: Dict[str, np.ndarray],
    y_val: Sequence[str],
    labels: Sequence[str],
    step: float = 0.1,
) -> Tuple[Dict[str, float], float]:
    """Grid-search val-only weights for three branches."""
    from sklearn.metrics import f1_score

    names = sorted(val_probabilities.keys())
    if len(names) != 3:
        # Generic fallback: uniform weights for non-3-branch use.
        weights = {n: 1.0 for n in names}
        fused = weighted_average_fusion(val_probabilities, labels, weights)
        pred = np.asarray(labels)[np.argmax(fused, axis=1)]
        score = float(f1_score(y_val, pred, labels=list(labels), average="macro"))
        return weights, score

    best_score = -1.0
    best_weights: Dict[str, float] = {n: 1.0 for n in names}
    values = np.arange(0.0, 1.0 + 1e-9, step)
    for w0 in values:
        for w1 in values:
            w2 = 1.0 - w0 - w1
            if w2 < 0.0:
                continue
            weights = {names[0]: float(w0), names[1]: float(w1), names[2]: float(w2)}
            fused = weighted_average_fusion(val_probabilities, labels, weights)
            pred = np.asarray(labels)[np.argmax(fused, axis=1)]
            score = float(f1_score(y_val, pred, labels=list(labels), average="macro"))
            if score > best_score:
                best_score = score
                best_weights = weights
    return best_weights, best_score


def _stack_meta_features(probabilities: Dict[str, np.ndarray]) -> np.ndarray:
    names = sorted(probabilities.keys())
    chunks = [probabilities[n] for n in names]
    return np.concatenate(chunks, axis=1)


def fit_stacking_meta_classifier(
    val_probabilities: Dict[str, np.ndarray],
    y_val: Sequence[str],
    random_state: int = 42,
) -> Tuple[LogisticRegression, LabelEncoder]:
    X_meta = _stack_meta_features(val_probabilities)
    le = LabelEncoder()
    y_enc = le.fit_transform(list(y_val))
    clf = LogisticRegression(
        max_iter=3000,
        multi_class="auto",
        solver="lbfgs",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_meta, y_enc)
    return clf, le


def stacking_predict_proba(
    meta_model: LogisticRegression,
    label_encoder: LabelEncoder,
    probabilities: Dict[str, np.ndarray],
    labels: Sequence[str],
) -> np.ndarray:
    X_meta = _stack_meta_features(probabilities)
    raw = meta_model.predict_proba(X_meta)
    return _aligned_probabilities(raw, label_encoder.classes_, labels)


def prediction_table(
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


def top_k_errors(
    pred_df: pd.DataFrame,
    labels: Sequence[str],
    k: int = 20,
) -> pd.DataFrame:
    """Return misclassifications with confidence margin."""
    conf_cols = [f"confidence_{l}" for l in labels if f"confidence_{l}" in pred_df.columns]
    out = pred_df.copy()
    if len(conf_cols) >= 2:
        sorted_conf = np.sort(out[conf_cols].to_numpy(), axis=1)
        out["margin_top2"] = sorted_conf[:, -1] - sorted_conf[:, -2]
    else:
        out["margin_top2"] = 0.0
    out = out[out["true_label"] != out["pred_label"]]
    return out.sort_values("margin_top2", ascending=True).head(k).reset_index(drop=True)
