"""BERT encoder + RandomForest helpers for experimental document classification.

This module is intentionally experimental:
- BERT is used only as a frozen encoder to produce embeddings.
- A classical RandomForest classifier is trained on top of embeddings (+ optional aux features).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def check_transformers_stack() -> Tuple[bool, str]:
    """Return availability of transformers+torch stack with a diagnostic message."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True, "transformers/torch available"
    except Exception as exc:
        return False, f"Missing transformers stack: {exc}"


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mean_pool(last_hidden_state, attention_mask):
    """Masked mean pooling over token dimension."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_texts_with_bert(
    texts: Sequence[str],
    model_name: str = "bert-base-uncased",
    pooling: str = "mean",
    batch_size: int = 16,
    max_length: int = 256,
    normalize: bool = False,
) -> np.ndarray:
    """Encode texts with frozen BERT and return dense embeddings."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = [str(t) if t is not None else "" for t in texts[start : start + batch_size]]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            if pooling == "cls":
                emb = out.last_hidden_state[:, 0, :]
            else:
                emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            arr = emb.detach().cpu().numpy().astype(np.float32)
            outputs.append(arr)

    X = np.vstack(outputs) if outputs else np.zeros((0, 768), dtype=np.float32)
    if normalize and len(X) > 0:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
    return X


def save_embedding_cache(
    path: str | Path,
    doc_ids: Sequence[str],
    embeddings: np.ndarray,
    meta: Optional[Dict[str, object]] = None,
) -> None:
    """Save embedding cache as NPZ + JSON metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, doc_ids=np.asarray(doc_ids), embeddings=embeddings)
    if meta is not None:
        with p.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def load_embedding_cache(path: str | Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load embedding cache if present."""
    p = Path(path)
    if not p.exists():
        return None
    obj = np.load(p, allow_pickle=True)
    return obj["doc_ids"], obj["embeddings"]


def build_aux_feature_tables(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    layout_feature_dir: str | Path = "data/interim/layout_features",
    selected_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load existing layout summary features and align by doc_id."""
    root = Path(layout_feature_dir)
    tr_path = root / "train_layout_features.csv"
    va_path = root / "val_layout_features.csv"
    te_path = root / "test_layout_features.csv"

    if not (tr_path.exists() and va_path.exists() and te_path.exists()):
        raise FileNotFoundError(
            "Layout feature tables not found. Expected "
            f"{tr_path}, {va_path}, {te_path}"
        )

    tr = pd.read_csv(tr_path)
    va = pd.read_csv(va_path)
    te = pd.read_csv(te_path)

    if selected_cols is None:
        selected_cols = [
            "image_width",
            "image_height",
            "aspect_ratio",
            "num_words",
            "num_lines",
            "numeric_token_ratio",
            "currency_token_count",
            "date_like_token_count",
            "invoice_anchor_count",
            "budget_anchor_count",
            "resume_anchor_count",
            "email_anchor_count",
            "form_anchor_count",
        ]

    keep_cols = ["doc_id"] + [c for c in selected_cols if c in tr.columns]
    tr = train_df[["doc_id"]].merge(tr[keep_cols], on="doc_id", how="left")
    va = val_df[["doc_id"]].merge(va[keep_cols], on="doc_id", how="left")
    te = test_df[["doc_id"]].merge(te[keep_cols], on="doc_id", how="left")
    feat_cols = [c for c in tr.columns if c != "doc_id"]
    return tr, va, te, feat_cols


def fit_aux_preprocessor(
    train_aux: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[SimpleImputer, StandardScaler]:
    """Fit imputer+scaler on train aux features only."""
    X = train_aux[list(feature_cols)]
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_imp)
    return imp, scaler


def transform_aux_features(
    aux_df: pd.DataFrame,
    feature_cols: Sequence[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply fitted aux preprocessing."""
    X = aux_df[list(feature_cols)]
    X_imp = imputer.transform(X)
    return scaler.transform(X_imp)


def build_combined_matrix(embeddings: np.ndarray, aux_array: Optional[np.ndarray]) -> np.ndarray:
    """Concatenate embedding vectors with optional dense aux features."""
    if aux_array is None:
        return embeddings
    if len(aux_array) != len(embeddings):
        raise ValueError("embeddings and aux_array must have same row count")
    return np.hstack([embeddings, aux_array])


def train_random_forest(
    X_train: np.ndarray,
    y_train: Sequence[str],
    random_state: int = 42,
    n_estimators: int = 600,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    class_weight: str | Dict[str, float] | None = "balanced_subsample",
) -> RandomForestClassifier:
    """Train RF classifier on dense feature matrix."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_labels_and_proba(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict labels and probabilities."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
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
    """Project-standard prediction table with optional class confidence columns."""
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
    """Save model or preprocessor artifact via joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)
