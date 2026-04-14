from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

GROUP_A = {"invoice", "form", "budget"}
GROUP_B = {"email", "resume"}
CLASS_ORDER = ["invoice", "form", "resume", "email", "budget"]

def make_group_labels(labels: pd.Series) -> pd.Series:
    return labels.map(lambda x: "group_a" if x in GROUP_A else "group_b")

def load_split(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def build_stage1_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_label"] = make_group_labels(out["class_name"])
    return out

def build_error_propagation_table(results_df: pd.DataFrame) -> pd.DataFrame:
    wrong_router = results_df["router_correct"].eq(False).sum()
    wrong_specialist = (
        results_df["router_correct"].eq(True) &
        results_df["pred_label"].ne(results_df["true_label"])
    ).sum()
    correct_final = results_df["pred_label"].eq(results_df["true_label"]).sum()

    return pd.DataFrame({
        "category": ["correct_final", "wrong_router", "wrong_specialist"],
        "count": [correct_final, wrong_router, wrong_specialist]
    })

def format_prediction_output(
    doc_ids: pd.Series,
    true_labels: pd.Series,
    pred_labels: np.ndarray,
    split_name: str,
    model_name: str,
    proba_df: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "doc_id": doc_ids.values,
        "true_label": true_labels.values,
        "pred_label": pred_labels,
        "split": split_name,
        "model_name": model_name,
    })

    for cls in CLASS_ORDER:
        col = f"confidence_{cls}"
        out[col] = proba_df[cls].values if cls in proba_df.columns else 0.0

    return out
