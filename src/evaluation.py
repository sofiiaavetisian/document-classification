"""Reusable evaluation helpers for document classification."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    invoice_label: str = "invoice",
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", labels=list(labels))),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", labels=list(labels))),
    }

    p, r, _, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=list(labels),
        average=None,
        zero_division=0,
    )

    for i, label in enumerate(labels):
        metrics[f"precision_{label}"] = float(p[i])
        metrics[f"recall_{label}"] = float(r[i])

    if invoice_label in labels:
        idx = list(labels).index(invoice_label)
        metrics["invoice_precision"] = float(p[idx])
        metrics["invoice_recall"] = float(r[idx])

    return metrics


def classification_report_df(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_matrix_df(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(cm, index=list(labels), columns=list(labels))


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (7, 6),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_xticks(np.arange(cm_df.shape[1]))
    ax.set_yticks(np.arange(cm_df.shape[0]))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, int(cm_df.values[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")


def metrics_dict_to_frame(metrics: Dict[str, float], model_name: str, split: str) -> pd.DataFrame:
    rec = {"model_name": model_name, "split": split, **metrics}
    return pd.DataFrame([rec])
