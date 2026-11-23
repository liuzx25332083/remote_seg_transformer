from __future__ import annotations

import numpy as np
import torch


def update_confusion_matrix(
    confusion: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    valid = target != ignore_index
    indices = (
        target[valid].to(torch.int64) * num_classes
        + prediction[valid].to(torch.int64)
    )
    counts = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    )
    confusion += counts.reshape(num_classes, num_classes).cpu()


def metrics_from_confusion(
    confusion: torch.Tensor,
    class_names: list[str],
    ignore_index: int,
) -> dict:
    matrix = confusion.numpy().astype(np.int64)
    evaluated = np.ones(len(class_names), dtype=bool)
    evaluated[ignore_index] = False
    diagonal = np.diag(matrix).astype(float)
    row_sum = matrix.sum(axis=1).astype(float)
    col_sum = matrix.sum(axis=0).astype(float)
    union = row_sum + col_sum - diagonal
    iou = np.divide(
        diagonal,
        union,
        out=np.full_like(diagonal, np.nan),
        where=union > 0,
    )
    precision = np.divide(diagonal, col_sum, out=np.zeros_like(diagonal), where=col_sum > 0)
    recall = np.divide(diagonal, row_sum, out=np.zeros_like(diagonal), where=row_sum > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) > 0,
    )
    valid_iou = iou[evaluated & ~np.isnan(iou)]
    correct = float(diagonal[evaluated].sum())
    total = float(matrix[evaluated].sum())
    per_class = []
    for index, name in enumerate(class_names):
        if index == ignore_index:
            continue
        per_class.append(
            {
                "class_name": name,
                "iou": float(iou[index]) if not np.isnan(iou[index]) else None,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(row_sum[index]),
            }
        )
    return {
        "pixel_accuracy": correct / total if total else 0.0,
        "miou": float(valid_iou.mean()) if len(valid_iou) else 0.0,
        "macro_f1": float(f1[evaluated].mean()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
