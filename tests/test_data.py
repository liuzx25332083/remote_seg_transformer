from __future__ import annotations

import numpy as np

from seg_transformer.data import rgb_mask_to_labels
from seg_transformer.metrics import metrics_from_confusion


def test_palette_round_trip() -> None:
    palette = [
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 255],
        [0, 0, 0],
    ]
    mask = np.asarray([palette, palette[::-1]], dtype=np.uint8)
    labels = rgb_mask_to_labels(mask, palette)
    assert labels.tolist() == [[0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]]


def test_macro_f1_counts_unpredicted_class_as_zero() -> None:
    import torch

    confusion = torch.tensor(
        [
            [10, 0, 0, 0],
            [0, 10, 0, 0],
            [5, 5, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    metrics = metrics_from_confusion(
        confusion,
        ["A", "B", "Never", "Unknown"],
        ignore_index=3,
    )
    assert metrics["per_class"][2]["f1"] == 0.0
    assert 0.0 < metrics["macro_f1"] < 1.0
