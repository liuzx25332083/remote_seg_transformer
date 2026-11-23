from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def class_weights_from_counts(
    counts: list[int],
    ignore_index: int,
    device: torch.device,
) -> torch.Tensor:
    counts_tensor = torch.tensor(counts, dtype=torch.float64)
    counts_tensor[ignore_index] = 0
    positive = counts_tensor[counts_tensor > 0]
    median = positive.median()
    weights = torch.where(
        counts_tensor > 0,
        torch.sqrt(median / counts_tensor.clamp_min(1)),
        torch.zeros_like(counts_tensor),
    )
    active = weights > 0
    weights[active] = weights[active] / weights[active].mean()
    weights[ignore_index] = 0
    return weights.float().to(device)


class SoftDiceLoss(nn.Module):
    def __init__(self, ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = self.ignore_index
        one_hot = F.one_hot(
            safe_target,
            num_classes=logits.shape[1],
        ).permute(0, 3, 1, 2).float()
        mask = valid.unsqueeze(1)
        probabilities = probabilities * mask
        one_hot = one_hot * mask
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        active = one_hot.sum(dim=(0, 2, 3)) > 0
        return 1.0 - dice[active].mean()


def explicit_l2_penalty(model: nn.Module) -> torch.Tensor:
    penalty = None
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.ndim > 1:
            value = parameter.square().sum()
            penalty = value if penalty is None else penalty + value
    if penalty is None:
        raise RuntimeError("Model has no trainable matrix parameters")
    return penalty
