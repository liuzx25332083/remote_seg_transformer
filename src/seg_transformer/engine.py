from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import resolve_path, save_json
from .losses import SoftDiceLoss, class_weights_from_counts, explicit_l2_penalty
from .metrics import metrics_from_confusion, update_confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda_if_available" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SegmentationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        mean: list[float],
        std: list[float],
        augment: bool,
    ):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        mask = Image.open(self.root / row["mask_path"])
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8)).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.uint8)).long()
        image_tensor = image_tensor.float().div_(255.0)
        if self.augment:
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[1])
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])
                mask_tensor = torch.flip(mask_tensor, dims=[0])
            rotations = random.randint(0, 3)
            if rotations:
                image_tensor = torch.rot90(image_tensor, rotations, dims=(1, 2))
                mask_tensor = torch.rot90(mask_tensor, rotations, dims=(0, 1))
        image_tensor = (image_tensor - self.mean) / self.std
        return image_tensor.contiguous(), mask_tensor.contiguous()


def make_loaders(
    config: dict[str, Any],
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    data_root = resolve_path(config["data_root"])
    manifest = pd.read_csv(data_root / "manifests" / "patch_manifest.csv")
    stats = json.loads(
        (data_root / "manifests" / "patch_stats.json").read_text(encoding="utf-8")
    )
    patch_root = data_root / "patches"
    loaders = {}
    counts: dict[str, int] = {}
    for split, augment in (("train", True), ("val", False), ("test", False)):
        frame = manifest.loc[manifest["split"] == split]
        dataset = SegmentationDataset(
            frame,
            patch_root,
            stats["normalization"]["mean"],
            stats["normalization"]["std"],
            augment=augment,
        )
        generator = torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[split])
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            num_workers=int(config["num_workers"]),
            generator=generator,
        )
        counts[split] = len(dataset)
    return loaders["train"], loaders["val"], loaders["test"], counts


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dice_loss: nn.Module,
    device: torch.device,
    l2_lambda: float,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=scaler.is_enabled(),
        ):
            logits = model(images)
            loss = criterion(logits, masks) + dice_loss(logits, masks)
            loss = loss + 0.5 * l2_lambda * explicit_l2_penalty(model)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * len(images)
        total_examples += len(images)
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    ignore_index: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    model.eval()
    confusion = torch.zeros(
        len(class_names),
        len(class_names),
        dtype=torch.int64,
    )
    total_loss = 0.0
    total_examples = 0
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, masks)
        predictions = logits.argmax(dim=1)
        update_confusion_matrix(
            confusion,
            predictions,
            masks,
            len(class_names),
            ignore_index,
        )
        total_loss += float(loss.item()) * len(images)
        total_examples += len(images)
    metrics = metrics_from_confusion(confusion, class_names, ignore_index)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics, confusion


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    class_weights: torch.Tensor,
    run_dir: Path,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=int(config["ignore_index"]),
    )
    dice_loss = SoftDiceLoss(int(config["ignore_index"]))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(config.get("amp", True)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"])),
        eta_min=float(config["learning_rate"]) * 0.05,
    )
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        lr = float(optimizer.param_groups[0]["lr"])
        epoch_started = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            dice_loss,
            device,
            float(config["weight_decay"]),
            scaler,
        )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            config["class_names"],
            int(config["ignore_index"]),
        )
        scheduler.step()
        score = float(val_metrics["miou"])
        improved = score > best_score + 1e-8
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_miou": best_score,
                    "config": config,
                },
                run_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_miou": val_metrics["miou"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "is_best": improved,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"epoch={epoch:02d} loss={train_loss:.4f} "
            f"val_mIoU={val_metrics['miou']:.4f} "
            f"val_F1={val_metrics['macro_f1']:.4f} best={improved}"
        )
        if stale_epochs >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    summary = {
        "best_epoch": best_epoch,
        "best_val_miou": best_score,
        "epochs_completed": len(history),
        "train_wall_seconds": time.perf_counter() - started,
        "total_epoch_seconds": sum(row["epoch_seconds"] for row in history),
    }
    save_json(summary, run_dir / "training_summary.json")
    return model, pd.DataFrame(history), summary
