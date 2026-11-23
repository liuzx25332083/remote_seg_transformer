from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import pandas as pd
import torch
import torchvision

from .config import load_config, resolve_path, save_json
from .data import download_source_subset, prepare_patches
from .engine import (
    choose_device,
    evaluate,
    fit,
    make_loaders,
    set_seed,
)
from .losses import class_weights_from_counts
from .models import build_model, parameter_count


def environment_metadata(device: torch.device) -> dict:
    return {
        "device": str(device),
        "device_name": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": torch.get_num_threads(),
    }


def command_download(config: dict) -> None:
    download_source_subset(config)


def command_prepare(config: dict) -> None:
    prepare_patches(config)


def command_train(config: dict, args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_root = resolve_path(config["output_root"])
    run_id = args.run_id or f"{args.model}_seed{args.seed}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    set_seed(args.seed)
    torch.set_num_threads(int(config["torch_num_threads"]))
    device = choose_device(config["device_preference"])
    data_root = resolve_path(config["data_root"])
    stats = json.loads(
        (data_root / "manifests" / "patch_stats.json").read_text(encoding="utf-8")
    )
    train_loader, val_loader, test_loader, counts = make_loaders(config, args.seed)
    class_weights = class_weights_from_counts(
        stats["train_class_pixel_counts"],
        int(config["ignore_index"]),
        device,
    )
    model = build_model(args.model, config).to(device)
    total_parameters, trainable_parameters = parameter_count(model)
    config_record = {
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "split_counts": counts,
        "data_manifest": str(data_root / "manifests" / "patch_manifest.csv"),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "environment": environment_metadata(device),
        "config": config,
    }
    save_json(config_record, run_dir / "config.json")
    print(
        f"run={run_id} model={args.model} device={device} "
        f"patches={counts} trainable={trainable_parameters:,}"
    )

    model, history, training_summary = fit(
        model,
        train_loader,
        val_loader,
        config,
        device,
        class_weights,
        run_dir,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=int(config["ignore_index"]),
    )
    test_started = time.perf_counter()
    test_metrics, confusion = evaluate(
        model,
        test_loader,
        criterion,
        device,
        config["class_names"],
        int(config["ignore_index"]),
    )
    metrics = {
        **test_metrics,
        **training_summary,
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "test_eval_seconds": time.perf_counter() - test_started,
    }
    save_json(metrics, run_dir / "metrics.json")
    pd.DataFrame(test_metrics["per_class"]).to_csv(
        run_dir / "per_class_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        confusion.numpy(),
        index=config["class_names"],
        columns=config["class_names"],
    ).to_csv(run_dir / "confusion_matrix.csv")
    print(
        f"test pixel_acc={test_metrics['pixel_accuracy']:.4f} "
        f"mIoU={test_metrics['miou']:.4f} "
        f"macro_F1={test_metrics['macro_f1']:.4f}"
    )


def command_evaluate(config: dict, args: argparse.Namespace) -> None:
    output_root = resolve_path(config["output_root"])
    run_dir = output_root / "runs" / args.run_id
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Run or checkpoint not found under {run_dir}"
        )
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = run_config["config"]
    model_name = run_config["model"]
    effective_config = copy.deepcopy(resolved)
    for model_config in effective_config["models"].values():
        model_config["pretrained"] = False

    set_seed(int(run_config["seed"]))
    torch.set_num_threads(int(resolved["torch_num_threads"]))
    device = choose_device(resolved["device_preference"])
    train_loader, val_loader, test_loader, counts = make_loaders(
        resolved,
        int(run_config["seed"]),
    )
    data_root = resolve_path(resolved["data_root"])
    stats = json.loads(
        (data_root / "manifests" / "patch_stats.json").read_text(encoding="utf-8")
    )
    class_weights = class_weights_from_counts(
        stats["train_class_pixel_counts"],
        int(resolved["ignore_index"]),
        device,
    )
    model = build_model(model_name, effective_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=int(resolved["ignore_index"]),
    )
    metrics, confusion = evaluate(
        model,
        test_loader,
        criterion,
        device,
        resolved["class_names"],
        int(resolved["ignore_index"]),
    )
    previous = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
    output = {
        **previous,
        **metrics,
        "run_id": args.run_id,
        "model": model_name,
        "seed": int(run_config["seed"]),
        "device": str(device),
        "parameter_count": run_config["total_parameters"],
        "trainable_parameters": run_config["trainable_parameters"],
        "split_counts": counts,
    }
    save_json(output, metrics_path)
    pd.DataFrame(metrics["per_class"]).to_csv(
        run_dir / "per_class_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        confusion.numpy(),
        index=resolved["class_names"],
        columns=resolved["class_names"],
    ).to_csv(run_dir / "confusion_matrix.csv")
    print(
        f"re-evaluated pixel_acc={metrics['pixel_accuracy']:.4f} "
        f"mIoU={metrics['miou']:.4f} macro_F1={metrics['macro_f1']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-oriented DeepGlobe land-cover segmentation"
    )
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--override", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download")
    subparsers.add_parser("prepare")
    train = subparsers.add_parser("train")
    train.add_argument(
        "--model",
        choices=[
            "unet_scratch",
            "resnet34_unet",
            "resnet34_unet_full_finetune",
            "resnet34_transformer",
            "resnet34_transformer_decoder",
            "resnet34_transformer_decoder_v2",
        ],
        required=True,
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--run-id", default=None)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    if args.command == "download":
        command_download(config)
    elif args.command == "prepare":
        command_prepare(config)
    elif args.command == "train":
        command_train(config, args)
    elif args.command == "evaluate":
        command_evaluate(config, args)


if __name__ == "__main__":
    main()
