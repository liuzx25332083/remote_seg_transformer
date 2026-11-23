from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DeepGlobe experiment runs.")
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--output-dir", default="outputs/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = PROJECT_ROOT / args.run_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in sorted(run_root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": metrics.get("run_id", metrics_path.parent.name),
                "model": metrics.get("model"),
                "seed": metrics.get("seed"),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_miou": metrics.get("best_val_miou"),
                "test_pixel_accuracy": metrics.get("pixel_accuracy"),
                "test_miou": metrics.get("miou"),
                "test_macro_f1": metrics.get("macro_f1"),
                "total_parameters": metrics.get("total_parameters"),
                "trainable_parameters": metrics.get("trainable_parameters"),
                "device": metrics.get("device"),
                "train_seconds": metrics.get("total_epoch_seconds"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {run_root}")
    runs = pd.DataFrame(rows).sort_values(["model", "seed"])
    summary = (
        runs.groupby("model", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            val_miou_mean=("best_val_miou", "mean"),
            val_miou_std=("best_val_miou", "std"),
            test_miou_mean=("test_miou", "mean"),
            test_miou_std=("test_miou", "std"),
            macro_f1_mean=("test_macro_f1", "mean"),
            macro_f1_std=("test_macro_f1", "std"),
            trainable_parameters=("trainable_parameters", "first"),
        )
        .sort_values("test_miou_mean", ascending=False)
    )
    runs.to_csv(output_dir / "runs.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)

    print("\nEach completed run\n")
    print(runs.to_string(index=False))
    print("\nModel summary\n")
    print(summary.to_string(index=False))

    if len(runs):
        figure, axis = plt.subplots(figsize=(9, 4.8))
        labels = summary["model"].tolist()
        x = range(len(labels))
        axis.bar(
            x,
            summary["test_miou_mean"],
            yerr=summary["test_miou_std"].fillna(0),
            capsize=4,
            color="#3568A8",
        )
        axis.set_xticks(list(x), labels, rotation=20, ha="right")
        axis.set_ylabel("Test mIoU")
        axis.set_title("DeepGlobe segmentation model comparison")
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / "model_comparison_miou.png", dpi=180)
        plt.close(figure)
    print(f"\nSaved analysis to {output_dir}")


if __name__ == "__main__":
    main()
