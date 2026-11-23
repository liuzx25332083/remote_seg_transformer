from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "notebooks" / "DeepGlobe_ResNet34_Transformer_Colab.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    }
    notebook["cells"] = [
        markdown(
            """
# DeepGlobe 土地覆盖分割：ResNet-34 + 轻量 Transformer

本 Notebook 用于 Google Colab GPU。项目独立于 EuroSAT 作业。

比较：

1. 从头训练 U-Net；
2. ImageNet 预训练 ResNet-34 + 卷积解码器；
3. ResNet-34 + Transformer bottleneck；
4. 再加入轻量 Transformer decoder。

所有输入先按原图划分，再切成 128x128 图块，避免相邻图块跨集合泄漏。
"""
        ),
        markdown("## 1. 挂载 Google Drive"),
        code(
            """
from google.colab import drive
drive.mount("/content/drive")
"""
        ),
        markdown(
            """
先把本地的 `remote_seg_transformer_colab.zip` 上传到 Google Drive 的
`MyDrive` 根目录。

下面的代码会自动把压缩包解压为：

`MyDrive/remote_seg_transformer`

如果已经解压过，代码会更新到压缩包中的版本，但不会覆盖已有数据或训练结果。
"""
        ),
        code(
            """
import os
import shutil
import sys
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_ROOT = DRIVE_ROOT / "remote_seg_transformer"
ZIP_PATH = DRIVE_ROOT / "remote_seg_transformer_colab.zip"

if not PROJECT_ROOT.exists() and not ZIP_PATH.exists():
    raise FileNotFoundError(
        "请先把 remote_seg_transformer_colab.zip 上传到 MyDrive 根目录"
    )
if ZIP_PATH.exists():
    shutil.unpack_archive(ZIP_PATH, DRIVE_ROOT)

assert PROJECT_ROOT.exists(), f"Project not found: {PROJECT_ROOT}"
os.chdir(PROJECT_ROOT)
os.environ["PYTHONPATH"] = str(PROJECT_ROOT / "src")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
print("Project:", PROJECT_ROOT)
"""
        ),
        markdown("## 2. 检查 GPU 与依赖"),
        code(
            """
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
"""
        ),
        markdown(
            """
如果缺少依赖，取消下一格的注释并运行。Colab 通常已经自带兼容版本的
PyTorch 和 TorchVision，优先保留 Colab 默认版本。
"""
        ),
        code(
            """
# !pip install -q pandas Pillow scikit-learn tqdm
"""
        ),
        markdown("## 3. 运行测试"),
        code(
            """
import subprocess

subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
"""
        ),
        markdown(
            """
## 4. 下载并准备 DeepGlobe 子集

固定抽取 96 张带掩码的原始大图，下载量约数百 MB。数据会保存在 Drive，
下次 Colab 会话可以直接复用。
"""
        ),
        code(
            """
from seg_transformer.cli import command_download, command_prepare, load_config

config = load_config("configs/base.json")
if not (Path(config["data_root"]) / "manifests" / "patch_manifest.csv").exists():
    command_download(config)
    command_prepare(config)
else:
    print("Prepared patches already exist; skipping download.")
"""
        ),
        markdown("## 5. 检查数据划分与类别分布"),
        code(
            """
import json
import pandas as pd

data_root = Path(config["data_root"])
stats = json.loads((data_root / "manifests" / "patch_stats.json").read_text(encoding="utf-8"))
print(json.dumps({
    "images_by_split": stats["images_by_split"],
    "patches_by_split": stats["patches_by_split"],
    "class_pixel_counts": stats["class_pixel_counts"],
}, ensure_ascii=False, indent=2))

image_split = pd.read_csv(data_root / "manifests" / "image_split_manifest.csv")
print(image_split.groupby("split")["image_id"].nunique())
"""
        ),
        markdown(
            """
## 6. 单模型训练

先建议训练 `resnet34_transformer`。如需比较，依次修改 `MODEL_NAME` 后运行。
"""
        ),
        code(
            """
import subprocess

MODEL_NAME = "resnet34_transformer"
SEED = 42
RUN_ID = f"{MODEL_NAME}_seed{SEED}"
metrics_path = Path("outputs") / "runs" / RUN_ID / "metrics.json"

if metrics_path.exists():
    print("Completed run already exists:", metrics_path)
else:
    subprocess.run([
        sys.executable,
        "-m",
        "seg_transformer.cli",
        "--config",
        "configs/base.json",
        "--override",
        "configs/colab_gpu.json",
        "train",
        "--model",
        MODEL_NAME,
        "--seed",
        str(SEED),
        "--run-id",
        RUN_ID,
    ], check=True)

# Recompute metrics from the saved checkpoint. This also counts an
# unpredicted class as F1=0 instead of dropping it from Macro-F1.
subprocess.run([
    sys.executable,
    "-m",
    "seg_transformer.cli",
    "--config",
    "configs/base.json",
    "evaluate",
    "--run-id",
    RUN_ID,
], check=True)
"""
        ),
        markdown("## 7. 查看结果"),
        code(
            """
import json

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
print(json.dumps({
    "model": metrics["model"],
    "best_epoch": metrics["best_epoch"],
    "best_val_miou": metrics["best_val_miou"],
    "test_pixel_accuracy": metrics["pixel_accuracy"],
    "test_miou": metrics["miou"],
    "test_macro_f1": metrics["macro_f1"],
    "trainable_parameters": metrics["trainable_parameters"],
    "device": metrics["device"],
}, ensure_ascii=False, indent=2))
"""
        ),
        code(
            """
import pandas as pd

run_dir = metrics_path.parent
display(pd.read_csv(run_dir / "history.csv"))
display(pd.read_csv(run_dir / "per_class_metrics.csv"))
display(pd.read_csv(run_dir / "confusion_matrix.csv", index_col=0))
"""
        ),
        markdown(
            """
## 8. 运行其余对照模型

下面的列表可以一次性跑完剩余模型。Colab 可能在一轮训练结束后断开，
重连时再次运行同一格即可跳过已完成运行。
"""
        ),
        code(
            """
for model_name in [
    "unet_scratch",
    "resnet34_unet",
    "resnet34_transformer_decoder",
]:
    run_id = f"{model_name}_seed42"
    output = Path("outputs") / "runs" / run_id / "metrics.json"
    if output.exists():
        print("skip", run_id)
        continue
    subprocess.run([
        sys.executable,
        "-m",
        "seg_transformer.cli",
        "--config",
        "configs/base.json",
        "--override",
        "configs/colab_gpu.json",
        "train",
        "--model",
        model_name,
        "--seed",
        "42",
        "--run-id",
        run_id,
    ], check=True)
    subprocess.run([
        sys.executable,
        "-m",
        "seg_transformer.cli",
        "--config",
        "configs/base.json",
        "evaluate",
        "--run-id",
        run_id,
    ], check=True)
"""
        ),
        markdown(
            """
## 9. 补跑 seed 7 与 seed 123

单种子只能说明一次运行，不能估计初始化波动。本节会把四个模型分别用
seed 7 和 seed 123 再运行，已经完成的实验会自动跳过。
"""
        ),
        code(
            """
for seed in [7, 123]:
    for model_name in [
        "unet_scratch",
        "resnet34_unet",
        "resnet34_transformer",
        "resnet34_transformer_decoder",
    ]:
        run_id = f"{model_name}_seed{seed}"
        output = Path("outputs") / "runs" / run_id / "metrics.json"
        if output.exists():
            print("skip", run_id)
            continue
        subprocess.run([
            sys.executable,
            "-m",
            "seg_transformer.cli",
            "--config",
            "configs/base.json",
            "--override",
            "configs/colab_gpu.json",
            "train",
            "--model",
            model_name,
            "--seed",
            str(seed),
            "--run-id",
            run_id,
        ], check=True)
        subprocess.run([
            sys.executable,
            "-m",
            "seg_transformer.cli",
            "--config",
            "configs/base.json",
            "evaluate",
            "--run-id",
            run_id,
        ], check=True)
"""
        ),
        markdown(
            """
## 10. 汇总所有运行结果

这一节会读取全部 `metrics.json`，显示每次运行结果、三种子均值和标准差，
并生成模型对比图。
"""
        ),
        code(
            """
!python scripts/summarize_results.py

import pandas as pd
display(pd.read_csv("outputs/analysis/model_summary.csv"))
from IPython.display import Image as DisplayImage
display(DisplayImage(filename="outputs/analysis/model_comparison_miou.png"))
"""
        ),
        markdown(
            """
## 11. Stage 2 控制实验

下面的三组实验一次只改变一个因素：

1. 完整微调 ResNet-34，不再冻结前两层；
2. 保持同一组原图和图块位置，把输入从 128 提高到 256；
3. 把 Transformer decoder 从 H/16 移到 H/8，提供更大的 token 网格。

先完成 seed 42。确认训练方向和显存成本后，再扩展到 seed 7 和 123。
"""
        ),
        code(
            """
STAGE2_RUN_ID = "resnet34_unet_full_finetune_seed42"
STAGE2_METRICS = Path("outputs") / "runs" / STAGE2_RUN_ID / "metrics.json"

if not STAGE2_METRICS.exists():
    subprocess.run([
        sys.executable,
        "-m",
        "seg_transformer.cli",
        "--config",
        "configs/base.json",
        "--override",
        "configs/full_finetune.json",
        "train",
        "--model",
        "resnet34_unet_full_finetune",
        "--seed",
        "42",
        "--run-id",
        STAGE2_RUN_ID,
    ], check=True)
else:
    print("skip", STAGE2_RUN_ID)

subprocess.run([
    sys.executable,
    "-m",
    "seg_transformer.cli",
    "--config",
    "configs/base.json",
    "--override",
    "configs/full_finetune.json",
    "evaluate",
    "--run-id",
    STAGE2_RUN_ID,
], check=True)
"""
        ),
        code(
            """
if not (Path("data_256") / "manifests" / "patch_manifest.csv").exists():
    subprocess.run([
        sys.executable,
        "-m",
        "seg_transformer.cli",
        "--config",
        "configs/base.json",
        "--override",
        "configs/resolution_256.json",
        "prepare",
    ], check=True)

STAGE2_RUN_ID = "resnet34_unet_256_seed42"
STAGE2_METRICS = Path("outputs") / "runs" / STAGE2_RUN_ID / "metrics.json"

if not STAGE2_METRICS.exists():
    subprocess.run([
        sys.executable,
        "-m",
        "seg_transformer.cli",
        "--config",
        "configs/base.json",
        "--override",
        "configs/resolution_256.json",
        "train",
        "--model",
        "resnet34_unet",
        "--seed",
        "42",
        "--run-id",
        STAGE2_RUN_ID,
    ], check=True)
else:
    print("skip", STAGE2_RUN_ID)

subprocess.run([
    sys.executable,
    "-m",
    "seg_transformer.cli",
    "--config",
    "configs/base.json",
    "--override",
    "configs/resolution_256.json",
    "evaluate",
    "--run-id",
    STAGE2_RUN_ID,
], check=True)
"""
        ),
        code(
            """
STAGE2_RUN_ID = "resnet34_transformer_decoder_v2_seed42"
STAGE2_METRICS = Path("outputs") / "runs" / STAGE2_RUN_ID / "metrics.json"

if not STAGE2_METRICS.exists():
    subprocess.run([
        sys.executable,
        "-m",
        "seg_transformer.cli",
        "--config",
        "configs/base.json",
        "--override",
        "configs/colab_gpu.json",
        "train",
        "--model",
        "resnet34_transformer_decoder_v2",
        "--seed",
        "42",
        "--run-id",
        STAGE2_RUN_ID,
    ], check=True)
else:
    print("skip", STAGE2_RUN_ID)

subprocess.run([
    sys.executable,
    "-m",
    "seg_transformer.cli",
    "--config",
    "configs/base.json",
    "evaluate",
    "--run-id",
    STAGE2_RUN_ID,
], check=True)
"""
        ),
        markdown(
            """
## 12. 结果解释边界

- 只使用固定的 96 张原始图像子集，不等同于完整 DeepGlobe 基准。
- 按原图切分可以避免图块泄漏，但仍不是按地区或采集条带划分。
- 128x128 输入会损失细节，mIoU 不能直接与 1024x1024 原报告比较。
- Transformer 架构本身不是新贡献，重点是迁移学习、全局上下文、逐类混淆和计算成本的对照。
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Notebook written to {OUTPUT}")


if __name__ == "__main__":
    build()
