# 基于 ResNet-34 与轻量 Transformer 的 DeepGlobe 土地覆盖分割

这是一个独立课程项目，不与 `eurosat_landcover` 共用数据、输出或训练代码。

## 研究问题

DeepGlobe 原始图像分辨率为 2448x2448。完整训练对纯 CPU 不现实，因此本项目
使用固定原图子集和图块分类式分割流程，研究：

1. ImageNet 预训练 ResNet-34 编码器是否优于从头训练的 U-Net；
2. 在 ResNet-34 深层特征上加入 Transformer 全局注意力，是否改善类别混淆；
3. 在粗分辨率解码阶段加入 Transformer，是否值得额外 CPU 计算；
4. Agriculture、Forest、Rangeland、Water 等类别的逐类 IoU 如何变化。

## 数据处理

- 公共镜像：`Zongrong/DeepGlobe_LandCover_Classification_Dataset`
- 仅使用带掩码的 803 张训练原图。
- 固定随机抽取 96 张原图，种子为 42。
- 先按原始大图分层划分 70%/15%/15%，再切图块，避免相邻图块泄漏。
- 每张原图生成 4x4=16 个图块：从 256x256 区域缩放为 128x128。
- 掩码颜色映射：

| ID | 类别 | RGB |
|---:|---|---|
| 0 | Urban | (0, 255, 255) |
| 1 | Agriculture | (255, 255, 0) |
| 2 | Rangeland | (255, 0, 255) |
| 3 | Forest | (0, 255, 0) |
| 4 | Water | (0, 0, 255) |
| 5 | Barren | (255, 255, 255) |
| 6 | Unknown | (0, 0, 0) |

`Unknown` 不进入损失和评价。

## 模型

- `unet_scratch`：小型 U-Net，从头训练。
- `resnet34_unet`：ImageNet ResNet-34 编码器 + 卷积解码器。
- `resnet34_transformer`：ResNet-34 + Transformer bottleneck + 卷积解码器。
- `resnet34_transformer_decoder`：再在 H/16 解码阶段加入 Transformer refiner。

ResNet-34 默认冻结 `conv1`、`bn1`、`layer1`、`layer2`，训练后两层编码器、
Transformer 和解码器。

## 安装与运行

### Google Colab（推荐）

1. 把 `remote_seg_transformer_colab.zip` 上传到 Google Drive 的 `MyDrive`
   根目录；
2. 在 Colab 中打开 `DeepGlobe_ResNet34_Transformer_Colab.ipynb`；
3. `Runtime -> Change runtime type -> T4 GPU`；
4. 从上到下执行 Notebook。

Notebook 会自动解压项目、挂载 Drive、下载数据并保存训练结果。免费 GPU
通常明显快于本机 CPU，但 Colab 也可能因长时间运行而断开；重新运行同一格
时会跳过已有完成结果。

项目复用已经安装好的 PyTorch CPU 环境，但代码和输出完全独立。

```powershell
cd C:\Users\LENOVO\Desktop\cs231n\remote_seg_transformer
$env:PYTHONPATH=(Resolve-Path "src").Path
$python = "C:\Users\LENOVO\Desktop\cs231n\eurosat_landcover\.venv\Scripts\python.exe"

& $python -m pytest -q
& $python -m seg_transformer.cli --config configs/base.json download
& $python -m seg_transformer.cli --config configs/base.json prepare
& $python -m seg_transformer.cli --config configs/base.json --override configs/smoke.json train --model resnet34_transformer --run-id smoke_resnet34_transformer
```

正式实验示例：

```powershell
& $python -m seg_transformer.cli --config configs/base.json train --model unet_scratch --seed 42
& $python -m seg_transformer.cli --config configs/base.json train --model resnet34_unet --seed 42
& $python -m seg_transformer.cli --config configs/base.json train --model resnet34_transformer --seed 42
& $python -m seg_transformer.cli --config configs/base.json train --model resnet34_transformer_decoder --seed 42
```

## 输出

每个运行目录位于 `outputs/runs/<run_id>/`，包括：

- 解析后的配置与设备信息；
- 每轮训练损失、验证 mIoU、Macro-F1 和耗时；
- 最佳模型；
- 测试集总指标；
- 逐类 Precision、Recall、F1、IoU；
- 混淆矩阵。

所有报告数字必须来自这些运行产物，不使用预设或模拟结果。
