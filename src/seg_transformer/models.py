from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNetScratch(nn.Module):
    """Small U-Net baseline trained without pretrained encoder weights."""

    def __init__(self, num_classes: int, base_channels: int = 32):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        self.enc1 = DoubleConv(3, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.enc4 = DoubleConv(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[3] * 2)
        self.dec4 = DoubleConv(channels[3] * 3, channels[3])
        self.dec3 = DoubleConv(channels[2] + channels[3], channels[2])
        self.dec2 = DoubleConv(channels[1] + channels[2], channels[1])
        self.dec1 = DoubleConv(channels[0] + channels[1], channels[0])
        self.head = nn.Conv2d(channels[0], num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))
        x = self.dec4(torch.cat([self._up(x5), x4], dim=1))
        x = self.dec3(torch.cat([self._up(x), x3], dim=1))
        x = self.dec2(torch.cat([self._up(x), x2], dim=1))
        x = self.dec1(torch.cat([self._up(x), x1], dim=1))
        return self.head(x)


def _sinusoidal_2d(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("Transformer dimension must be divisible by 4")
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, 2, device=device, dtype=torch.float32)
        / half
    )
    pe_y = y.unsqueeze(-1) * frequencies
    pe_x = x.unsqueeze(-1) * frequencies
    encoded = torch.cat(
        [torch.sin(pe_y), torch.cos(pe_y), torch.sin(pe_x), torch.cos(pe_x)],
        dim=-1,
    )
    return encoded.flatten(0, 1).unsqueeze(0)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        position = _sinusoidal_2d(height, width, channels, feature_map.device)
        tokens = tokens + position.to(tokens.dtype)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_transformer: bool = False,
        heads: int = 4,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        self.refiner = (
            SpatialTransformerBlock(out_channels, heads=heads)
            if use_transformer
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
        return self.refiner(self.conv(x))


class ResNetTransformerUNet(nn.Module):
    """ResNet-34 encoder with an optional lightweight Transformer bottleneck."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool,
        use_transformer_bottleneck: bool,
        use_transformer_decoder: bool,
        transformer_decoder_stage: int = 4,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        freeze_prefixes: list[str] | None = None,
    ):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if freeze_prefixes:
            for name, parameter in self.named_parameters():
                if any(name == prefix or name.startswith(prefix + ".") for prefix in freeze_prefixes):
                    parameter.requires_grad = False

        self.use_transformer_bottleneck = use_transformer_bottleneck
        if use_transformer_bottleneck:
            self.bottleneck_projection = nn.Conv2d(512, d_model, 1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.bottleneck_output = nn.Conv2d(d_model, 256, 1)
            bottleneck_channels = 256
        else:
            self.bottleneck_output = nn.Identity()
            bottleneck_channels = 512

        self.decoder4 = DecoderBlock(
            bottleneck_channels,
            256,
            256,
            use_transformer_decoder and transformer_decoder_stage == 4,
            num_heads,
        )
        self.decoder3 = DecoderBlock(
            256,
            128,
            128,
            use_transformer_decoder and transformer_decoder_stage == 3,
            num_heads,
        )
        self.decoder2 = DecoderBlock(128, 64, 64, False, num_heads)
        self.decoder1 = DecoderBlock(64, 0, 32, False, num_heads)
        self.decoder0 = DecoderBlock(32, 0, 16, False, num_heads)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        bottleneck = self.layer4(skip3)

        if self.use_transformer_bottleneck:
            bottleneck = self.bottleneck_projection(bottleneck)
            batch, channels, height, width = bottleneck.shape
            tokens = bottleneck.flatten(2).transpose(1, 2)
            position = _sinusoidal_2d(height, width, channels, x.device)
            tokens = self.transformer_encoder(tokens + position.to(tokens.dtype))
            bottleneck = tokens.transpose(1, 2).reshape(
                batch,
                channels,
                height,
                width,
            )
            bottleneck = self.bottleneck_output(bottleneck)

        x = self.decoder4(bottleneck, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.decoder1(x)
        x = self.decoder0(x)
        return self.head(x)


def build_model(model_name: str, config: dict[str, Any]) -> nn.Module:
    if model_name == "unet_scratch":
        return UNetScratch(
            num_classes=int(config["num_classes"]),
            base_channels=int(config.get("base_channels", 32)),
        )
    if model_name not in config["models"]:
        raise ValueError(f"Unknown model: {model_name}")
    model_config = config["models"][model_name]
    return ResNetTransformerUNet(
        num_classes=int(config["num_classes"]),
        pretrained=bool(model_config["pretrained"]),
        use_transformer_bottleneck=bool(
            model_config["use_transformer_bottleneck"]
        ),
        use_transformer_decoder=bool(model_config["use_transformer_decoder"]),
        transformer_decoder_stage=int(
            model_config.get("transformer_decoder_stage", 4)
        ),
        d_model=int(model_config["d_model"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        freeze_prefixes=model_config.get("freeze_prefixes"),
    )


def parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
