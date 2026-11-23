from __future__ import annotations

import torch

from seg_transformer.config import load_config
from seg_transformer.models import build_model


def test_scratch_unet_forward() -> None:
    config = load_config("configs/base.json")
    model = build_model("unet_scratch", config)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 128, 128))
    assert output.shape == (1, 7, 128, 128)


def test_transformer_unet_forward_without_pretrained_download() -> None:
    config = load_config("configs/base.json")
    config["models"]["resnet34_transformer"]["pretrained"] = False
    model = build_model("resnet34_transformer", config)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 128, 128))
    assert output.shape == (1, 7, 128, 128)


def test_decoder_v2_forward_without_pretrained_download() -> None:
    config = load_config("configs/base.json")
    config["models"]["resnet34_transformer_decoder_v2"]["pretrained"] = False
    model = build_model("resnet34_transformer_decoder_v2", config)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 128, 128))
    assert output.shape == (1, 7, 128, 128)
