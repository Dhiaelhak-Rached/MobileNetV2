"""MobileNetV2 acoustic classifier: 1-channel log-Mel -> 2 logits.

Input [B, 1, 128, 32] (1s window). ImageNet-pretrained backbone; first conv
adapted 3->1 channel by averaging RGB weights; classifier head replaced with
Dropout(0.3) + Linear(1280, 2).

CLASS_NAMES = ["non_drone", "drone"] — logit index 1 = drone.

Run directly for forward-pass smoke test: python src/models/mobilenet_acoustic.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CLASS_NAMES = ["non_drone", "drone"]


def build_model(pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)

    # first conv: 3 -> 1 channel, keep pretrained info by averaging RGB
    conv = model.features[0][0]
    new_conv = nn.Conv2d(1, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_conv

    # head: 1280 -> Dropout -> 2
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 2))
    return model


def head_parameters(model: nn.Module):
    return model.classifier.parameters()


def backbone_parameters(model: nn.Module):
    return model.features.parameters()


def demo() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    x = torch.randn(8, 1, 128, 32, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 2), out.shape
    assert torch.isfinite(out).all()
    n_params = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in head_parameters(model))
    print(f"OK: forward [8,1,128,32] -> [8,2] on {device}; "
          f"params {n_params/1e6:.2f}M (head {n_head})")


if __name__ == "__main__":
    demo()
