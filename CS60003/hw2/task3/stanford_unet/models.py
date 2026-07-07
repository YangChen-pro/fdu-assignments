"""Hand-written U-Net family models for HW2 Task3."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .advanced_models import NestedUNet


class DoubleConv(nn.Module):
    """Two Conv-BatchNorm-ReLU blocks used in the plain U-Net."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double convolution block."""
        return self.net(x)


class ResidualDoubleConv(nn.Module):
    """Residual two-convolution block for a from-scratch ResUNet variant."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.net = nn.Sequential(*layers)
        self.projection: nn.Module
        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolution and final activation."""
        return self.activation(self.net(x) + self.projection(x))


class Down(nn.Module):
    """Downsampling block: max pooling followed by a configurable conv block."""

    def __init__(self, in_channels: int, out_channels: int, block: type[nn.Module], dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), block(in_channels, out_channels, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one encoder downsampling step."""
        return self.net(x)


class AttentionGate(nn.Module):
    """Additive attention gate for filtering U-Net skip features."""

    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        hidden_channels = max(skip_channels // 2, 1)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Return attention-filtered skip features."""
        gate = _match_spatial_size(gate, skip)
        return skip * self.psi(self.skip_proj(skip) + self.gate_proj(gate))


class Up(nn.Module):
    """Upsampling block with optional attention-gated skip concatenation."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        block: type[nn.Module],
        bilinear: bool,
        dropout: float = 0.0,
        attention: bool = False,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            up_channels = out_channels
        self.attention = AttentionGate(skip_channels, up_channels) if attention else None
        self.conv = block(up_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample decoder features and concatenate encoder features."""
        x = self.up(x)
        x = _match_spatial_size(x, skip)
        if self.attention is not None:
            skip = self.attention(skip, x)
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    """Final 1x1 convolution that maps features to class logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel class logits."""
        return self.conv(x)


class UNet(nn.Module):
    """Configurable hand-written U-Net / ResUNet / Attention U-Net."""

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 32,
        bilinear: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
        attention: bool = False,
    ) -> None:
        super().__init__()
        block = ResidualDoubleConv if residual else DoubleConv
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.inc = block(3, channels[0], dropout)
        self.down1 = Down(channels[0], channels[1], block, dropout)
        self.down2 = Down(channels[1], channels[2], block, dropout)
        self.down3 = Down(channels[2], channels[3], block, dropout)
        self.down4 = Down(channels[3], channels[4], block, dropout)
        self.up1 = Up(channels[4], channels[3], channels[3], block, bilinear, dropout, attention)
        self.up2 = Up(channels[3], channels[2], channels[2], block, bilinear, dropout, attention)
        self.up3 = Up(channels[2], channels[1], channels[1], block, bilinear, dropout, attention)
        self.up4 = Up(channels[1], channels[0], channels[0], block, bilinear, dropout, attention)
        self.outc = OutConv(channels[0], num_classes)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder, decoder and final classifier head."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def build_model(model_config: dict) -> nn.Module:
    """Construct a hand-written U-Net family model from config."""
    if bool(model_config.get("pretrained", False)):
        raise ValueError("Task3 forbids pretrained weights.")
    name = str(model_config.get("name", "unet")).lower()
    if name in {"nested_unet", "unetpp", "unet_plus_plus"}:
        return NestedUNet(
            num_classes=int(model_config.get("num_classes", 8)),
            base_channels=int(model_config.get("base_channels", 48)),
            dropout=float(model_config.get("dropout", 0.0)),
            deep_supervision=bool(model_config.get("deep_supervision", False)),
            scse=bool(model_config.get("scse", True)),
            aspp=bool(model_config.get("aspp", True)),
        )
    variants = {
        "unet": (False, False),
        "resunet": (True, False),
        "attention_unet": (False, True),
        "attention_resunet": (True, True),
    }
    if name not in variants:
        raise ValueError(f"Unsupported model: {name}")
    residual, attention = variants[name]
    return UNet(
        num_classes=int(model_config.get("num_classes", 8)),
        base_channels=int(model_config.get("base_channels", 32)),
        bilinear=bool(model_config.get("bilinear", False)),
        dropout=float(model_config.get("dropout", 0.0)),
        residual=residual,
        attention=attention,
    )


def _match_spatial_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
