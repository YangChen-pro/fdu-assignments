"""Aggressive from-scratch U-Net variants for HW2 Task3."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ScSE(nn.Module):
    """Concurrent spatial and channel squeeze-excitation block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweight features by channel and spatial attention."""
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class ConvScSEBlock(nn.Module):
    """Two convolution block with optional dropout and scSE attention."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, scse: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        if scse:
            layers.append(ScSE(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and attention layers."""
        return self.net(x)


class ASPP(nn.Module):
    """Small atrous spatial pyramid bridge for the U-Net bottleneck."""

    def __init__(self, channels: int, rates: tuple[int, ...] = (1, 2, 4, 6)) -> None:
        super().__init__()
        branch_channels = max(channels // len(rates), 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for rate in rates
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * len(rates), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ScSE(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse multi-dilation context features."""
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class NestedUNet(nn.Module):
    """U-Net++ style model with optional scSE, ASPP bridge and deep supervision."""

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 48,
        dropout: float = 0.0,
        deep_supervision: bool = False,
        scse: bool = True,
        aspp: bool = True,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(5)]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)
        self.conv0_0 = ConvScSEBlock(3, channels[0], dropout, scse)
        self.conv1_0 = ConvScSEBlock(channels[0], channels[1], dropout, scse)
        self.conv2_0 = ConvScSEBlock(channels[1], channels[2], dropout, scse)
        self.conv3_0 = ConvScSEBlock(channels[2], channels[3], dropout, scse)
        self.conv4_0 = ConvScSEBlock(channels[3], channels[4], dropout, scse)
        self.bridge = ASPP(channels[4]) if aspp else nn.Identity()

        self.conv0_1 = ConvScSEBlock(channels[0] + channels[1], channels[0], dropout, scse)
        self.conv1_1 = ConvScSEBlock(channels[1] + channels[2], channels[1], dropout, scse)
        self.conv2_1 = ConvScSEBlock(channels[2] + channels[3], channels[2], dropout, scse)
        self.conv3_1 = ConvScSEBlock(channels[3] + channels[4], channels[3], dropout, scse)
        self.conv0_2 = ConvScSEBlock(channels[0] * 2 + channels[1], channels[0], dropout, scse)
        self.conv1_2 = ConvScSEBlock(channels[1] * 2 + channels[2], channels[1], dropout, scse)
        self.conv2_2 = ConvScSEBlock(channels[2] * 2 + channels[3], channels[2], dropout, scse)
        self.conv0_3 = ConvScSEBlock(channels[0] * 3 + channels[1], channels[0], dropout, scse)
        self.conv1_3 = ConvScSEBlock(channels[1] * 3 + channels[2], channels[1], dropout, scse)
        self.conv0_4 = ConvScSEBlock(channels[0] * 4 + channels[1], channels[0], dropout, scse)

        self.final1 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Run the nested encoder-decoder graph."""
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.bridge(self.conv4_0(self.pool(x3_0)))

        x0_1 = self.conv0_1(torch.cat([x0_0, _up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, _up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, _up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, _up(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, _up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, _up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, _up(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, _up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, _up(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, _up(x1_3, x0_0)], dim=1))

        outputs = [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        return outputs if self.deep_supervision and self.training else outputs[-1]

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def _up(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
