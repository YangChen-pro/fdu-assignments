"""Model definitions for Flowers102 classification experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torchvision.models import convnext_tiny, efficientnet_b0, resnet18, resnet34, resnet50
from torchvision.models.resnet import BasicBlock, ResNet


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.fc[2].weight)
        nn.init.constant_(self.fc[2].bias, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class SEBasicBlock(BasicBlock):
    """ResNet BasicBlock with SE attention before residual addition."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.bn2.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def build_model(model_config: dict) -> nn.Module:
    """Create a Task1 model from config."""
    name = str(model_config.get("name", "resnet18")).lower()
    pretrained = bool(model_config.get("pretrained", True))
    num_classes = int(model_config.get("num_classes", 102))

    if name == "resnet18":
        model = _build_resnet(resnet18, "ResNet18_Weights", pretrained, num_classes)
    elif name == "resnet34":
        model = _build_resnet(resnet34, "ResNet34_Weights", pretrained, num_classes)
    elif name == "resnet50":
        model = _build_resnet(resnet50, "ResNet50_Weights", pretrained, num_classes)
    elif name == "efficientnet_b0":
        model = _build_efficientnet_b0(pretrained=pretrained, num_classes=num_classes)
    elif name == "convnext_tiny":
        model = _build_convnext_tiny(pretrained=pretrained, num_classes=num_classes)
    elif name == "se_resnet18":
        model = _build_se_resnet18(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {name}")

    return model


def classifier_parameter_names(model: nn.Module) -> set[str]:
    """Return parameter names belonging to the classification head."""
    prefixes = ("fc.", "classifier.")
    return {name for name, _ in model.named_parameters() if name.startswith(prefixes)}


def build_parameter_groups(
    model: nn.Module,
    backbone_lr: float,
    classifier_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Build optimizer parameter groups for backbone and classifier."""
    classifier_names = classifier_parameter_names(model)
    backbone_params = []
    classifier_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in classifier_names:
            classifier_params.append(parameter)
        else:
            backbone_params.append(parameter)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if classifier_params:
        groups.append({"params": classifier_params, "lr": classifier_lr, "weight_decay": weight_decay})
    return groups


def _build_resnet(builder, weights_name: str, pretrained: bool, num_classes: int) -> nn.Module:
    model = _torchvision_model(builder, weights_name, pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_se_resnet18(pretrained: bool, num_classes: int) -> nn.Module:
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        source = _torchvision_model(resnet18, "ResNet18_Weights", pretrained=True)
        source.fc = nn.Linear(source.fc.in_features, num_classes)
        _load_matching_state(model, source.state_dict())
    return model


def _build_efficientnet_b0(pretrained: bool, num_classes: int) -> nn.Module:
    model = _torchvision_model(efficientnet_b0, "EfficientNet_B0_Weights", pretrained)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_convnext_tiny(pretrained: bool, num_classes: int) -> nn.Module:
    model = _torchvision_model(convnext_tiny, "ConvNeXt_Tiny_Weights", pretrained)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _torchvision_model(builder, weights_name: str, pretrained: bool) -> nn.Module:
    try:
        import torchvision.models as tv_models

        weights_enum = getattr(tv_models, weights_name)
        weights = weights_enum.DEFAULT if pretrained else None
        return builder(weights=weights)
    except Exception:
        return builder(pretrained=pretrained)


def _load_matching_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    own_state = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in own_state and own_state[key].shape == value.shape
    }
    model.load_state_dict(filtered, strict=False)


def trainable_parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    """Count trainable parameters in an iterable."""
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
