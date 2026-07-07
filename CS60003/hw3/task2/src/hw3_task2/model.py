from __future__ import annotations

import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy as LeRobotACTPolicy


class ACTPolicy(nn.Module):
    """LeRobot ACT policy adapter for CALVIN parquet batches."""

    def __init__(
        self,
        image_size: int,
        use_wrist_image: bool,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int,
        nheads: int,
        num_layers: int,
        dropout: float,
        amp: bool,
        use_vae: bool,
        kl_weight: float,
    ) -> None:
        super().__init__()
        input_features = {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, image_size, image_size)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        }
        if use_wrist_image:
            input_features["observation.images.wrist_image"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, image_size, image_size),
            )
        output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
        config = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,
            dim_model=hidden_dim,
            n_heads=nheads,
            dim_feedforward=hidden_dim * 4,
            n_encoder_layers=num_layers,
            n_decoder_layers=1,
            use_vae=use_vae,
            dropout=dropout,
            kl_weight=kl_weight,
            use_amp=amp,
        )
        self.policy = LeRobotACTPolicy(config)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        policy_batch = {
            "observation.images.image": batch["image"],
            "observation.state": batch["state"],
            "action": batch["actions"],
            "action_is_pad": ~batch["valid"].bool(),
        }
        image_tensors = [policy_batch["observation.images.image"]]
        if "wrist_image" in batch:
            policy_batch["observation.images.wrist_image"] = batch["wrist_image"]
            image_tensors.append(batch["wrist_image"])
        policy_batch["observation.images"] = image_tensors
        if self.training:
            policy_batch_for_loss = dict(policy_batch)
            policy_batch_for_loss.pop("observation.images")
            return self.policy(policy_batch_for_loss)
        return self._inference_loss(policy_batch)

    def _inference_loss(self, policy_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        previous_use_vae = self.policy.config.use_vae
        self.policy.config.use_vae = False
        try:
            actions_hat, _ = self.policy.model(policy_batch)
        finally:
            self.policy.config.use_vae = previous_use_vae
        valid = ~policy_batch["action_is_pad"].unsqueeze(-1)
        denom = valid.sum().clamp_min(1) * actions_hat.shape[-1]
        l1_loss = (torch.abs(actions_hat - policy_batch["action"]) * valid).sum() / denom
        return l1_loss, {"l1_loss": float(l1_loss.item())}


def build_policy(config, image_size: int, use_wrist_image: bool, chunk_size: int, amp: bool) -> ACTPolicy:
    return ACTPolicy(
        image_size=image_size,
        use_wrist_image=use_wrist_image,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        chunk_size=chunk_size,
        hidden_dim=config.hidden_dim,
        nheads=config.nheads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        amp=amp,
        use_vae=config.use_vae,
        kl_weight=config.kl_weight,
    )
