import torch
import torch.nn as nn

def compute_ppo_clip_loss(
    old_log_probs: torch.Tensor,  # 旧策略对数概率 (batch,)
    new_log_probs: torch.Tensor,  # 新策略对数概率 (batch,)
    advantages: torch.Tensor,     # 优势函数 (batch,)
    clip_ratio: float = 0.2       # PPO clip 阈值
) -> torch.Tensor:
    # 计算新旧策略概率比值
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 裁剪后的概率比值
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # PPO surrogate objective
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages

    # PPO loss: 负号 + 取较小值
    loss = -torch.minimum(surr1, surr2)

    return loss


def validate_ppo_implementation():
    torch.manual_seed(42)

    old_log_probs = torch.tensor([-0.5, -1.0, -0.3, -0.8], requires_grad=False)
    new_log_probs = torch.tensor([-0.4, -1.1, -0.25, -0.7], requires_grad=True)
    advantages = torch.tensor([1.2, -0.8, 0.9, -1.5])
    clip_ratio = 0.2

    loss = compute_ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_ratio)
    total_loss = loss.mean()

    total_loss.backward()
    student_grad = new_log_probs.grad.clone()

    true_loss = torch.tensor([-1.3262053, 0.7238699, -0.9461439, 1.6577564])
    true_grad = torch.tensor([-0.3315513, 0.18096748, -0.23653598, 0.4144391])

    loss_correct = torch.allclose(loss, true_loss, atol=1e-4)
    grad_correct = torch.allclose(student_grad, true_grad, atol=1e-4)

    print(f"损失计算: {loss_correct}")
    print(f"梯度计算: {grad_correct}")


if __name__ == "__main__":
    validate_ppo_implementation()
