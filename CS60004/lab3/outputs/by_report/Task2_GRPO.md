# Task 2 GRPO

这一部分对应报告里的 r3 和 r7_vllm 两轮 GRPO 训练。

## 代码和配置

| 内容 | 文件 |
|---|---|
| GRPO 主脚本 | `../../scripts/2-grpo.py` |
| 训练工具函数 | `../../scripts/train_utils.py` |
| r3 配置 | `../../configs/thinking_grpo_r3.yaml` |
| r7_vllm 配置 | `../../configs/thinking_grpo_r7_vllm.yaml` |

## 输出文件

| 报告内容 | 文件 | 记录的结果 |
|---|---|---|
| r3 训练日志 | `../grpo/thinking_grpo_r3/metrics.jsonl` | reward、accuracy、format、entropy、耗时 |
| r3 step 50 评测 | `../grpo/thinking_grpo_r3/eval_step_50.json` | accuracy 0.332，format 0.398 |
| r7_vllm 训练日志 | `../grpo/thinking_grpo_r7_vllm/metrics.jsonl` | 500 steps 的训练记录 |
| r7_vllm 摘要 | `../grpo/thinking_grpo_r7_vllm/train_summary.json` | 500 steps，约 87339 秒 |

## 备注

- r7 的 checkpoint 权重没有放进 Git，只保留了配置和日志。
- `../submit/thinking_grpo_r7_final_c32_cleaned_test500.jsonl` 是 r7 的一个候选输出，但没有配套的 eval json。
