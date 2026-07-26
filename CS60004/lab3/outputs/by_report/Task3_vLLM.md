# Task 3 vLLM

报告里 vLLM 主要用于 GRPO rollout 加速，最终 r7_vllm 用的是 `vllm_exclusive`。

| 内容 | 文件 |
|---|---|
| vLLM 启动脚本 | `../../scripts/3-vllm.sh` |
| vLLM API 调用 | `../../scripts/train_utils.py` |
| r7 vLLM 配置 | `../../configs/thinking_grpo_r7_vllm.yaml` |
| r7 每步耗时 | `../grpo/thinking_grpo_r7_vllm/metrics.jsonl` |
| r7 总耗时 | `../grpo/thinking_grpo_r7_vllm/train_summary.json` |

`metrics.jsonl` 里有 `rollout_time_sec`、`update_time_sec`、`step_time_sec`、`gpu_mem_mb` 等字段，可以对应报告里的耗时和显存描述。

目前没有单独保留 local backend 的完整对照日志，所以 local vs vLLM 的加速比主要作为实验观察记录。
