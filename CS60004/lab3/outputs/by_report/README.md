# Lab3 输出文件对照说明

这个文件夹只是我自己整理的索引，方便看报告的时候快速找到对应的实验输出。原来的 `outputs/dpo`、`outputs/grpo`、`outputs/eval`、`outputs/submit` 都没有移动。

## 怎么看

- `Task0_PPO.md`：PPO 代码和校验位置
- `Task1_DPO.md`：DPO 数据、训练日志和评测结果
- `Task2_GRPO.md`：GRPO r3 / r7 的日志和结果
- `Task3_vLLM.md`：vLLM 相关配置和耗时记录
- `Task4_Leaderboard.md`：最终提交文件和本地评测结果

## 主要对应关系

| 报告里的内容 | 对应文件 |
|---|---|
| PPO loss 实现 | `../../scripts/0-ppo.py` |
| Full-AnswerOnly DPO 训练 | `../dpo/full_answer_only_from_plan_d/` |
| continue735 结果 | `../submit/full_answer_only_continue735_test500.eval.json` |
| GRPO r3 | `../grpo/thinking_grpo_r3/` |
| GRPO r7_vllm | `../grpo/thinking_grpo_r7_vllm/` |
| DPO-3 / DPO-4 | `../dpo/r8_sft_pass32_onepos_ep2/`、`../dpo/r8_onepos_targeted_failures_ep2/` |
| 最终 0.796 | `../submit/dpo4_pass32_bon_cleaned_test500.eval.json` |

## 说明

模型权重和 checkpoint 太大，没有放进仓库。这里主要保留的是配置、训练日志、数据构造文件、评测 summary 和提交 jsonl。
