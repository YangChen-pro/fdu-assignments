# Task 1 DPO

这一部分主要对应报告里的 DPO 数据构造、Full-AnswerOnly 主训练、continue735，以及后面的 DPO-3 / DPO-4 修正。

## 数据文件

| 报告内容 | 文件 |
|---|---|
| Plan D 400 条数据 | `../../datasets/dpo/plan_d_formal_train.jsonl`、`../../datasets/dpo/plan_d_formal_valid.jsonl` |
| Full-AnswerOnly 数据 | `../../datasets/dpo/full_answer_only_from_plan_d_train.jsonl`、`../../datasets/dpo/full_answer_only_from_plan_d_valid.jsonl` |
| DPO-3 pass@32 数据 | `../../datasets/dpo/r8_ckpt382_pass32_onepos_train.jsonl`、`../../datasets/dpo/r8_ckpt382_pass32_onepos_valid.jsonl` |
| DPO-4 失败题数据 | `../../datasets/dpo/r8_onepos_failures_solver_target_train.jsonl`、`../../datasets/dpo/r8_onepos_failures_solver_target_valid.jsonl` |

## 训练和评测

| 报告内容 | 文件 | 记录的结果 |
|---|---|---|
| Full-AnswerOnly DPO | `../dpo/full_answer_only_from_plan_d/metrics.jsonl` | loss、reward margin、step 过程 |
| Full-AnswerOnly 摘要 | `../dpo/full_answer_only_from_plan_d/train_summary.json` | 3000 steps，约 2059.6 秒 |
| continue735 结果 | `../submit/full_answer_only_continue735_test500.eval.json` | accuracy 0.324，format 1.000 |
| DPO-3 训练 | `../dpo/r8_sft_pass32_onepos_ep2/metrics.jsonl` | 训练日志 |
| DPO-3 single | `../eval/dpo_r8_sft_pass32_onepos_ep2_raw_test500_single.summary.json` | accuracy 0.432 |
| DPO-4 训练 | `../dpo/r8_onepos_targeted_failures_ep2/metrics.jsonl` | 训练日志 |
| DPO-4 single | `../eval/dpo_r8_onepos_targeted_failures_ep2_raw_test500_single.summary.json` | accuracy 0.472 |
| DPO-4 pass@32 | `../eval/dpo4_pass32_rerun.summary.json` | pass@32 accuracy 0.796 |

## 备注

- `plan_d_formal_swan` 的训练目录现在没有，只保留了 Plan D 数据和配置。
- continue735 有最终评测文件，但训练日志目录没有保留下来。
- SFT 初始化相关目录没有放在本机仓库里，所以报告里这部分主要作为中间过程说明。
