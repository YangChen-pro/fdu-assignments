# Task 4 最终提交

报告最后用的是 DPO-4 模型的 pass@32 BoN cleaned 结果。

## 最终结果文件

| 内容 | 文件 |
|---|---|
| 最终提交 jsonl | `../submit/dpo4_pass32_bon_cleaned_test500.jsonl` |
| 本地评测 | `../submit/dpo4_pass32_bon_cleaned_test500.eval.json` |
| meta 信息 | `../submit/dpo4_pass32_bon_cleaned_test500_meta.json` |
| pass@32 原始 summary | `../eval/dpo4_pass32_rerun.summary.json` |
| pass@32 候选输出 | `../eval/dpo4_pass32_rerun.pass32.jsonl` |

`dpo4_pass32_bon_cleaned_test500.eval.json` 里对应报告的最终数字：

- accuracy：0.796
- format_rate：0.800
- avg_output_tokens：185.398

## 其他候选结果

| 候选 | 文件 | 说明 |
|---|---|---|
| Full-AnswerOnly continue735 | `../submit/full_answer_only_continue735_test500.eval.json` | accuracy 0.324 |
| DPO-4 single cleaned | `../submit/dpo_r8_onepos_targeted_failures_ep2_single_cleaned_test500.eval.json` | accuracy 0.472 |
| GRPO r3 single | `../submit/thinking_grpo_r3_ckpt50_single_test500.jsonl` | 有 jsonl 和 meta |
| GRPO r7 single / c32 | `../submit/thinking_grpo_r7_final_single_cleaned_test500.jsonl`、`../submit/thinking_grpo_r7_final_c32_cleaned_test500.jsonl` | 只有输出文件，没有 eval json |

## 备注

`../../configs/submit_solver_always_raw_test500.json` 是调试用配置，里面的 `rule_fallback_mode=always` 会直接调用规则 solver，不是报告最终结果来源。
