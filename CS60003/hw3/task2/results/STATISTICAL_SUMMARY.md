# HW3 Task2 Statistical Summary

## 主结论

本节以未使用 D 环境选择 checkpoint 的 `final.pt` 作为主口径：`act_splitABC` 在 splitD 上的 Action L1 为 `0.1549013643`，低于 `act_splitA` 的 `0.1886211640`。
相对提升为 `17.88%`，说明多环境训练在未见过的 D 环境上仍有稳定优势。

`best.pt` 只作为参考结果；它使用 splitD 离线误差做 checkpoint 选择，因此不作为最干净的 zero-shot 主结论。

## Paired 统计

| 粒度 | 配对数 | ABC 更优数量 | ABC 更优比例 | 平均 L1 提升 | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| episode | 5124 | 4041 | 78.86% | 0.032959 | [0.031753, 0.034230] |
| task | 389 | 355 | 91.26% | 0.031384 | [0.028808, 0.034193] |
| action_dim | 7 | 7 | 100.00% | 0.033752 | [0.012247, 0.070059] |

解释：`平均 L1 提升 = act_splitA_final - act_splitABC_final`，大于 0 表示多环境模型更好。

## 证据文件

- `statistical_summary.csv` / `statistical_summary.json`：本页对应的机器可读汇总。
- `act_splitA_final_episode_breakdown.csv` 与 `act_splitABC_final_episode_breakdown.csv`：episode 级配对来源。
- `act_splitA_final_task_breakdown.csv` 与 `act_splitABC_final_task_breakdown.csv`：task 级配对来源。
- `act_splitA_final_action_dim_breakdown.csv` 与 `act_splitABC_final_action_dim_breakdown.csv`：动作维度级配对来源。
