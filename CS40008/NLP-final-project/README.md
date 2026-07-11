# LLM Commonsense Reliability on ComVE

本项目面向 SemEval-2020 Task 4 ComVE 子任务 A：给定两个语义相近的英文句子，判断哪一个违背常识。项目不只比较模型准确率，还评估大语言模型在常识验证中的可靠性，包括鲁棒性、一致性、验证增强和不确定性感知能力。

## Research Questions

1. 提示式大语言模型在 ComVE Task A 上能否接近或超过 RoBERTa 微调基线？
2. 模型对句子顺序、提示模板、少样本示例和采样随机性是否稳定？
3. 推理时验证、候选重排和不确定性估计能否提升常识判断可靠性？
4. 模型剩余错误主要来自物理常识缺失、语言歧义、过度推理，还是输出随机性？

## Project Structure

```text
project/
  README.md
  data/                         # ComVE 原始与预处理数据
  docs/
    division_of_work.md         # 两人分工与合并协议
  task_a_robustness/            # 鲁棒性与一致性方向
  task_b_verification_uncertainty/
                                # 验证增强与不确定性感知方向
  results/                      # 结果表、预测文件、图表占位
  paper/                        # 干净 ACL 模版与中文报告草稿
```

## Two Parallel Tasks

- Task A：鲁棒性与一致性评估。负责 RoBERTa/LLM 基线、自一致性、顺序交换、提示改写和 few-shot seed 鲁棒性。
- Task B：验证增强与不确定性感知推理。负责 constraint-first prompting、generate-verify-rerank、consistency confidence、selective prediction 和少量 counterfactual minimal pair 分析。

两个方向共享 ComVE Task A 数据划分，最终通过统一预测文件和指标表合并。

## Expected Deliverables

- `results/task_a/roberta/` 中的 RoBERTa 预测与指标
- `results/task_a/vllm/` 中的 LLM predictions、metrics 和 raw outputs
- `results/task_a/report_tables/` 中的 Task A 报告汇总表
- `results/task_b_predictions.csv` and `results/task_b_metrics.csv`（由 Task B 补充）
- `paper/main.tex` 编译得到课程报告

## Notes

当前 Task A 已包含可执行实验脚本、数据、预测文件、指标文件与报告汇总表。模型权重、缓存、运行日志和 PID 仍放在 checkpoints/ 与 outputs/ 下，不作为普通 Git 文本产物管理。
