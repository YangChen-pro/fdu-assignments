# 项目分工介绍：LLM 常识验证可靠性评估

## 1. 项目主题

本课程项目研究大语言模型在常识验证任务中的可靠性。我们使用 SemEval-2020 Task 4 ComVE 子任务 A：模型需要在两个语义相近的句子中判断哪一句违背常识。

项目不只比较模型准确率，而是进一步分析：

- 模型是否会因为 prompt、句子顺序、示例选择或采样随机性而改变答案；
- 模型是否能通过解释、验证和多次采样得到更可靠的判断；
- 模型在哪些常识类型上仍然容易出错。

最终项目可以概括为：

> 在 ComVE 常识验证任务上，系统评估 LLM 的准确率、鲁棒性、一致性和不确定性，并探索轻量的推理时增强方法。

## 2. 总体实验框架

我们共用同一份 ComVE Task A 数据集和统一评价指标。基础方法包括：

1. RoBERTa 微调基线；
2. LLM zero-shot prompting；
3. LLM few-shot prompting；
4. LLM self-consistency。

在此基础上，项目分成两个可以并行完成的方向：

- Task A：鲁棒性与一致性评估；
- Task B：验证增强与不确定性感知推理。

两个方向各自独立跑实验，最后合并结果表、图表和错误分析。

## 3. 同学 A：鲁棒性与一致性评估

### 研究问题

同学 A 主要回答：

> LLM 在常识验证任务中是否稳定？如果改变输入形式，模型是否仍然给出等价答案？

### 主要任务

同学 A 负责基础模型对比和输入扰动实验：

1. 跑 RoBERTa supervised baseline；
2. 跑 LLM zero-shot、few-shot 和 self-consistency；
3. 做句子顺序交换实验；
4. 做 prompt variation 实验；
5. 做 few-shot 示例随机种子实验；
6. 汇总 accuracy、order consistency、prompt consistency 和 sampling consistency。

### 需要产出的结果

同学 A 最后需要交付：

- `results/task_a/roberta/predictions.csv`
- `results/task_a/roberta/metrics.csv`
- 一张主结果表：RoBERTa、zero-shot、few-shot、self-consistency 的准确率；
- 一张鲁棒性表：顺序交换、prompt 改写、few-shot seed 和采样一致性；
- 2-3 个错误案例：说明模型为什么不稳定。

### 报告中负责的部分

同学 A 主要负责报告中的：

- Baseline Methods；
- Robustness Experiments；
- Robustness Results；
- 与输入扰动相关的错误分析。

## 4. 同学 B：验证增强与不确定性感知推理

### 研究问题

同学 B 主要回答：

> 如果不重新训练模型，只在推理时加入解释、验证和不确定性估计，能否让模型判断更可靠？

### 主要任务

同学 B 负责轻量的 test-time enhancement 实验：

1. Direct prompting：直接输出违背常识的句子编号；
2. Constraint-first prompting：先抽取常识约束，再判断两个句子；
3. Generate-verify-rerank：生成多个候选解释和答案，再用 verifier 选择；
4. Consistency confidence：多次采样，用多数标签比例作为置信度；
5. Selective prediction：只回答高置信样本，画 coverage-accuracy 曲线；
6. 可选做 counterfactual minimal pair：生成少量最小反事实样本做案例分析。

### 需要产出的结果

同学 B 最后需要交付：

- `results/task_b_predictions.csv`
- `results/task_b_metrics.csv`
- 一张验证增强方法对比表；
- 一张 confidence bucket 或 coverage-accuracy 图；
- 2-3 个解释或验证失败案例。

### 报告中负责的部分

同学 B 主要负责报告中的：

- Verification-enhanced Methods；
- Uncertainty Estimation；
- Selective Prediction；
- 与解释、验证、不确定性相关的错误分析。

## 5. 合并方式

两个方向最终合并时，需要统一以下内容：

### 统一数据和预测格式

所有预测文件使用统一字段：

```text
id,method,pred,gold,correct,notes
```

其中：

- `id`：样本编号；
- `method`：方法名称；
- `pred`：模型预测标签；
- `gold`：真实标签；
- `correct`：是否预测正确；
- `notes`：可选备注，例如 prompt 类型、采样次数、错误类型。

### 统一指标

最终报告至少包含：

- Accuracy；
- Order consistency；
- Prompt consistency；
- Sampling consistency；
- Confidence bucket accuracy；
- Coverage-accuracy。

### 统一错误分析类型

错误案例统一归入以下类别：

- 物理常识错误；
- 语言歧义；
- 过度推理；
- prompt 敏感；
- 句子顺序敏感；
- 高置信但错误。

## 6. 最终报告结构建议

最终报告可以按以下结构撰写：

1. Introduction：说明常识验证任务和 LLM 可靠性问题；
2. Problem Definition：介绍 ComVE Task A；
3. Methods：介绍 RoBERTa、LLM prompting、Task A 和 Task B 方法；
4. Experiments：展示基础准确率、鲁棒性实验和验证增强实验；
5. Analysis：合并两位同学的错误案例和现象分析；
6. Conclusion：总结 LLM 常识验证的能力和局限。

## 7. 时间安排建议

| 阶段 | 同学 A | 同学 B | 合并任务 |
|---|---|---|---|
| 第 1 阶段 | 跑基础 baseline | 整理 prompt 模板 | 确认数据格式 |
| 第 2 阶段 | 做鲁棒性实验 | 做验证增强实验 | 汇总预测文件 |
| 第 3 阶段 | 整理错误案例 | 整理不确定性图表 | 合并结果表 |
| 第 4 阶段 | 写 Task A 部分 | 写 Task B 部分 | 完成最终报告 |

## 8. 一句话分工总结

- 同学 A 负责证明：模型是否稳定。
- 同学 B 负责探索：如何让模型更可靠。
- 最终项目共同回答：LLM 在常识验证中不只是能不能答对，更重要的是能不能稳定、可解释、可置信地答对。
