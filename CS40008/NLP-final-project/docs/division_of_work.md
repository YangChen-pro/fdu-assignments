# 分工文档：ComVE 常识验证可靠性评估

## 项目目标

本项目研究大语言模型在 ComVE Task A 常识验证任务中的可靠性。基础目标是复现 RoBERTa 与 LLM 提示式方法的准确率对比；扩展目标是从两个并行方向分析模型是否稳定、是否知道自己何时不可靠，以及是否能通过推理时验证提升判断质量。

## 共同约定

- 数据集：SemEval-2020 Task 4 ComVE 子任务 A。
- 输入字段：`id,sent0,sent1,gold`。
- 预测字段：`id,method,pred,gold,correct,notes`。
- 所有方法必须使用同一测试集和同一标签定义：`0` 表示 `sent0` 违背常识，`1` 表示 `sent1` 违背常识。
- 不伪造实验结果；未完成结果在报告中用 `待补充` 标记。

## Task A：鲁棒性与一致性评估

负责人 A 回答的问题：模型在常规输入和扰动输入下是否稳定？

### 必做实验

1. RoBERTa supervised baseline：使用训练集微调二分类器。
2. LLM zero-shot：只给任务说明，直接输出 `0/1`。
3. LLM few-shot：加入少量平衡训练样例。
4. Self-consistency：同一样本采样 `k=3/5/10` 次，多数投票。
5. Order swap：交换 `sent0/sent1` 后测试预测是否同步翻转。
6. Prompt variation：使用 3 个语义等价 prompt 测试一致性。

### 指标

- Accuracy
- Order consistency
- Prompt consistency
- Sampling consistency
- Accuracy-cost curve

### 交付

- `results/task_a/roberta/predictions.csv`
- `results/task_a/roberta/metrics.csv`
- 2-3 个典型错误案例：顺序敏感、提示敏感、采样不稳定或示例敏感。

## Task B：验证增强与不确定性感知推理

负责人 B 回答的问题：推理时验证和不确定性估计能否让常识判断更可靠？

### 必做实验

1. Direct prompting：直接判断哪一句违背常识。
2. Constraint-first prompting：先抽取常识约束，再逐句验证，最后输出标签。
3. Generate-verify-rerank：生成多个候选解释和答案，再由 verifier 打分重排。
4. Consistency confidence：同一样本多次采样，用多数标签比例作为 confidence。
5. Selective prediction：只回答 confidence 高于阈值的样本，画 coverage-accuracy 曲线。

### 可选实验

- Counterfactual minimal pair：用 LLM 对少量样本生成最小反事实改写，检查模型是否保持正确判断。

### 指标

- Accuracy
- Verifier rerank gain
- Confidence bucket accuracy
- Coverage-accuracy curve
- Explanation-label consistency

### 交付

- `results/task_b_predictions.csv`
- `results/task_b_metrics.csv`
- 解释和验证失败案例：理由正确但标签错、标签正确但理由错、verifier 误判。

## 合并方式

最终报告中，Task A 作为“可靠性评估”主线，Task B 作为“可靠性增强”主线。两个方向的结果表分别汇总，再在错误分析部分合并 taxonomy：

- 物理常识错误
- 语言歧义
- 过度推理
- 顺序或 prompt 敏感
- 高不确定性样本

合并时优先统一术语、统一指标定义、统一测试集样本编号，避免两个方向各自解释同一指标。
