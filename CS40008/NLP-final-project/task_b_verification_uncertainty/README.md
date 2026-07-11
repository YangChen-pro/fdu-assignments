# Task B: Verification and Uncertainty

本方向研究推理时验证与不确定性估计能否提升常识验证可靠性。代码支持两种运行方式：一种是离线 batch 工作流，先生成 JSONL prompts，外部调用模型后把 responses 写回 JSONL；另一种是直接用本地 vLLM 读取 Task A 的 Qwen 模型注册表完成推理。两种方式最终都复用本目录脚本解析、重排和评估。

## 路径约定

输入数据复用 Task A 的处理结果：

```text
data/processed/comve_task_a_train.csv
data/processed/comve_task_a_dev.csv
data/processed/comve_task_a_test.csv
```

运行输出：

```text
outputs/task_b/
results/task_b_predictions.csv
results/task_b_metrics.csv
```

预测文件统一字段：

```text
id,method,pred,gold,correct,notes
```

## 代码结构

```text
task_b_verification_uncertainty/
  cli.py       # prompt 生成、response 解析、rerank、不确定性指标
  prompts.py   # direct/constraint/candidate/verifier prompt 和解析器
  metrics.py   # accuracy、confidence bucket、selective prediction
  paths.py     # 输出路径常量
  vllm_infer.py # 本地 vLLM direct/constraint/candidate/verifier 推理入口
```

## Local vLLM Pipeline

如果本地已经按 Task A 约定准备好模型目录，可直接跑验证增强流水线：

```bash
MODEL=qwen3_1_7b THINKING_MODE=thinking CUDA_VISIBLE_DEVICES=0 \
  bash scripts/task_b/run_vllm_verification_pipeline.sh
```

该脚本依次运行：

```text
direct prompting
constraint-first prompting
candidate generation k=3
verifier scoring
generate-verify-rerank
combined task_b_predictions.csv
combined task_b_metrics.csv
```

输出路径：

```text
outputs/task_b/vllm/*_aligned_*_responses.jsonl
results/task_b/vllm/*_aligned_direct_predictions.csv
results/task_b/vllm/*_aligned_constraint_predictions.csv
results/task_b/vllm/*_aligned_rerank_predictions.csv
results/task_b/vllm/*_aligned_predictions.csv
results/task_b/vllm/*_aligned_metrics.csv
```

默认 `PROMPT_TAG=aligned`。这是为了避免 `--resume` 复用早期 JSON+reason prompt 的旧结果。若需要复现实验草稿阶段的旧输出，可显式设置其他 tag 并指定旧文件。

如果想把同一个 split 按样本区间拆到多张 GPU 上跑，可以额外设置 `SAMPLE_START` 和 `SAMPLE_END`。例如把 test 集拆成前后两半：

```bash
MODEL=qwen3_1_7b THINKING_MODE=thinking CUDA_VISIBLE_DEVICES=0 SAMPLE_START=0 SAMPLE_END=500 \
  bash scripts/task_b/run_vllm_verification_pipeline.sh

MODEL=qwen3_1_7b THINKING_MODE=thinking CUDA_VISIBLE_DEVICES=1 SAMPLE_START=500 SAMPLE_END=1000 \
  bash scripts/task_b/run_vllm_verification_pipeline.sh
```

两段跑完后，再把对应的 `*_predictions.csv` 合并即可。

一致性置信度和 selective prediction 单独运行：

```bash
MODEL=qwen3_1_7b THINKING_MODE=thinking REPEAT=5 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/task_b/run_vllm_uncertainty.sh
```

输出：

```text
results/task_b_consistency_samples.csv
results/task_b_uncertainty_predictions.csv
results/task_b_uncertainty_metrics.csv
```

## Direct / Constraint-first prompts

direct 和 constraint-first 的可比基线与 Task A 对齐：模型只允许输出单个数字 `0` 或 `1`，不输出 JSON 或解释。candidate/verifier 仍保留 JSON，因为 generate-verify-rerank 需要候选解释和 verifier score。

生成 direct prompts：

```bash
python3 -m task_b_verification_uncertainty.cli make-prompts \
  --split test \
  --mode direct \
  --output outputs/task_b/direct_prompts.jsonl
```

生成 constraint-first prompts：

```bash
python3 -m task_b_verification_uncertainty.cli make-prompts \
  --split test \
  --mode constraint \
  --output outputs/task_b/constraint_prompts.jsonl
```

LLM 返回结果整理为 JSONL 后，每行至少包含：

```json
{"id":"test_00000","method":"task_b_direct:direct","gold":0,"response":"0"}
```

解析为统一预测 CSV：

```bash
python3 -m task_b_verification_uncertainty.cli parse-responses \
  --input outputs/task_b/direct_responses.jsonl \
  --output results/task_b_predictions.csv
```

生成准确率指标：

```bash
python3 -m task_b_verification_uncertainty.cli evaluate \
  --predictions results/task_b_predictions.csv \
  --output results/task_b_metrics.csv
```

## Generate-Verify-Rerank

第一步，生成候选答案 prompts。`--repeat 3` 表示每个样本生成 3 个候选：

```bash
python3 -m task_b_verification_uncertainty.cli make-prompts \
  --split test \
  --mode candidate \
  --repeat 3 \
  --output outputs/task_b/candidate_prompts_k3.jsonl
```

第二步，模型返回候选 responses 后，生成 verifier prompts：

```bash
python3 -m task_b_verification_uncertainty.cli make-verifier-prompts \
  --candidates outputs/task_b/candidate_responses_k3.jsonl \
  --output outputs/task_b/verifier_prompts_k3.jsonl
```

第三步，模型返回 verifier responses 后，按 verifier score 重排：

```bash
python3 -m task_b_verification_uncertainty.cli rerank \
  --candidates outputs/task_b/candidate_responses_k3.jsonl \
  --verifier-responses outputs/task_b/verifier_responses_k3.jsonl \
  --output results/task_b_rerank_predictions.csv \
  --metrics-output results/task_b_rerank_metrics.csv
```

Verifier response 推荐格式：

```json
{"candidate_id":"test_00000::task_b_candidate:candidate::0","response":"{\"score\":5,\"final_label\":0,\"reason\":\"...\"}"}
```

## Consistency Confidence / Selective Prediction

如果已有多次采样的预测 CSV，可计算多数投票、置信度分桶和 selective prediction：

```bash
python3 -m task_b_verification_uncertainty.cli uncertainty \
  --samples results/task_b_prediction_samples.csv \
  --method task_b_consistency_confidence_k5 \
  --thresholds 0.6 0.8 1.0 \
  --output results/task_b_uncertainty_predictions.csv \
  --metrics-output results/task_b_uncertainty_metrics.csv
```

置信度定义：

```text
confidence = majority_label_count / number_of_samples
```

## Metrics

- Accuracy
- Verifier rerank accuracy
- Mean confidence
- Confidence bucket accuracy
- Selective coverage
- Selective accuracy

## Analysis Questions

- Constraint-first 是否比 direct prompting 更稳定？
- Verifier 是否真的能筛掉错误解释，还是会偏向更长、更自信的答案？
- Confidence 是否与正确率相关？
- 模型高 confidence 但错误的样本有什么共同特征？
