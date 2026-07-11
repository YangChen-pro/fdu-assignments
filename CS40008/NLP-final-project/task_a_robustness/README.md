# Task A: Robustness and Consistency

本目录负责 ComVE Task A 的鲁棒性与一致性实验。代码目标是把数据、预测和指标格式固定下来，并用统一接口跑 RoBERTa、vLLM zero-shot、order swap、8-shot few-shot 和 self-consistency。

## 路径约定

输入数据：

```text
data/processed/comve_task_a_train.csv
data/processed/comve_task_a_dev.csv
data/processed/comve_task_a_test.csv
```

模型位置：

```text
checkpoints/hf_cache/        # Hugging Face 原始模型缓存，例如 roberta-base
checkpoints/llm_models/      # Qwen / Qwen3.5 原始模型目录
checkpoints/task_a_roberta/  # RoBERTa 微调输出
```

运行输出：

```text
outputs/task_a/              # 运行日志、PID、临时 prompt
results/task_a/              # Task A 预测、指标和 vLLM raw outputs
```

`checkpoints/hf_cache/` 是原始模型下载缓存路径，`checkpoints/llm_models/` 是 Qwen 系列原始模型路径，`checkpoints/task_a_roberta/` 是 RoBERTa 微调模型路径。模型内容通常较大，仍由 `.gitignore` 控制；目录本身通过 `.gitkeep` 保留。

## 代码结构

```text
task_a_robustness/
  data.py         # 读取 ComVE schema、写预测文件、生成 order swap 数据
  metrics.py      # accuracy、order consistency、prompt consistency
  prompts.py      # LLM zero-shot/few-shot prompt 模板和标签解析
  sampling.py     # self-consistency 多次采样多数投票
  roberta.py      # RoBERTa fine-tuning 与预测入口
  vllm_config.py  # Qwen 本地模型路径和 README 推荐生成参数
  vllm_helpers.py # vLLM prompt、resume、输出整理辅助逻辑
  vllm_infer.py   # vLLM in-process 推理入口
  cli.py          # 数据验证、prompt batch、评估等通用命令
  paths.py        # 项目路径常量
```

## 快速验证数据

```bash
python3 -m task_a_robustness.cli validate-data
```

预期输出：

```text
train: rows=10000, gold0=5000, gold1=5000
dev: rows=1000, gold0=500, gold1=500
test: rows=1000, gold0=500, gold1=500
```

## RoBERTa baseline

RoBERTa 训练依赖 PyTorch、Transformers、NumPy 和 Accelerate。按当前实验环境的包管理方式安装即可；仓库不单独维护 requirements 文件。

训练：

```bash
python3 -m task_a_robustness.roberta train \
  --model-name roberta-base \
  --cache-dir checkpoints/hf_cache \
  --output-dir checkpoints/task_a_roberta \
  --epochs 3 \
  --batch-size 16
```

预测测试集：

```bash
python3 -m task_a_robustness.roberta predict \
  --model-dir checkpoints/task_a_roberta \
  --cache-dir checkpoints/hf_cache \
  --split test \
  --output results/task_a/roberta/predictions.csv
```

生成指标：

```bash
python3 -m task_a_robustness.cli evaluate \
  --split test \
  --predictions results/task_a/roberta/predictions.csv \
  --output results/task_a/roberta/metrics.csv
```

## vLLM 入口

vLLM 使用代码内直接 import 的方式，不走 Transformers 推理，也不部署 OpenAI 兼容服务。Qwen3/Qwen3.5 的 thinking 与 non-thinking 参数在 `vllm_config.py` 中按模型 README 写死；默认最大输出长度为：

```text
non-thinking: 10240 tokens
thinking:     16384 tokens
```

常用脚本默认跑较快的 no-thinking 模式；需要补 thinking 对照时，加 `MODE_LIST="non_thinking thinking"` 覆盖。

```bash
bash scripts/task_a/run_vllm_all_zero_shot_chunked.sh
bash scripts/task_a/run_vllm_order_swap.sh
bash scripts/task_a/run_vllm_few_shot_8.sh
bash scripts/task_a/run_vllm_self_consistency.sh
```

结果目录：

```text
results/task_a/vllm/predictions/
results/task_a/vllm/metrics/
results/task_a/vllm/raw_outputs/
```

`run_vllm_self_consistency.sh` 默认只跑 `qwen3_1_7b` 和 `qwen3_5_2b` 的 no-thinking self-consistency，每题每模板重复 5 次。需要扩大模型范围或补 thinking 时可用环境变量覆盖：

```bash
MODEL_LIST="qwen3_0_6b qwen3_1_7b qwen3_5_0_8b qwen3_5_2b" \
MODE_LIST="non_thinking thinking" \
REPEAT=5 \
bash scripts/task_a/run_vllm_self_consistency.sh
```

## LLM zero-shot / few-shot prompt batch

生成 zero-shot prompts：

```bash
python3 -m task_a_robustness.cli make-prompts \
  --split test \
  --output outputs/task_a/llm_zero_shot_prompts.jsonl
```

生成 few-shot prompts：

```bash
python3 -m task_a_robustness.cli make-prompts \
  --split test \
  --few-shot 8 \
  --output outputs/task_a/llm_few_shot_prompts.jsonl
```

生成 self-consistency prompts，例如每个样本重复 5 次：

```bash
python3 -m task_a_robustness.cli make-prompts \
  --split test \
  --repeat 5 \
  --output outputs/task_a/llm_sc_k5_prompts.jsonl
```

LLM 返回结果整理成 JSONL 后，可解析为统一预测 CSV。每行至少包含：

```json
{"id":"test_00000","method":"llm_zero_shot:direct","gold":0,"response":"0"}
```

解析命令：

```bash
python3 -m task_a_robustness.cli parse-llm-responses \
  --input outputs/task_a/llm_responses.jsonl \
  --output results/task_a/roberta/predictions.csv
```

## Self-consistency 多数投票

如果已有重复采样预测 CSV，可生成多数投票结果。默认保留原 method 分组：

```bash
python3 -m task_a_robustness.cli self-consistency \
  --samples results/task_a/vllm/predictions/samples.csv \
  --output results/task_a/vllm/predictions/self_consistency_predictions.csv \
  --metrics-output results/task_a/vllm/metrics/self_consistency_metrics.csv
```

如果要把所有样本合并成一个指定 method，也可以传 `--method`。

## Order swap

生成顺序交换测试集：

```bash
python3 -m task_a_robustness.cli make-order-swap \
  --split test \
  --output outputs/task_a/comve_task_a_test_swap.csv
```

交换后标签会自动翻转：`gold=0` 变成 `gold=1`，反之亦然。

如果已经分别得到原始测试集与交换测试集预测，可计算顺序一致性。默认会对两个文件共同包含的所有 method 逐一计算：

```bash
python3 -m task_a_robustness.cli evaluate-order-swap \
  --original results/task_a/roberta/predictions.csv \
  --swapped results/task_a/roberta/predictions_swap.csv \
  --output results/task_a/roberta/order_metrics.csv
```

需要只算一个 method 时传 `--method`。

## 结果 schema

预测文件：

```text
id,method,pred,gold,correct,notes
```

指标文件：

```text
method,metric,value,split,notes
```
