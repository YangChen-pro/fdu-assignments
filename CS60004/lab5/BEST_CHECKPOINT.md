# Lab5 当前最佳统一模型训练记录

本文档记录当前自测最佳统一生成模型的来源、训练步骤和评估口径，便于复现实验和报告引用。

## 结论

当前最佳 checkpoint：

```text
lab5/outputs/task4/megatron_instruction_teacher_grpo_rlvr_probe100/v0-20260531-201911/checkpoint-100
```

自测分数：

| 任务 | 分数 | 评估口径 |
| --- | ---: | --- |
| 角色模拟 | 72.40 / 100 | 500 条 interview，全量 LLM-as-a-Judge，`overall=3.62/5` |
| 通用指令 | 70.06 / 100 | Lab5 instruction dev，IFEval strict，`110/157` 条约束通过 |
| 70/30 加权估计 | 71.70 / 100 | `0.7 * 72.40 + 0.3 * 70.06` |

注意：这个 checkpoint 本身主要通过通用指令数据训练出来，没有用角色训练数据继续训练。角色模拟推理时使用 Task1 的 Memory Agent 检索角色记忆，再由同一个 checkpoint 生成回答；没有使用外部模型代答、重试择优、多模型路由或生成后处理。

## 数据来源与隔离

训练和评估严格分离：

- `lab5/public_pack/` 只读。
- Lab5 instruction dev/test 只用于训练数据的精确 prompt 去重过滤，不作为训练答案。
- 角色模拟自测使用 `individual_simulation_data/interview` 中的 500 条 interview 问题。
- 通用指令自测使用 `lab5/public_pack/data/instruction_following/dev.jsonl`。

训练数据：

| 数据目录 | 用途 | 规模 | 来源与清洗 |
| --- | --- | ---: | --- |
| `lab5/outputs/task4/data_instruction_only_tulu_r1` | 第一阶段通用 SFT | train 16000 / val 350 | ModelScope `LLM-Research/tulu-3-sft-personas-instruction-following`；过滤与 Lab5 instruction dev/test 完全相同的 prompt，实际重叠为 0 |
| `lab5/outputs/task4/data_instruction_grpo_rlvr_r1` | IFEval 规则奖励 GRPO | train 12000 / val 500 | ModelScope `LLM-Research/RLVR-IFeval`；过滤与 Lab5 instruction dev/test 完全相同的 prompt，实际重叠为 0 |
| `lab5/outputs/task4/data_instruction_teacher_Qwen3.5-35B-A3B_rlvr_probe1k` | teacher SFT | train 963 / val 107 | 从外部 RLVR-IFeval prompt 生成；teacher 为 `Qwen3.5-35B-A3B`；尝试 1400 条，保留 1070 条通过公开规则检查的回答，330 条未通过规则的样本丢弃；与 Lab5 instruction dev/test 完全相同的 prompt 重叠为 0 |

## 训练链路

所有训练在 AutoDL 上完成：

```text
工作目录：/root/autodl-tmp/CS60004
conda 环境：lab3
基础模型：models/Qwen2.5-1.5B-Instruct
训练框架：ms-swift Megatron，全参数训练，保存 safetensors
记录工具：SwanLab
```

训练分四步：

### 1. Tulu instruction SFT

从基础模型训练到：

```text
lab5/outputs/task4/megatron_instruction_only_tulu_sft_r1/v0-20260531-145015/checkpoint-180
```

关键参数：

```text
model = models/Qwen2.5-1.5B-Instruct
dataset = lab5/outputs/task4/data_instruction_only_tulu_r1/sft_train.jsonl
val_dataset = lab5/outputs/task4/data_instruction_only_tulu_r1/sft_val.jsonl
train_iters = 180
lr = 1e-5
min_lr = 1e-6
global_batch_size = 16
micro_batch_size = 1
max_length = 4096
save_steps = 90
eval_steps = 90
tuner_type = full
```

等价命令：

```bash
cd /root/autodl-tmp/CS60004
MODEL=models/Qwen2.5-1.5B-Instruct \
DATASET=lab5/outputs/task4/data_instruction_only_tulu_r1/sft_train.jsonl \
VAL_DATASET=lab5/outputs/task4/data_instruction_only_tulu_r1/sft_val.jsonl \
OUTPUT_DIR=lab5/outputs/task4/megatron_instruction_only_tulu_sft_r1 \
TRAIN_ITERS=180 \
LR=1e-5 \
MIN_LR=1e-6 \
GLOBAL_BATCH_SIZE=16 \
MAX_LENGTH=4096 \
SAVE_STEPS=90 \
EVAL_STEPS=90 \
bash lab5/task_scripts/task2_train_sft.sh
```

### 2. RLVR-IFeval GRPO

从第 1 步 checkpoint 继续训练到：

```text
lab5/outputs/task4/megatron_instruction_grpo_rlvr_r1/v0-20260531-152712/checkpoint-40
```

关键参数：

```text
model = lab5/outputs/task4/megatron_instruction_only_tulu_sft_r1/v0-20260531-145015/checkpoint-180
dataset = lab5/outputs/task4/data_instruction_grpo_rlvr_r1/grpo_train.jsonl
val_dataset = lab5/outputs/task4/data_instruction_grpo_rlvr_r1/grpo_val.jsonl
reward_funcs = ifeval_rule
train_iters = 40
lr = 1e-6
min_lr = 1e-7
beta = 0.02
num_generations = 4
generation_batch_size = 8
global_batch_size = 8
max_length = 4096
max_completion_length = 768
vllm_max_model_len = 8192
vllm_max_num_seqs = 8
```

等价命令：

```bash
cd /root/autodl-tmp/CS60004
MODEL=lab5/outputs/task4/megatron_instruction_only_tulu_sft_r1/v0-20260531-145015/checkpoint-180 \
REF_MODEL=lab5/outputs/task4/megatron_instruction_only_tulu_sft_r1/v0-20260531-145015/checkpoint-180 \
DATASET=lab5/outputs/task4/data_instruction_grpo_rlvr_r1/grpo_train.jsonl \
VAL_DATASET=lab5/outputs/task4/data_instruction_grpo_rlvr_r1/grpo_val.jsonl \
OUTPUT_DIR=lab5/outputs/task4/megatron_instruction_grpo_rlvr_r1 \
REWARD_FUNCS=ifeval_rule \
REWARD_WEIGHTS=1.0 \
TRAIN_ITERS=40 \
LR=1e-6 \
MIN_LR=1e-7 \
BETA=0.02 \
NUM_GENERATIONS=4 \
GENERATION_BATCH_SIZE=8 \
GLOBAL_BATCH_SIZE=8 \
MAX_LENGTH=4096 \
MAX_COMPLETION_LENGTH=768 \
SAVE_STEPS=40 \
EVAL_STEPS=40 \
VLLM_GPU_MEMORY_UTILIZATION=0.20 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_MAX_NUM_SEQS=8 \
bash lab5/task_scripts/task2_train_rlhf.sh
```

### 3. Teacher SFT

从第 2 步 checkpoint 继续训练到：

```text
lab5/outputs/task4/megatron_instruction_teacher_sft_probe1k_lr1e5/v0-20260531-185953/checkpoint-80
```

关键参数：

```text
model = lab5/outputs/task4/megatron_instruction_grpo_rlvr_r1/v0-20260531-152712/checkpoint-40
dataset = lab5/outputs/task4/data_instruction_teacher_Qwen3.5-35B-A3B_rlvr_probe1k/sft_train.jsonl
val_dataset = lab5/outputs/task4/data_instruction_teacher_Qwen3.5-35B-A3B_rlvr_probe1k/sft_val.jsonl
train_iters = 80
lr = 1e-5
min_lr = 1e-6
global_batch_size = 32
micro_batch_size = 1
max_length = 4096
save_steps = 40
eval_steps = 40
tuner_type = full
```

等价命令：

```bash
cd /root/autodl-tmp/CS60004
MODEL=lab5/outputs/task4/megatron_instruction_grpo_rlvr_r1/v0-20260531-152712/checkpoint-40 \
DATASET=lab5/outputs/task4/data_instruction_teacher_Qwen3.5-35B-A3B_rlvr_probe1k/sft_train.jsonl \
VAL_DATASET=lab5/outputs/task4/data_instruction_teacher_Qwen3.5-35B-A3B_rlvr_probe1k/sft_val.jsonl \
OUTPUT_DIR=lab5/outputs/task4/megatron_instruction_teacher_sft_probe1k_lr1e5 \
TRAIN_ITERS=80 \
LR=1e-5 \
MIN_LR=1e-6 \
GLOBAL_BATCH_SIZE=32 \
MAX_LENGTH=4096 \
SAVE_STEPS=40 \
EVAL_STEPS=40 \
bash lab5/task_scripts/task2_train_sft.sh
```

### 4. Final RLVR-IFeval GRPO

从第 3 步 checkpoint 继续训练，得到当前最佳 checkpoint：

```text
lab5/outputs/task4/megatron_instruction_teacher_grpo_rlvr_probe100/v0-20260531-201911/checkpoint-100
```

关键参数：

```text
model = lab5/outputs/task4/megatron_instruction_teacher_sft_probe1k_lr1e5/v0-20260531-185953/checkpoint-80
dataset = lab5/outputs/task4/data_instruction_grpo_rlvr_r1/grpo_train.jsonl
val_dataset = lab5/outputs/task4/data_instruction_grpo_rlvr_r1/grpo_val.jsonl
reward_funcs = ifeval_rule
train_iters = 100
lr = 2e-6
min_lr = 2e-7
beta = 0.02
num_generations = 4
generation_batch_size = 8
global_batch_size = 8
micro_batch_size = 1
max_length = 6144
max_completion_length = 1024
vllm_gpu_memory_utilization = 0.22
vllm_max_model_len = 8192
vllm_max_num_seqs = 8
save_steps = 50
eval_steps = 50
```

等价命令：

```bash
cd /root/autodl-tmp/CS60004
MODEL=lab5/outputs/task4/megatron_instruction_teacher_sft_probe1k_lr1e5/v0-20260531-185953/checkpoint-80 \
REF_MODEL=lab5/outputs/task4/megatron_instruction_teacher_sft_probe1k_lr1e5/v0-20260531-185953/checkpoint-80 \
DATASET=lab5/outputs/task4/data_instruction_grpo_rlvr_r1/grpo_train.jsonl \
VAL_DATASET=lab5/outputs/task4/data_instruction_grpo_rlvr_r1/grpo_val.jsonl \
OUTPUT_DIR=lab5/outputs/task4/megatron_instruction_teacher_grpo_rlvr_probe100 \
REWARD_FUNCS=ifeval_rule \
REWARD_WEIGHTS=1.0 \
TRAIN_ITERS=100 \
LR=2e-6 \
MIN_LR=2e-7 \
BETA=0.02 \
NUM_GENERATIONS=4 \
GENERATION_BATCH_SIZE=8 \
GLOBAL_BATCH_SIZE=8 \
MAX_LENGTH=6144 \
MAX_COMPLETION_LENGTH=1024 \
SAVE_STEPS=50 \
EVAL_STEPS=50 \
VLLM_GPU_MEMORY_UTILIZATION=0.22 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_MAX_NUM_SEQS=8 \
bash lab5/task_scripts/task2_train_rlhf.sh
```

## 评估方式

### 角色模拟

推理：

- 生成模型：`checkpoint-100`
- 推理服务：`vllm serve`
- 记忆检索：`Qwen3-Embedding-0.6B`
- Prompt：Task1 Memory Agent 的 `voice` 风格
- Memory layout：`dedup`
- 输出：每个 interview 问题只生成一次，不做后处理

评估：

```text
评估文件：lab5/outputs/experiments/role_instruction_teacher_grpo_probe100_ckpt100_memory_dedup_voice_vs_persona/judge_metrics_instruction_teacher_grpo_probe100_ckpt100_memory_dedup_voice_vs_persona_500.json
judge 模型：Qwen3.5-35B-A3B
样本数：500
candidate wins / persona wins / tie = 371 / 128 / 1
candidate overall = 3.62 / 5 = 72.40 / 100
```

### 通用指令

推理：

- 生成模型：`checkpoint-100`
- 推理服务：`vllm serve`
- 不使用记忆检索
- System prompt：`You are a helpful assistant. Follow the user instruction exactly and answer directly.`
- `temperature=0.0`
- `top_p=1.0`
- `repetition_penalty=1.05`
- `max_tokens=1024`

评估：

```text
评估文件：lab5/outputs/experiments/instruction_teacher_grpo_probe100_ckpt100_dev_strict/eval/eval_results_strict.jsonl
样本数：100 个 prompt
约束数：157
通过约束数：110
strict instruction score = 110 / 157 = 70.06 / 100
```
