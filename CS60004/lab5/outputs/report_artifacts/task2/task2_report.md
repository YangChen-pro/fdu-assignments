# Lab5 Task2 结果摘要

## 设置

- 训练方法：Megatron LoRA SFT，全角色共享一个模型。
- 基座模型：Qwen2.5-1.5B-Instruct。
- 训练数据：从 individual_simulation_data/dialogue 中抽取目标角色 speaking 轮次，构造 messages 格式 SFT 数据。
- 数据规模：训练集 5042 条，验证集 265 条。
- 推理方式：只使用 vllm serve 启动 OpenAI-compatible 服务，task2_infer.py 通过 API 调用，不使用 AutoModelForCausalLM。
- 评估方式：LLM-as-a-Judge，比对 Task2 SFT 输出和 Task1 persona baseline，共 500 条。

## 产物

- SFT 数据：lab5/outputs/task2/data/sft_train.jsonl、lab5/outputs/task2/data/sft_val.jsonl
- 训练输出：lab5/outputs/task2/megatron_lora_sft_all/v1-20260527-125813/checkpoint-5042-merged
- 推理输出：lab5/outputs/task2/sft_all/predictions.jsonl
- 推理明细：lab5/outputs/task2/sft_all/details.jsonl
- 推理指标：lab5/outputs/task2/sft_all/metrics.json
- 评估结果：lab5/outputs/task2_judge/sft_vs_persona/judge_results_sft_vs_persona_500.jsonl
- 评估指标：lab5/outputs/task2_judge/sft_vs_persona/judge_metrics_sft_vs_persona_500.json

## Task2 问题回答

### 1. 训练方法是否提升了角色回答质量？

本次 LoRA SFT 相比 persona prompt-only baseline 没有带来整体提升。500 条 LLM-as-a-Judge 评估中，SFT 胜出 206 条，persona baseline 胜出 291 条，平局 3 条。平均 overall 分数为 SFT 2.988，persona baseline 3.128。

但 SFT 并非完全无效：它在 personality 上高于 persona baseline（2.974 vs 2.780），hallucination 维度也更好（3.612 vs 3.380）。说明 SFT 学到了一部分角色说话风格，并且生成更稳定；不足是 memorisation 和 values 较弱，分别为 2.270 vs 2.758、3.320 vs 3.614。原因是 SFT 推理时不再显式注入完整外部记忆，容易丢失 persona 中的具体事实和价值取向信息。

### 2. 不同方法或组合的比较

本次实际跑通并完整评估的是全角色共享 LoRA SFT。可对比的基线包括 Task1 的 persona prompt-only 和 memory-augmented 方法。

- persona prompt-only：无需训练，能直接使用完整 persona 信息；在本次 Task2 judge 中整体强于 SFT。
- LoRA SFT：不依赖外部 Memory，推理更快，角色语气和稳定性有提升，但事实记忆和价值观保持不足。
- Task1 memory-augmented：Task1 评估中 memory 胜出 persona 284/500，persona 胜出 203/500，memory_helped_count 为 465/500。说明显式检索记忆对角色回答仍然更有帮助，当前 SFT 尚未达到 memory-augmented 的效果。

结论：只做全角色共享 LoRA SFT 不足以替代 Memory；更合理的后续组合是 SFT 学习角色表达风格，再在推理时继续使用 Memory/RAG 补充事实和事件记忆。

### 3. 是否训练多个单角色模型？

本次没有训练 N 个单角色模型，而是训练一个全角色共享 LoRA SFT 模型。原因是当前目标先验证 Task2 的基本 SFT 流程和可评估闭环。基于现有结果，单角色模型可能会提升特定角色的语气一致性和事实记忆，但需要额外训练和评估，当前没有实验证据支撑定量结论。

## 时延

Task2 SFT 使用 vLLM serve，全量 500 条推理耗时 23.40 秒，并发 32，平均单条 0.856 秒，p50 为 0.795 秒，p95 为 1.602 秒。

Task2 judge 使用本机 8318 端口的 Qwen3.5-35B-A3B，500 条全部评估成功，平均 judge 调用时延 15.28 秒。
