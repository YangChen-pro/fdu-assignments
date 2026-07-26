# Lab5 上传内容说明

## 当前最终提交包

当前真实统一模型提交文件：

```text
outputs/submission_task5_unified_ckpt100/task_5_test.zip
```

zip 内部只包含平台要求的两个 JSONL：

- `role_interview_predictions.jsonl`
  - 数量：500 条
  - 格式：`{"id": ..., "character_id": ..., "prediction": ...}`
  - 来源：`outputs/experiments/role_instruction_teacher_grpo_probe100_ckpt100_memory_dedup_voice/predictions.jsonl`
  - 说明：同一个 `checkpoint-100` 通过 vLLM 生成，输入包含 Persona 与检索到的 Memory。
- `instruction_test_predictions.jsonl`
  - 数量：441 条
  - 格式：`{"key": ..., "prediction": ...}`
  - 来源：`outputs/experiments/instruction_teacher_grpo_probe100_ckpt100_test_strict/predictions.jsonl`
  - 说明：同一个 `checkpoint-100` 通过 vLLM 生成，不使用 Memory。

当前提交包对应的模型是：

```text
outputs/task4/megatron_instruction_teacher_grpo_rlvr_probe100/v0-20260531-201911/checkpoint-100
```

该 checkpoint 的本地自测结果记录在 `BEST_CHECKPOINT.md`：角色模拟 `72.40 / 100`，通用指令 dev strict `70.06 / 100`。通用指令 test 的 441 条没有官方标签，本地只能做格式和生成 sanity check，不能得到官方分数。

## 目录结构

- `BEST_CHECKPOINT.md`：当前自测最佳统一 checkpoint 的训练链路、数据隔离和评估口径。
- `README.md`：Lab5 原始任务说明。
- `task_scripts/`：主线任务脚本；`public_pack/` 只读未修改。
- `outputs/`：实验过程输出、评估结果、预测文件和提交包；默认被忽略，只精确放行报告证据和最终 zip。
- `outputs/report_artifacts/`：最终报告需要引用的小型证据文件，已纳入 git 管理。
- `outputs/report_artifacts/task4/current_unified_checkpoint/`：当前统一模型的数据来源、训练样本示例、评估和提交证据。
- `outputs/submission_task5_unified_ckpt100/task_5_test.zip`：当前最终提交包，已纳入 git 管理。

## 脚本功能

- `task0_memory.py`：实现轻量级 Memory 检索，按 semantic similarity、lexical overlap、importance 加权排序。
- `task1_memory_agent.py`：运行 Base / Persona / Persona+Memory 三种角色采访问答配置。
- `task1_judge.py`：对 Task1 结果做 LLM-as-a-Judge 对比评估，支持并发、实时写入和断点续跑。
- `task2_build_data.py`：从公开角色数据构造 SFT / GRPO 训练数据。
- `task2_generate_synthetic_interview.py`：构造更贴近访谈任务的合成训练数据，并记录严格评估隔离信息。
- `task2_grpo_reward.py`：GRPO 奖励函数，默认使用外部 judge 服务，异常后返回中性 reward。
- `task2_train_sft.sh`：Megatron-SWIFT 全参数 SFT 训练脚本，接入 SwanLab。
- `task2_train_rlhf.sh`：Megatron-SWIFT GRPO 训练脚本，接入 SwanLab。
- `task2_judge.py`：角色模拟结果的成对 LLM-as-a-Judge 评估脚本，支持断点续跑。
- `task3_build_memory_data.py`：构造带 retrieved memories 的 SFT 数据，用于训练更会利用 Memory 的模型。
- `task4_instruction_agent.py`：通用指令 dev/test 推理脚本。
- `task5_build_submission.py`：将角色预测和通用指令预测打包为 `task_5_test.zip`。
- `serve_vllm.sh`：统一的 vLLM 服务启动脚本，供 Task1/Task3/Task4 推理复用。
- `utils.py`：JSON/JSONL、Persona 加载、OpenAI-compatible 调用、judge prompt、实时写入等通用工具函数。

## 任务实行情况

- Task0：检索 demo 检查项通过。
- Task1：完成 Question-only / Persona-only / Persona+Memory 对比；Memory 配置在 judge 结果中显著优于 Persona-only。
  - 最终 Task1 主系统 Persona+Memory：500 条全量推理、并发 32；端到端平均 5.57 s/条，P50 5.71 s，P95 6.86 s，总 wall-clock 107.24 s。
  - Memory 检索平均 0.045 s/条，vLLM 生成平均 4.80 s/条；judge 只用于离线评分，不计入系统推理时延。
- Task2：比较原始 dialogue SFT/GRPO 和合成访谈 SFT/GRPO，合成访谈路线显著更接近最终采访评测分布。
- Task3：比较 Base/SFT 模型与 Memory 的组合；Memory 能补充角色事实，训练模型加 Memory 优于单独训练模型。
- Task4/5：最终采用一个统一 `Qwen2.5-1.5B-Instruct` 派生 checkpoint 同时生成角色和通用指令提交结果。

## Git 管理范围

- 不上传模型权重、完整训练集、完整实验输出、memory cache 或细节日志。
- git 管理 `outputs/report_artifacts/`、当前最终 zip `outputs/submission_task5_unified_ckpt100/task_5_test.zip` 及其 manifest。
- `public_pack/` 保持只读，未修改。
