# Current unified checkpoint evidence

This directory keeps small evidence files for the current true single-model Lab5 Task5 submission.

## Model

- Checkpoint: `lab5/outputs/task4/megatron_instruction_teacher_grpo_rlvr_probe100/v0-20260531-201911/checkpoint-100`
- Base model: `Qwen2.5-1.5B-Instruct`
- Inference: vLLM serve only; no `AutoModelForCausalLM` inference.
- Submission package: `lab5/outputs/submission_task5_unified_ckpt100/task_5_test.zip`

## Files

- `stage1_tulu_manifest.json`, `stage1_tulu_samples.jsonl`: Tulu instruction SFT data metadata and examples.
- `stage2_rlvr_manifest.json`, `stage2_rlvr_samples.jsonl`: RLVR-IFeval GRPO data metadata and examples.
- `stage3_teacher_manifest.json`, `stage3_teacher_samples.jsonl`: Qwen3.5-35B-A3B teacher SFT metadata and examples.
- `role_generation_metrics.json`: 500-role-question generation latency/metadata for the unified checkpoint.
- `role_judge_metrics_500.json`: 500-role-question local LLM-as-a-Judge result, role score `72.40 / 100`.
- `instruction_dev_generation_metrics.json`: instruction dev generation metadata.
- `instruction_dev_eval_results_strict.jsonl`: IFEval strict evidence for instruction dev score `70.06 / 100`.
- `instruction_test_generation_metrics_441.json`: instruction test generation metadata for the 441 submitted records.
- `submission_manifest_unified_ckpt100.json`: final zip line-count and field validation.

Large full prediction files, details logs, training datasets, checkpoints, and model weights stay ignored.
