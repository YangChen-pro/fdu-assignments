# Lab5 report artifacts

This directory is intentionally tracked even though most of `lab5/outputs/` is ignored.
It keeps only small, stable evidence needed for the final report, local evaluation, and submission provenance.

## Included

- `summary/local_scores.json`: consolidated historical local judge/evaluator metrics.
- `summary/leaderboard_result.json`: platform result recorded for the earlier leaderboard upload.
- `summary/submission_manifest.json`: historical submission zip validation.
- `summary/submission_manifest_unified_ckpt100.json`: current true single-model submission zip validation.
- `task1/`: Task1 memory-vs-baseline judge metrics and latency metrics.
- `task2/`: Task2 dataset isolation metadata, report note, and judge metric JSON files.
- `task3/`: Task3 memory-aware data metadata and judge metric JSON files.
- `task4/`: compared instruction-following dev/test metric JSON files.
- `task4/current_unified_checkpoint/`: current best unified checkpoint evidence, including training-data manifests, samples, role judge metrics, instruction dev strict results, instruction test generation metrics, and final submission manifest.
- `weakness_analysis/`: qualitative analysis over model answers and failure cases.

## Current final package

The current true single-model package is tracked at:

```text
lab5/outputs/submission_task5_unified_ckpt100/task_5_test.zip
```

It contains exactly:

```text
role_interview_predictions.jsonl      500 records
instruction_test_predictions.jsonl    441 records
```

The current package uses one `Qwen2.5-1.5B-Instruct` derived checkpoint for both tasks. Role simulation uses Task1 memory retrieval as input context, while general instruction following uses no memory retrieval. No external model answers, multi-model routing, retry-and-rerank, or generation post-processing are used for this current package.

## Excluded

Large training data, raw details logs, memory caches, checkpoints, full unzipped predictions, and model weights remain ignored.
