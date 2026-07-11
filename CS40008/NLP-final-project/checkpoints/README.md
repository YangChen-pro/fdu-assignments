# Model Checkpoints

本目录用于放置本地训练得到的模型文件。大文件仍由 `.gitignore` 忽略。

Task A 默认模型路径：

```text
checkpoints/task_a_roberta/
```

RoBERTa 微调脚本会把 tokenizer、config 和模型权重保存到该目录；后续预测脚本也默认从这里加载。

Hugging Face 原始模型缓存默认放在当前项目内：

```text
checkpoints/hf_cache/
```

训练脚本会通过 `--cache-dir checkpoints/hf_cache` 把 `roberta-base` 等原始模型下载到这里。缓存内容和微调模型权重一样默认不提交。
