# Data

本目录用于放置 SemEval-2020 Task 4 ComVE 数据。请不要提交原始数据文件或处理后的大文件。

建议结构：

```text
data/
  raw/
    train.csv
    dev.csv
    test.csv
  processed/
    comve_task_a_train.jsonl
    comve_task_a_dev.jsonl
    comve_task_a_test.jsonl
```

统一字段：

```text
id,sent0,sent1,gold
```

其中 `gold=0` 表示 `sent0` 违背常识，`gold=1` 表示 `sent1` 违背常识。
