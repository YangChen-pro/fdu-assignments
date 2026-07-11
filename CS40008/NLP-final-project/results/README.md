# Results

本目录用于汇总两个方向的预测、指标、raw outputs 和图表。根目录只放说明文件与公共图表目录；具体任务结果放到任务子目录，避免根目录堆积实验文件。

当前结构：

```text
results/
  README.md
  figures/
  task_a/
    roberta/
      predictions.csv
      predictions_swap.csv
      metrics.csv
      order_metrics.csv
    vllm/
      predictions/
      metrics/
      raw_outputs/
```

预测文件统一字段：

```text
id,method,pred,gold,correct,notes
```

指标文件统一字段：

```text
method,metric,value,split,notes
```

报告中不能填写未实际运行的数字；未完成结果使用 `待补充`。
