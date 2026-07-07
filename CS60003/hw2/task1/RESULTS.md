# HW2 Task1 正式实验结果

实验时间：2026-04-29
远程主机：`135-3090-8`
远程仓库：`/data/yc/CS60003`
原始 Baseline 提交：`91b6b8beadeb8fc7f5e8e021b1cdd65e718af43e`
优化实验提交：`c3d49cea3882ba92836a9a86020f583d77daea74`
Python 环境：`/data/yc/miniconda/envs/llm-26-gpu`
主要依赖：PyTorch `2.11.0+cu130`，torchvision `0.26.0+cu130`
GPU：`CUDA_VISIBLE_DEVICES=5`，NVIDIA GeForce RTX 3090

## 数据校验

`hw2/Flowers102/` 官方划分校验通过：

| Split | 样本数 |
|---|---:|
| train | 1020 |
| val | 1020 |
| test | 6149 |

## 实验结果

| 实验 | 配置 | best epoch | best val acc | test acc | 结论 |
|---|---|---:|---:|---:|---|
| ConvNeXt-Tiny 优化 | `opt_convnext_tiny_strong.yaml` | 36 | 0.9784 | 0.9608 | 当前最佳结果 |
| EfficientNet-B0 320 输入 | `opt_efficientnet_b0_320_ra_ls.yaml` | 46 | 0.9647 | 0.9439 | 高分辨率有收益，但低于 ConvNeXt |
| ResNet-50 AdamW | `opt_resnet50_adamw_mild.yaml` | 60 | 0.9451 | 0.9257 | 明显优于 ResNet-18 |
| EfficientNet-B0 | `opt_efficientnet_b0_strong.yaml` | 71 | 0.9480 | 0.9228 | 明显优于 ResNet-18 |
| ResNet-34 强增强 | `opt_resnet34_sgd_ra_ls.yaml` | 59 | 0.8284 | 0.8172 | ResNet 系列的稳健升级 |
| ResNet-18 label smoothing | `opt_resnet18_sgd_ls.yaml` | 60 | 0.7745 | 0.7689 | 低成本显著提升 |
| ResNet-50 320 输入 SGD | `opt_resnet50_sgd_320_ra_ls.yaml` | 66 | 0.7833 | 0.7421 | 高分辨率 + SGD 收敛较慢 |
| ResNet-18 强增强 | `opt_resnet18_sgd_ra_ls.yaml` | 55 | 0.7500 | 0.7416 | 强增强提升有限 |
| Baseline TTA 复评 | `baseline_resnet18.yaml` + horizontal flip TTA | 33 | 0.7186 | 0.6936 | 不重训的小幅提升 |
| Baseline | `baseline_resnet18.yaml` | 33 | 0.7186 | 0.6819 | 原始基线 |
| 低学习率 | `baseline_resnet18_low_lr.yaml` | 34 | 0.4873 | 0.4506 | 明显欠拟合 |
| 较短训练 | `baseline_resnet18_short.yaml` | 22 | 0.5725 | 0.5307 | 训练轮数不足 |
| 随机初始化 | `random_resnet18.yaml` | 40 | 0.1794 | 0.1571 | 预训练带来显著提升 |
| SE 注意力 | `se_resnet18.yaml` | 39 | 0.4500 | 0.4245 | 当前设置下不如 Baseline，需额外调参 |

相对原始 Baseline，当前最佳 ConvNeXt-Tiny 将 test accuracy 从 `0.6819` 提升到 `0.9608`，提升 `+27.89` 个百分点。

## 远程产物路径

```text
hw2/task1/outputs/20260429_031731_baseline_resnet18/
hw2/task1/outputs/20260429_032048_baseline_resnet18_low_lr/
hw2/task1/outputs/20260429_032344_baseline_resnet18_short/
hw2/task1/outputs/20260429_032540_random_resnet18/
hw2/task1/outputs/20260429_032833_se_resnet18/
hw2/task1/outputs/20260429_034317_opt_resnet18_sgd_ls/
hw2/task1/outputs/20260429_034317_opt_resnet18_sgd_ra_ls/
hw2/task1/outputs/20260429_034317_opt_resnet34_sgd_ra_ls/
hw2/task1/outputs/20260429_034831_opt_efficientnet_b0_strong/
hw2/task1/outputs/20260429_034831_opt_resnet50_adamw_mild/
hw2/task1/outputs/20260429_034831_opt_convnext_tiny_strong/
hw2/task1/outputs/20260429_035700_opt_efficientnet_b0_320_ra_ls/
hw2/task1/outputs/20260429_035700_opt_resnet50_sgd_320_ra_ls/
```

每个目录包含：

- `source_config.yaml`
- `config.json`
- `dataset_stats.json`
- `history.csv`
- `curves.png`
- `best.pt`
- `metrics.json`
- `test_details.json`

## SwanLab 记录

已补充 SwanLab 接入代码并在远程 `135-3090-8` 完成历史正式实验回放上传：

- 重新训练时，可在 YAML 中开启 `logging.swanlab.enabled: true`，实时记录 train / val loss、train / val accuracy 和最终 test 指标。
- 不重跑训练时，可运行 `hw2/task1/upload_swanlab_history.py --all`，把已有正式实验的 `history.csv` 和 `metrics.json` 回放到 SwanLab，用于报告截图。
- SwanLab 项目：<https://swanlab.cn/@youngchen/cs60003-hw2-task1>
- 已上传回放分组：`task1-report-curves`
- 每个 run 额外包含 `report/curves_with_axis_labels` 图像，横轴为 `Epoch`，纵轴为 `Loss` / `Accuracy`。
- 逐实验链接：`hw2/task1/SWANLAB_RUNS.md`

## ModelScope 模型权重

Task1 最佳 ConvNeXt-Tiny 权重和关键评估产物已上传到 ModelScope：

- 仓库：<https://modelscope.cn/models/youngchen/CS60003/>
- 最佳模型：`hw2/task1/flowers102_convnext_tiny/best.pt`
- 同目录产物：`source_config.yaml`、`config.json`、`metrics.json`、`test_details.json`、`history.csv`、`curves.png`、`swanlab_report_curves.png`

## 报告建议口径

- Baseline 使用 ImageNet 预训练 ResNet-18，分类头学习率 `1e-3`，backbone 学习率 `1e-4`，40 epochs。
- 最终推荐报告主结果使用 ConvNeXt-Tiny，配置为 AdamW、RandAugment、RandomErasing、label smoothing、horizontal flip TTA。
- 低学习率和短训练轮数均明显降低性能，说明 Task1 对充分微调强依赖。
- 随机初始化 ResNet-18 在相同训练预算下远低于预训练 Baseline，能直接支撑预训练消融结论。
- SE 注意力模块虽然增加了通道注意力，但原始版本新增参数随机初始化，在当前学习率和训练轮数下未超过 Baseline；报告中可解释为“当前实现与训练策略下未提升”，不要泛化为注意力机制无效。
- 错例分析显示原始 ResNet-18 主要混淆 `canterbury bells`、`camellia`、`japanese anemone`、`columbine`、`hibiscus`、`rose` 等细粒度类别；更强 ImageNet 预训练 backbone 是最有效提升方向。
