# HW2 Task1：Flowers102 CNN 微调

本目录用于完成 HW2 Task1：在 102 Category Flower Dataset 上微调 ImageNet 预训练 ResNet，完成超参数分析、预训练消融和注意力机制对比。

## 目录

```text
hw2/task1/
├── configs/                 # 每个 YAML 对应一组实验
├── flowers102_task1/         # 数据、模型、训练与指标代码
├── train.py                  # 训练入口
├── evaluate.py               # checkpoint 评估入口
├── requirements.txt
└── outputs/                  # 训练输出，默认不进入 Git
```

## 远程执行约定

本任务不在本机做 smoke test 或正式训练。代码在本机修改后通过 Git 同步到远程 `135-3090-8`，在远程直接运行正式实验。

远程使用前先检查：

```bash
git config user.name
git config user.email
git rev-parse HEAD
/data/yc/miniconda/envs/llm-26-gpu/bin/python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda", torch.cuda.is_available())
PY
```

Git 身份目标：

```text
YangChen-pro <1369792882@qq.com>
```

## 数据

默认数据目录：

```text
hw2/Flowers102/
├── jpg/
├── imagelabels.mat
├── setid.mat
└── README.txt
```

代码严格读取官方 `setid.mat` 划分：

- train：1020 张
- val：1020 张
- test：6149 张

## 实验配置

- `baseline_resnet18.yaml`：ImageNet 预训练 ResNet-18 Baseline。
- `baseline_resnet18_low_lr.yaml`：较低学习率组合，用于超参数对比。
- `baseline_resnet18_short.yaml`：较短训练轮数，用于 epoch 对比。
- `random_resnet18.yaml`：随机初始化 ResNet-18，做预训练消融。
- `se_resnet18.yaml`：SE-ResNet-18 注意力模型。
- `opt_resnet18_adamw_strong.yaml`：ResNet-18 + AdamW + 强增强 + label smoothing。
- `opt_resnet18_sgd_ls.yaml`：ResNet-18 + SGD + label smoothing。
- `opt_resnet18_sgd_ra_ls.yaml`：ResNet-18 + SGD + RandAugment + label smoothing。
- `opt_resnet34_sgd_ra_ls.yaml`：ResNet-34 + SGD + RandAugment + label smoothing。
- `opt_resnet34_adamw_strong.yaml`：ResNet-34 + AdamW + 强增强。
- `opt_resnet50_adamw_mild.yaml`：ResNet-50 + AdamW + 温和增强。
- `opt_resnet50_sgd_320_ra_ls.yaml`：ResNet-50 + 320 输入 + SGD + RandAugment。
- `opt_efficientnet_b0_strong.yaml`：EfficientNet-B0 + AdamW + 强增强。
- `opt_efficientnet_b0_320_ra_ls.yaml`：EfficientNet-B0 + 320 输入 + AdamW + RandAugment。
- `opt_convnext_tiny_strong.yaml`：ConvNeXt-Tiny + AdamW + 强增强。

## 训练命令

在远程仓库根目录执行：

```bash
/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task1/train.py \
  --config hw2/task1/configs/baseline_resnet18.yaml \
  --device auto
```

其他实验只需要替换 `--config`：

```bash
/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task1/train.py \
  --config hw2/task1/configs/random_resnet18.yaml \
  --device auto

/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task1/train.py \
  --config hw2/task1/configs/se_resnet18.yaml \
  --device auto
```

## SwanLab 记录

训练代码已支持 SwanLab。默认配置不强制开启，避免复现实验时误连云端；需要重新训练并实时记录时，在 YAML 中加入：

```yaml
logging:
  swanlab:
    enabled: true
    project: cs60003-hw2-task1
    mode: cloud
    group: task1-live
```

也可以不重跑训练，直接把已有正式实验的 `history.csv`、`metrics.json` 回放到 SwanLab，生成报告所需的 train / val loss、train / val accuracy 曲线：

```bash
/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task1/upload_swanlab_history.py \
  --all \
  --project cs60003-hw2-task1 \
  --group task1-history-replay
```

SwanLab API key 按用户要求记录在 `.helloagents/modules/hw2.md`；运行时也可优先使用 `SWANLAB_API_KEY` 环境变量。

已上传的正式实验回放链接见 `hw2/task1/SWANLAB_RUNS.md`。

## 输出

每次训练会在 `hw2/task1/outputs/{timestamp}_{experiment_name}/` 下保存：

- `source_config.yaml`：本次实验原始配置
- `config.json`：展开默认值后的配置
- `dataset_stats.json`：数据划分校验结果
- `history.csv`：每轮 train / val loss 与 accuracy
- `curves.png`：loss / accuracy 曲线
- `best.pt`：按验证集 accuracy 选择的最佳模型
- `metrics.json`：best epoch、best val accuracy、test loss、test accuracy
- `test_details.json`：测试集详细指标和混淆矩阵
- SwanLab 开启时，同步记录每轮 `train/loss`、`train/accuracy`、`val/loss`、`val/accuracy` 和最终 test 指标。

## 单独评估

```bash
/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task1/evaluate.py \
  --config hw2/task1/configs/baseline_resnet18.yaml \
  --checkpoint hw2/task1/outputs/<run_dir>/best.pt \
  --split test
```

## 报告素材整理

报告中至少记录：

- Baseline 的 train / val 曲线和 test accuracy。
- 不同学习率 / epoch 的对比表。
- 随机初始化与 ImageNet 预训练的 accuracy 差异。
- SE 注意力模型与 Baseline 的 accuracy 差异。
- 每组实验的配置、best epoch 和 checkpoint 路径。
- 最佳模型公网权重：ModelScope `youngchen/CS60003` 仓库下的 `hw2/task1/flowers102_convnext_tiny/best.pt`。

## 当前正式结果

远程正式实验已完成，结果详见 `hw2/task1/RESULTS.md`。

| 实验 | best val acc | test acc |
|---|---:|---:|
| ConvNeXt-Tiny 优化 | 0.9784 | 0.9608 |
| EfficientNet-B0 320 输入 | 0.9647 | 0.9439 |
| ResNet-50 AdamW | 0.9451 | 0.9257 |
| EfficientNet-B0 | 0.9480 | 0.9228 |
| ResNet-34 强增强 | 0.8284 | 0.8172 |
| Baseline ResNet-18 | 0.7186 | 0.6819 |
| 低学习率 ResNet-18 | 0.4873 | 0.4506 |
| 短训练 ResNet-18 | 0.5725 | 0.5307 |
| 随机初始化 ResNet-18 | 0.1794 | 0.1571 |
| SE-ResNet-18 | 0.4500 | 0.4245 |
