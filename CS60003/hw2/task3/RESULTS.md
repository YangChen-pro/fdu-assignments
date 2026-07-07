# HW2 Task3 实验结果

## 任务说明

Task3 使用 Stanford Background Dataset 从零训练手写 U-Net，并比较三种损失函数配置在验证集上的 mIoU 表现。训练全部在远程 `135-3090-8` 的 `/data/yc/miniconda/envs/llm-26-gpu` 环境完成。

## 数据与评估口径

- 数据集：`hw2/StanfordBackground/iccv09Data/`
- 样本数：715 张图像 / 715 个 `.regions.txt` 语义标签
- 划分：固定 seed=42，train 572 / val 143
- 输入尺寸：`240x320`
- 类别数：8 类（sky、tree、road、grass、water、building、mountain、foreground object）
- Ignore：标签 `-1` 转为 `255`，不参与 loss、pixel accuracy、mIoU 和 per-class IoU 统计
- 选模：以验证集 mIoU 最优 epoch 保存 `best.pt`

## 正式结果

SwanLab 云端已于 2026-04-30 清理为 6 个报告必要 run；其余探索性实验的指标保留在本文档和远程产物中，云端 run 不再保留。保留清单见 `hw2/task3/SWANLAB_RUNS.md`。

| 实验 | 损失函数 | 最佳 epoch | Val mIoU | Val pixel acc | SwanLab |
|---|---|---:|---:|---:|---|
| `task3_unet_ce` | Cross-Entropy | 69 | 0.648151 | 0.833211 | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/gky8mykesv1tyjf6hquq9> |
| `task3_unet_dice` | 手写 Dice Loss | 63 | 0.648211 | 0.829996 | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/nawj01kfoerht6cp02hcn> |
| `task3_unet_ce_dice` | Cross-Entropy + 手写 Dice Loss | 52 | **0.648970** | **0.834739** | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/fbpzyv27p65mc0xiqksk3> |

结论：三组基础 loss 实验 mIoU 非常接近，`CE + Dice` 组合损失略优；继续优化后最终推荐 `task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`。

## Per-class IoU

| 类别 | CE | Dice | CE + Dice |
|---|---:|---:|---:|
| sky | 0.883785 | 0.880028 | **0.884562** |
| tree | 0.659176 | 0.663875 | **0.668288** |
| road | **0.811282** | 0.784136 | 0.806054 |
| grass | 0.723195 | 0.726316 | **0.747842** |
| water | 0.640778 | 0.607719 | **0.645330** |
| building | 0.702307 | 0.706188 | **0.712474** |
| mountain | 0.187689 | **0.253740** | 0.158662 |
| foreground_object | **0.576994** | 0.563686 | 0.568549 |

观察：组合损失在 sky、tree、grass、water、building 等主要类别上更稳定；Dice-only 对 mountain 这类稀有类别更有利，但整体 pixel accuracy 略低。

## 远程产物

```text
hw2/task3/outputs/20260429_085730_task3_unet_ce/
hw2/task3/outputs/20260429_085801_task3_unet_dice/
hw2/task3/outputs/20260429_085805_task3_unet_ce_dice/
```

每个目录均包含：`source_config.yaml`、`config.json`、`dataset_stats.json`、`env.json`、`history.csv`、`curves.png`、`best.pt`、`metrics.json`、`val_samples.png`、`palette_legend.png`。

## ModelScope

最佳模型和关键产物已上传到：

- 仓库：<https://modelscope.cn/models/youngchen/CS60003/>
- 最佳模型路径：`hw2/task3/unet_ce_dice/best.pt`
- 指标文件路径：`hw2/task3/unet_ce_dice/metrics.json`
- 曲线图路径：`hw2/task3/unet_ce_dice/curves.png`
- 验证样例路径：`hw2/task3/unet_ce_dice/val_samples.png`

## 继续优化结果

在原始三组 loss 对比完成后，继续在 `hw2.md` 允许范围内做额外冲分：仍然从零训练手写 U-Net，不使用预训练或现成分割网络；只调整 U-Net 容量、轻量正则、输入尺寸、class weighting 和 horizontal-flip TTA 评估。

| 实验 | 关键变化 | 最佳 epoch | Val mIoU | Val pixel acc | SwanLab |
|---|---|---:|---:|---:|---|
| `task3_unet_ce_dice_tta` | 原 CE+Dice 配置 + TTA | 69 | 0.655269 | 0.839074 | 已清理（指标保留） |
| `task3_unet_ce_dice_wide_tta` | base_channels=48 + 256x320 + dropout + TTA | 92 | 0.662884 | 0.841969 | 已清理（指标保留） |
| `task3_unet_ce_dice_wide_weighted_tta` | base_channels=48 + class-weighted CE+Dice + TTA | 83 | 0.660594 | 0.839734 | 已清理（指标保留） |
| `task3_unet_ce_dice_b64_tta` | base_channels=64 + 256x320 + dropout + TTA | 67 | **0.665089** | **0.842011** | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/9odmxzsyxzeo44c8joex7> |

优化后最佳结果从 `0.648970` 提升到 **`0.665089`**，提升 **+0.016118 mIoU**。

### 优化后最佳模型 per-class IoU

| 类别 | IoU |
|---|---:|
| sky | 0.893930 |
| tree | 0.680075 |
| road | 0.814219 |
| grass | 0.724855 |
| water | 0.683296 |
| building | 0.714319 |
| mountain | 0.213420 |
| foreground_object | 0.596594 |

### 优化后 ModelScope

- 最佳模型路径：`hw2/task3/unet_ce_dice_b64_tta/best.pt`
- 指标文件路径：`hw2/task3/unet_ce_dice_b64_tta/metrics.json`
- 曲线图路径：`hw2/task3/unet_ce_dice_b64_tta/curves.png`
- 验证样例路径：`hw2/task3/unet_ce_dice_b64_tta/val_samples.png`

## 进一步冲分结果：Attention U-Net + 几何增强 + 多尺度 TTA

在用户要求继续冲击 `0.7+` 后，继续在 `hw2.md` 合规边界内优化：仍从零训练手写 U-Net 家族模型，不使用 SAM、DeepLab、SegFormer、torchvision segmentation models 或任何预训练权重；固定原 train/val 划分，验证集只用于评估和选模。

### 结构优化实验

| 实验 | 关键变化 | 最佳 epoch | Val mIoU | Val pixel acc | SwanLab |
|---|---|---:|---:|---:|---|
| `task3_resunet_b64_tta` | 手写 ResUNet + CE+Dice + TTA | 43 | 0.639848 | 0.831607 | 已清理（指标保留） |
| `task3_attention_unet_b64_tta` | 手写 Attention U-Net + CE+Dice + TTA | 91 | 0.667801 | 0.843214 | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/6s3s3o8thd5p4q6p2ju12> |
| `task3_attention_resunet_b48_tta` | 手写 Attention ResUNet，base=48 | 85 | 0.645112 | 0.838861 | 已清理（指标保留） |
| `task3_attention_resunet_b64_tta` | 手写 Attention ResUNet，base=64 | 66 | 0.638071 | 0.838809 | 已清理（指标保留） |

结构优化结论：Attention U-Net 对当前数据最有效；ResUNet / Attention ResUNet 更容易过拟合，验证 mIoU 低于 plain Attention U-Net。

### 几何增强与多 seed 实验

| 实验 | 关键变化 | 最佳 epoch | Val mIoU | Val pixel acc | SwanLab |
|---|---|---:|---:|---:|---|
| `task3_attention_unet_b64_aug_tta` | random scale crop，seed=42 | 105 | 0.693841 | 0.858079 | 已清理（指标保留） |
| `task3_attention_unet_b64_aug_seed7_tta` | random scale crop，seed=7 | 113 | 0.695953 | 0.859910 | <https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/fw55rpcbgagnmqbcz0q90> |
| `task3_attention_unet_b64_aug_seed2026_tta` | random scale crop，seed=2026 | 118 | 0.697984 | 0.860638 | 已清理（指标保留） |
| `task3_attention_unet_b64_aug_seed99_tta` | random scale crop，seed=99 | 129 | 0.691606 | 0.858449 | 已清理（指标保留） |
| `task3_attention_unet_b64_aug_seed13_tta` | random scale crop，seed=13 | 104 | 0.689485 | 0.859299 | 已清理（指标保留） |
| `task3_attention_unet_b64_aug_seed21_tta` | random scale crop，seed=21 | 126 | 0.681764 | 0.854328 | 已清理（指标保留） |
| `task3_attention_unet_b64_aug_seed1234_tta` | random scale crop，seed=1234 | 90 | 0.682729 | 0.851476 | 已清理（指标保留） |
| `task3_attention_unet_b64_aug_seed3407_tta` | random scale crop，seed=3407 | 107 | 0.691328 | 0.858285 | 已清理（指标保留） |

### 最终推荐结果

最终选择 `task3_attention_unet_b64_aug_seed7_tta` 的 best epoch 113 checkpoint。继续做单模型优化时，补充了 EMA + mountain-aware sampling / rare-class crop 试验，并扫描多尺度 TTA 组合。EMA + mountain-aware 版本提升了 `mountain` IoU，但整体 mIoU 未超过原 checkpoint；最终采用同一个单模型 checkpoint，并把评估尺度调整为 `[0.6, 0.8, 1.0, 1.2, 1.4]`。

| 实验 | 评估口径 | Best epoch | Val mIoU | Val pixel acc |
|---|---|---:|---:|---:|
| `task3_attention_unet_b64_aug_seed7_ms_tta` | flip + multi-scale TTA `[0.7, 0.85, 1.0, 1.15, 1.3]` | 113 | 0.700608 | 0.864100 |
| `task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta` | flip + multi-scale TTA `[0.6, 0.8, 1.0, 1.2, 1.4]` | 113 | **0.701053** | **0.864931** |
| `task3_attention_unet_b64_aug_seed7_ema_mountain_tta` | EMA + mountain-aware sampling/crop + flip + multi-scale TTA `[0.6, 0.8, 1.0, 1.2, 1.4]` | 110 | 0.700604 | 0.860460 |

相对基础三组 loss 最佳 `task3_unet_ce_dice`，最终提升：`0.701053 - 0.648970 = +0.052083` mIoU。相对上一轮 U-Net b64 + TTA 最佳，提升：`+0.035964` mIoU。

### 最终最佳模型 per-class IoU

| 类别 | IoU |
|---|---:|
| sky | 0.904431 |
| tree | 0.711124 |
| road | 0.839109 |
| grass | 0.764386 |
| water | 0.738596 |
| building | 0.759705 |
| mountain | 0.249698 |
| foreground_object | 0.641374 |

### 最终 ModelScope

- 最佳模型路径：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/best.pt`
- 指标文件路径：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/metrics.json`
- 曲线图路径：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/curves.png`
- 验证样例路径：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/val_samples.png`

## 0.75 单模型冲分尝试

在用户明确要求继续冲击 `0.75` 后，新增了更激进但仍符合 `hw2.md` 的单模型实验：U-Net++、scSE、ASPP bridge、deep supervision、CE+Dice+Lovasz、更高分辨率和更大 batch。所有模型仍为从零训练，不使用预训练权重、外部数据、SAM、DeepLab、SegFormer 或现成分割网络；固定 train 572 / val 143。

这些实验均未超过当前最终最佳 `0.701053`，因此不替换 ModelScope 最终模型。

| 实验 | 关键变化 | 已跑 epoch | 最佳 epoch | Val mIoU | Val pixel acc | 结论 |
|---|---|---:|---:|---:|---:|---|
| `task3_unetpp_b64_384_lovasz_ds_tta` | U-Net++ + scSE + ASPP + deep supervision + CE+Dice+Lovasz，384x512 | 77 | 74 | 0.693158 | 0.853670 | 多尺度复评为 0.6953，低于最终最佳 |
| `task3_unetpp_b48_480_lovasz_ds_tta` | U-Net++ b48，高分辨率 480x640 | 61 | 54 | 0.689058 | 0.849398 | 低于最终最佳 |
| `task3_attention_unet_b64_384_aug_seed7_tta` | Attention U-Net，384x512，CE+Dice | 99 | 97 | 0.669259 | 0.846414 | 低于最终最佳 |
| `task3_attention_unet_b64_480_lovasz_ema_tta` | Attention U-Net，480x640，CE+Dice+Lovasz + EMA | 101 | 101 | 0.666050 | 0.843837 | 高分辨率和 Lovasz/EMA 反而降低整体 mIoU |
| `task3_unetpp_b64_320_ce_dice_ds_tta` | U-Net++ + deep supervision，CE+Dice | 48 | 48 | 0.666117 | 0.835744 | 低于最终最佳 |
| `task3_attention_unet_b96_320_aug_seed7_tta` | 更宽 Attention U-Net，base_channels=96 | 39 | 29 | 0.606960 | 0.818540 | 明显低于最终最佳 |
| `task3_attention_unet_b64_480_aug_seed7_tta` | Attention U-Net，480x640，CE+Dice | 38 | 38 | 0.601919 | 0.807661 | 明显低于最终最佳 |

结论：在当前 `hw2.md` 约束下，单模型继续堆高分辨率、宽度或 U-Net++ 结构没有带来接近 `0.75` 的收益；`0.75` 更可能需要预训练 backbone、外部数据、现成强分割模型或 ensemble，但这些不符合当前约束或用户选择。
