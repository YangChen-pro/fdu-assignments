# HW2 Task3：Stanford Background U-Net 语义分割

本目录用于完成 HW2 Task3：从零手写 U-Net，在 Stanford Background Dataset 上比较 Cross-Entropy、Dice Loss、Cross-Entropy + Dice 三种损失配置的验证集 mIoU。

## 数据与标签

- 数据集：`hw2/StanfordBackground/iccv09Data/`
- 图像：`images/*.jpg`
- 语义标签：`labels/*.regions.txt`
- 类别：sky、tree、road、grass、water、building、mountain、foreground object
- `-1` 标签表示 unknown，训练 loss 和所有指标中均忽略。

## 实现要求

- `stanford_unet/models.py` 手写经典 U-Net，包含 encoder、decoder、skip connection。
- `stanford_unet/losses.py` 手写 Dice Loss。
- 不使用任何预训练权重或现成分割网络。
- 训练输出默认写入 `hw2/task3/outputs/`，该目录不进入 Git。

## 远程环境

正式实验使用远程 135：

```bash
ssh 135-3090-8
cd /data/yc/CS60003
/data/yc/miniconda/envs/llm-26-gpu/bin/python -m pip install -r hw2/task3/requirements.txt
```

## 训练命令

```bash
/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task3/train.py --config hw2/task3/configs/ce.yaml --device cuda
/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task3/train.py --config hw2/task3/configs/dice.yaml --device cuda
/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task3/train.py --config hw2/task3/configs/ce_dice.yaml --device cuda
```

每个 run 产物包括：

- `source_config.yaml`
- `config.json`
- `dataset_stats.json`
- `env.json`
- `history.csv`
- `curves.png`
- `best.pt`
- `metrics.json`
- `val_samples.png`
- `palette_legend.png`

## 评估命令

```bash
/data/yc/miniconda/envs/llm-26-gpu/bin/python hw2/task3/evaluate.py --checkpoint hw2/task3/outputs/<run>/best.pt --device cuda
```

## 实验记录

SwanLab 项目：`cs60003-hw2-task3`。训练脚本会记录训练/验证 loss、验证 mIoU、pixel accuracy、per-class IoU，并上传带明确横轴/纵轴标注的报告曲线。云端已清理为报告必要 run，保留清单见 `SWANLAB_RUNS.md`。

## 结果

正式实验结果见 `RESULTS.md`。最佳权重后续发布到 ModelScope：`https://modelscope.cn/models/youngchen/CS60003/` 的 `hw2/task3/` 子目录。

## 当前正式结果

三组远程正式实验已完成，结果见 `RESULTS.md`。

| 实验 | Val mIoU | Val pixel acc |
|---|---:|---:|
| CE | 0.648151 | 0.833211 |
| Dice | 0.648211 | 0.829996 |
| CE + Dice | **0.648970** | **0.834739** |

最佳模型：`task3_unet_ce_dice`。ModelScope 路径：`hw2/task3/unet_ce_dice/best.pt`。


## 优化后最佳结果

在 `hw2.md` 允许范围内继续优化后，当前最佳模型为 `task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`：

- validation mIoU：`0.701053`
- validation pixel accuracy：`0.864931`
- best epoch：113
- 训练结构：从零手写 Attention U-Net，未使用预训练或现成分割网络
- 训练增强：horizontal flip、color jitter、random scale crop
- 评估增强：horizontal-flip TTA + multi-scale TTA `[0.6, 0.8, 1.0, 1.2, 1.4]`
- SwanLab：<https://swanlab.cn/@youngchen/cs60003-hw2-task3/runs/fw55rpcbgagnmqbcz0q90>
- ModelScope：`hw2/task3/attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta/best.pt`

## 0.75 冲分尝试结论

后续尝试了更激进的单模型路线，包括 U-Net++、scSE、ASPP、deep supervision、CE+Dice+Lovasz、更高分辨率和更大 batch。最好结果为 `task3_unetpp_b64_384_lovasz_ds_tta`，多尺度复评 `val_mIoU=0.6953`，低于当前最终最佳 `0.701053`。

因此最终模型仍保持 `task3_attention_unet_b64_aug_seed7_ms060_080_100_120_140_tta`。
