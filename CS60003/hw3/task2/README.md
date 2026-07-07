# HW3 Task2：LeRobot ACT 跨环境泛化实验

本任务使用老师划分好的 CALVIN LeRobot 数据集完成 ACT 行为克隆实验，并比较单环境训练与多环境联合训练在未见过环境 D 上的 zero-shot 泛化能力。

## 运行环境

远程机器：`135-3090-8`

```bash
cd /data/yc/CS60003
/data/yc/miniconda/envs/llm-26-gpu/bin/python --version
```

关键依赖已收敛到 `hw3/task2/requirements.txt`，用于复现实验环境中的核心包版本；CALVIN simulator 只作为额外探测，不放进精简 requirements。

## 数据路径

```text
/data/yc/CS60003/hw3/task2/data/calvin_lerobot
├── splitA  # 环境 A
├── splitB  # 环境 B
├── splitC  # 环境 C
└── splitD  # 环境 D
```

每个 episode 是 LeRobot/parquet 格式，核心字段包括 `image`, `wrist_image`, `state`, `actions`, `timestamp`, `task_index`。

## 实验设置

两个模型使用相同 ACT 网络结构和核心超参数：

| 模型 | 训练数据 | 测试数据 | 主要指标 |
|---|---|---|---|
| `act_splitA` | `splitA` | `splitD` | Action L1 Error |
| `act_splitABC` | `splitA + splitB + splitC` | `splitD` | Action L1 Error |

当前实现直接调用 LeRobot 的 `ACTPolicy`：双视角图像编码、机器人状态编码、ACT transformer decoder 和 VAE 训练目标，一次预测未来 `chunk_size` 步动作。离线评估使用 `splitD` 的动作 L1 误差。已额外探测官方 CALVIN simulator；当前 LeRobot 数据不含官方 simulator 所需的 `validation/.hydra/merged_config.yaml`，因此本次不把真实 Success Rate 作为已完成指标。

## Dry-run

先确认数据读取、训练、评估、SwanLab 和 checkpoint 保存链路：

```bash
ssh 135-3090-8
cd /data/yc/CS60003
bash hw3/task2/scripts/dry_run.sh
```

## 正式训练

脚本会自动选择空闲 GPU，并用 `torchrun` 多 GPU 训练：

```bash
ssh 135-3090-8
cd /data/yc/CS60003
bash hw3/task2/scripts/train.sh hw3/task2/configs/act_splitA.yaml 8
bash hw3/task2/scripts/train.sh hw3/task2/configs/act_splitABC.yaml 8
```

输出目录：

```text
/data/yc/CS60003/hw3/task2/outputs/act_splitA
/data/yc/CS60003/hw3/task2/outputs/act_splitABC
```

每个实验会保存：

```text
config.yaml
metrics.csv
dataset_summary.json
train_summary.json
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/final.pt
```

## 评估

```bash
ssh 135-3090-8
cd /data/yc/CS60003
bash hw3/task2/scripts/evaluate.sh \
  hw3/task2/configs/act_splitA.yaml \
  hw3/task2/outputs/act_splitA/checkpoints/best.pt \
  act_splitA
bash hw3/task2/scripts/evaluate.sh \
  hw3/task2/configs/act_splitABC.yaml \
  hw3/task2/outputs/act_splitABC/checkpoints/best.pt \
  act_splitABC
```

评估结果：

```text
hw3/task2/outputs/eval/act_splitA_splitD.json
hw3/task2/outputs/eval/act_splitA_splitD.csv
hw3/task2/outputs/eval/act_splitABC_splitD.json
hw3/task2/outputs/eval/act_splitABC_splitD.csv
hw3/task2/outputs/eval/task2_results_table.csv
hw3/task2/outputs/eval/task2_results_table.json
```

当前正式结果（主口径优先看 `final.pt`，`best.pt` 只作为参考）：

| 模型 | 训练数据 | 测试数据 | best Action L1 | final-only Action L1 |
|---|---|---|---:|---:|
| `act_splitA` | `splitA` | `splitD` | 0.1731817282 | 0.1886211640 |
| `act_splitABC` | `splitA + splitB + splitC` | `splitD` | 0.1464656246 | 0.1549013643 |

结论：多环境联合训练的 `act_splitABC` 在未见过的 `splitD` 上动作误差更低。主结论优先采用不按 D 环境选择 checkpoint 的 `final.pt`：`act_splitABC` 将 Action L1 从 `0.1886211640` 降到 `0.1549013643`，相对提升 `17.88%`。`best.pt` 结果更强，但它使用了 D 上离线验证误差做 checkpoint 选择，因此只作为模型潜力参考。

可提交的小体积证据集中在 `hw3/task2/results/`：

- `task2_results_table.csv/json`：历史原始 best checkpoint 汇总。
- `best_checkpoint_eval_table.csv/json`：字段无歧义的 best checkpoint 汇总，避免把 best 评估误读为 final.pt。
- `final_only_eval_table.csv/json`：final checkpoint 汇总，作为最干净的 zero-shot 主口径。
- `STATISTICAL_SUMMARY.md`、`statistical_summary.csv/json`：episode、task、action-dim paired 统计与 bootstrap 置信区间。
- `*_task_breakdown.csv`、`*_episode_breakdown.csv`、`*_action_dim_breakdown.csv`：按任务、episode 和动作维度的误差分解。
- `curves/*_metrics_clean.csv`：从 SwanLab 拉取并清洗后的训练曲线源数据。
- `figures/*.png`：训练 loss、训练 Action L1、验证 Action L1 和模型对比图。
- `calvin_simulator_probe.md`：官方 CALVIN simulator 接入探测和阻塞证据。

## 复现检查

在 135 上可先跑前置检查，确认环境、数据、结果文件和 final.pt 主结论都在位：

```bash
ssh 135-3090-8
cd /data/yc/CS60003
bash hw3/task2/scripts/check_reproducibility.sh --strict-data --strict-env
```

该脚本会检查 Python 包版本、GPU 可见性、四个 split 的数据目录与 episode/frame 数、可提交结果文件和 `final.pt` 的 ABC 优势。普通本机没有数据或 GPU 时可以不加 `--strict-data --strict-env`，用于检查仓库里的小体积结果证据是否完整。

## SwanLab

训练脚本会从 `.helloagents/secrets/hw3.env` 加载 SwanLab key，并记录训练 loss、验证 Action L1、学习率、超参数和数据配置。不要在命令行或日志中打印 key。

正式记录：

- 项目：<https://swanlab.cn/@youngchen/CS60003_HW3_Task2>
- `act_splitA`：<https://swanlab.cn/@youngchen/CS60003_HW3_Task2/runs/05kubpls24j5jrp2wbl1t>
- `act_splitABC`：<https://swanlab.cn/@youngchen/CS60003_HW3_Task2/runs/4si6dcrut2krorrbalfkn>

## ModelScope 上传

训练完成后上传到课程统一 ModelScope 项目 `youngchen/CS60003` 的 `hw3/task2/` 目录，不再创建独立模型仓：

```bash
ssh 135-3090-8
cd /data/yc/CS60003
bash hw3/task2/scripts/upload_modelscope.sh \
  hw3/task2/outputs/act_splitA \
  youngchen/CS60003 \
  hw3/task2/outputs/act_splitA/modelscope_upload.json \
  hw3/task2
bash hw3/task2/scripts/upload_modelscope.sh \
  hw3/task2/outputs/act_splitABC \
  youngchen/CS60003 \
  hw3/task2/outputs/act_splitABC/modelscope_upload.json \
  hw3/task2
```

已上传位置：<https://modelscope.cn/models/youngchen/CS60003>

```text
hw3/task2/
├── act_splitA/
│   ├── checkpoints/best.pt
│   ├── checkpoints/final.pt
│   ├── checkpoints/latest.pt
│   ├── config.yaml
│   ├── dataset_summary.json
│   ├── metrics.csv
│   ├── results_summary.json
│   └── train_summary.json
├── act_splitABC/
│   ├── checkpoints/best.pt
│   ├── checkpoints/final.pt
│   ├── checkpoints/latest.pt
│   ├── config.yaml
│   ├── dataset_summary.json
│   ├── metrics.csv
│   ├── results_summary.json
│   └── train_summary.json
└── eval/
    ├── task2_results_table.csv
    └── task2_results_table.json
```

其中 `eval/` 里的汇总结果表由评估脚本生成后单独上传；两个旧独立模型仓不是最终交付位置。最新 final-only 与统计汇总结论保存在 Git 的 `hw3/task2/results/`，包括 `final_only_eval_table.*`、`best_checkpoint_eval_table.*` 和 `STATISTICAL_SUMMARY.md`。

## 代码结构

```text
hw3/task2/configs/        # 实验配置
hw3/task2/scripts/        # 训练、评估、曲线导出、结果整理、上传启动脚本
hw3/task2/src/hw3_task2/  # 数据、模型、训练、评估、结果整理、上传实现
hw3/task2/results/        # 可提交的小体积结果证据
hw3/task2/outputs/        # 运行输出，不提交 Git
```
