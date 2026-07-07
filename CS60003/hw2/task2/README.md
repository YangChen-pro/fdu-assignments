# HW2 Task2：Road Vehicle 目标检测与多目标跟踪

本目录用于完成 HW2 Task2：在 Road Vehicle Images Dataset 上微调 YOLOv8s 检测器，并结合 ByteTrack 完成交通视频多目标跟踪、稳定 display ID 显示、遮挡片段分析和越线计数。

本作业按任务分工完成，Task2 的 SwanLab workspace 和数据 YAML 路径可能与其他任务略有差异。最终统一交付入口仍是本仓库与 ModelScope 仓库；复现实验时请按本 README 的目录与命令检查本机数据路径。

## 目录

```text
hw2/task2/
├── configs/                 # 检测训练与跟踪配置
├── road_yolo/               # 配置、SwanLab、曲线与 JSON 工具
├── tests/                   # 跟踪后处理逻辑单元测试
├── train.py                 # YOLOv8 训练入口
├── evaluate.py              # checkpoint 验证入口
├── track_video.py           # 视频跟踪与越线计数入口
├── upload_swanlab_history.py
└── outputs/                 # 本地训练/跟踪输出，默认不进入 Git
```

## 数据

默认数据目录：

```text
hw2/RoadVehicleImages/trafic_data/
├── train/images/   # 2704 张训练图像
├── train/labels/
├── valid/images/   # 300 张验证图像
├── valid/labels/
└── data_hw2.yaml
```

数据集为 YOLO 标注格式，共 21 个车辆/道路交通相关类别。`data_hw2.yaml` 可能保留实验机器上的绝对路径；如果换机器复现，需要先确认其中的 `path` 指向当前仓库下的 `hw2/RoadVehicleImages/trafic_data/`。

## 环境

根据当前机器 CUDA 版本安装 PyTorch，再安装 Task2 依赖：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r hw2/task2/requirements.txt
```

快速检查：

```bash
python - <<'PY'
import torch
import ultralytics
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('ultralytics', ultralytics.__version__)
PY
```

## 训练

在仓库根目录执行：

```bash
export SWANLAB_API_KEY=<your_key>
python hw2/task2/train.py \
  --config hw2/task2/configs/yolov8s_baseline.yaml \
  --device auto
```

每次训练会写入 `hw2/task2/outputs/{timestamp}_{name}/`：

- `source_config.yaml`、`config.json`、`env.json`
- `history.csv`、`curves.png`
- `best.pt`
- `metrics.json`：包含 `best_epoch`、`best_mAP50`、`best_mAP50_95`

最终模型与关键产物已发布到 ModelScope：

```text
https://modelscope.cn/models/youngchen/CS60003/tree/master/hw2/task2/yolov8s_baseline
```

## 当前正式结果

| 实验 | best mAP50 | best mAP50-95 | best epoch |
|---|---:|---:|---:|
| YOLOv8s Baseline | 0.5150 | 0.2854 | 38 |

检测结果和报告口径见 `RESULTS.md`；训练曲线截图已整理到 HW2 报告工程的 `pic/task2_yolov8s_curves.png`。

## 单独评估

```bash
python hw2/task2/evaluate.py \
  --checkpoint hw2/task2/outputs/<run_dir>/best.pt \
  --config hw2/task2/configs/yolov8s_baseline.yaml
```

## 视频跟踪与越线计数

```bash
python hw2/task2/track_video.py \
  --model hw2/task2/outputs/<run_dir>/best.pt \
  --video <path/to/video.mp4> \
  --output-dir hw2/task2/outputs/tracking_result \
  --line-x 0.5
```

输出内容：

- `tracked.mp4`：带检测框、类别名、display ID 和累计计数的标注视频
- `frames.json`：逐帧检测、跟踪 ID 和累计计数
- `summary.json`：`total_crossed`、`display_ids_crossed`、计数线与跟踪设置
- `snapshots/`：按间隔保存的关键帧截图，可用于遮挡与 ID 跳变分析

默认跟踪策略：

- `--conf 0.1` 保留低置信度检测，交给 ByteTrack 做时序关联。
- `configs/bytetrack_occlusion.yaml` 将 `track_buffer` 提高到 120 帧，增强短时遮挡后的轨迹保留。
- 新 raw track ID 会在类别族、预测位置和框尺寸兼容时接回最近 display ID。
- 短时消失的轨迹可通过最近运动趋势预测是否越线；使用 `--disable-missing-prediction` 可关闭该补计逻辑。

## 遮挡分析与报告素材

报告中选取 60 fps 测试视频 frame 950--953 的连续四帧作为失败案例：两辆相似二轮车在密集交汇时被错误映射到同一个 display ID，说明仅依赖几何位置、框尺寸和短时运动信息无法完全解决遮挡后的身份恢复问题。对应裁剪图已保存为报告素材 `pic/task2_occlusion_id_jump_crop.png`。

两个视频版本的越线计数结果已写入报告：30 fps 版本统计 19 次越线，60 fps 版本统计 25 次越线。当前仓库保留代码、配置、指标文档和报告截图作为轻量交付证据。

## SwanLab

SwanLab 项目名：`cs60003-hw2-task2`。记录或回放前设置 `SWANLAB_API_KEY`：

```bash
export SWANLAB_API_KEY=<your_key>
python hw2/task2/upload_swanlab_history.py \
  --all \
  --project cs60003-hw2-task2 \
  --group task2-history-replay
```

已上传的 run 见 `SWANLAB_RUNS.md`。不同任务的 SwanLab workspace 可能不同；最终报告中只引用导出的本地曲线图，不公开私有实验页面。
