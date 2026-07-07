# CS60003 HW2

本目录是 CS60003 期中作业 HW2 的代码、数据配置与实验结果入口。作业包含三个任务：图像分类、车辆检测与跟踪、语义分割。

## 目录结构

```text
hw2/
├── hw2.md                  # 作业题面
├── Flowers102/             # Task1 数据
├── RoadVehicleImages/      # Task2 数据
├── StanfordBackground/     # Task3 数据
├── task1/                  # Flowers102 分类实验
├── task2/                  # Road Vehicle 检测、跟踪与越线计数
└── task3/                  # Stanford Background 语义分割实验
```

## 任务概览

| 任务 | 内容 | 入口 |
|---|---|---|
| Task1 | 微调 ImageNet 预训练 CNN 完成 Flowers102 分类，并比较超参数、预训练消融和注意力模块 | [`task1/README.md`](task1/README.md) |
| Task2 | 微调 YOLOv8s 完成 Road Vehicle 检测，并结合 ByteTrack 做视频多目标跟踪、遮挡分析和越线计数 | [`task2/README.md`](task2/README.md) |
| Task3 | 从零手写 U-Net 和 Dice Loss，在 Stanford Background 上比较 CE、Dice、CE+Dice，并做合规优化 | [`task3/README.md`](task3/README.md) |

## 报告与模型

- 正式报告工程：`/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report/`
- 最终 PDF：`/Users/yangchen/Documents/Latex_Project/CS60003_HW2_Report/out/hw2.pdf`
- 模型与关键实验产物统一发布到 ModelScope 仓库：<https://www.modelscope.cn/models/youngchen/CS60003/tree/master/hw2>

## 复现说明

- 每个任务都有独立的 `README.md`、配置文件和结果文档，复现时优先阅读对应任务目录。
- 大模型权重、训练输出、跟踪视频等大文件默认不进入 Git，按各任务 README 中的 ModelScope 路径或输出说明获取。
- SwanLab 仅用于实验记录和曲线导出，报告正文不公开私有实验页面。
