# HW3 Task1 真实高质量输入目录

把真实素材放到以下位置后，`hw3/task1/configs/real_high_quality.yaml` 可以直接读取，不需要改代码。

```text
hw3/assets/
├── object_a_multiview/                 # 物体 A：真实环绕照片；也可在配置中改用 object_a_video
├── object_c_single/
│   └── object_c_single_front.png       # 物体 C：真实单图，按题目要求先获得纯净前景
└── background_scene/
    └── images/                         # 背景：开源 3D 数据集场景图片，如 Mip-NeRF 360 garden/bicycle/counter
```

输入要求按 `hw3/hw3.md` 执行：

- 物体 A：手机拍摄真实物体的环绕视频或多视角照片。
- 物体 C：手机拍摄真实物体单图，生成前手工或用大模型去背景。
- 背景：从开源 3D 数据集选择场景并用 3DGS 重建，不要求自行拍摄背景。

本目录只提供默认读取路径；图片数量、视频长度和具体数据集选择不在代码中额外设硬性阈值。

真实高质量链路入口：

```bash
python hw3/task1/train.py --config hw3/task1/configs/real_high_quality.yaml
```

默认 `real_chain.execution.mode=plan`，会先校验素材并生成实际运行脚本。确认 136 已安装 Nerfstudio、COLMAP、threestudio、Zero123、Blender 后，把 YAML 改成 `mode: run` 即可执行完整链路。

背景默认按题目示例使用 Mip-NeRF 360 `counter`。136 上可用以下脚本下载官方数据并把 `hw3/assets/background_scene/images` 链接到场景图片目录：

```bash
bash hw3/task1/scripts/prepare_mipnerf360_counter_136.sh
```
