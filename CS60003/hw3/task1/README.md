# HW3 Task 1: 3DGS 与 AIGC 多源资产融合

本目录实现三类 3D 资产的生成与统一渲染：

| 资产 | 输入 | 方法 | 融合表示 |
|---|---|---|---|
| A | 真实物体环绕视频或多视角照片 | COLMAP + Nerfstudio 3DGS | Gaussian splats |
| B | 文本 prompt | threestudio + SDS | 带纹理 mesh，渲染前采样为 surface splats |
| C | 单张真实物体照片 | 前景分割 + Zero123XL | 带纹理 mesh，渲染前采样为 surface splats |
| 背景 | Mip-NeRF 360 `counter` | Nerfstudio 3DGS | Gaussian splats |

最终渲染由 `scripts/render_fused_splats.py` 完成。A、背景以及由 B/C mesh
采样得到的 splats 进入同一个 `gsplat` rasterizer，共享相机、深度排序和
alpha compositing。

## 目录结构

```text
configs/                 实验配置
scripts/                 数据预处理、外部工具封装和融合渲染
task1_3dgs_aigc/         配置解析、脚本生成和几何处理
tests/                   几何、预处理、渲染配置和发布逻辑测试
train.py                 流程入口
evaluate.py              产物验证入口
upload_modelscope.py     权重上传入口
```

训练输出写入 `hw3/task1/outputs/`，外部仓库安装到
`hw3/task1/external/`；两个目录都不进入 Git。

## 环境

建议使用 Python 3.10、CUDA 11.8+ 和支持 CUDA 的 PyTorch：

```bash
python -m pip install -r hw3/task1/requirements.txt
bash hw3/task1/scripts/setup_environment.sh
```

`setup_environment.sh` 安装 Nerfstudio、gsplat、threestudio，并下载
Zero123XL checkpoint。COLMAP 和 FFmpeg 需要由系统包管理器安装。

## 数据准备

默认配置读取：

```text
hw3/milk_task1_1080.m4v
hw3/objectC.HEIC
hw3/assets/background_scene/images/
```

这些素材不随代码仓库发布。可以在
`configs/real_high_quality.yaml` 中替换 `object_a_video`、
`object_c_image` 和 `background_images`。

下载并链接 Mip-NeRF 360 `counter`：

```bash
bash hw3/task1/scripts/prepare_mipnerf360_counter.sh
```

数据目录可通过 `DATA_ROOT` 覆盖，下载代理可通过 `DOWNLOAD_PROXY` 设置。

## 运行

配置默认使用 `plan` 模式，只检查输入并生成各阶段脚本：

```bash
python hw3/task1/train.py \
  --config hw3/task1/configs/real_high_quality.yaml
```

确认输入和依赖后，将配置中的执行模式改为：

```yaml
real_chain:
  execution:
    mode: run
```

再次执行同一命令即可依次运行：

1. A 的 COLMAP、前景 mask 和 3DGS 训练；
2. 背景 3DGS 训练；
3. B 的 SDS 优化；
4. C 的 Zero123XL 优化；
5. mesh/splat 导出；
6. 统一 `gsplat` 渲染。

## 当前结果

当前 run 名为 `task1_real_quality_v2`：

```text
hw3/task1/outputs/task1_real_quality_v2/
```

主要设置与产物：

- A：30k 3DGS steps，最终融合使用 42,924 个 splats；
- B：15k SDS steps，融合时采样 260,000 个 surface splats；
- C：1200 Zero123 steps，融合时采样 120,000 个 surface splats；
- 背景：30k 3DGS steps，最终使用 494,805 个 splats；
- 视频：1920x1080、24 fps、144 帧；
- 相机：以 A/B/C 为中心的 `foreground_orbit`。

完整产物验证：

```bash
python hw3/task1/evaluate.py \
  --run-dir hw3/task1/outputs/task1_real_quality_v2 \
  --strict-real-outputs
```

已知限制是 A 的反光区域仍有少量高亮 splat，B 的颜色与 prompt 存在偏差，
C 的侧面和背面依赖生成先验。

## SwanLab 与模型权重

训练曲线记录在
[cs60003-hw3-task1](https://swanlab.cn/@youngchen/cs60003-hw3-task1/)。
模型权重存放在
[ModelScope: youngchen/CS60003](https://www.modelscope.cn/models/youngchen/CS60003)
的 `hw3/task1/real_high_quality/` 目录，文件清单见
[`MODELSCOPE_WEIGHTS.md`](MODELSCOPE_WEIGHTS.md)。
该目录已于 2026-06-21 更新并核验，仅保留当前 run 对应的六个模型文件。

检查上传文件：

```bash
python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_real_quality_v2 \
  --remote-subdir real_high_quality \
  --dry-run
```

覆盖该远端目录：

```bash
MODELSCOPE_API_TOKEN=... python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_real_quality_v2 \
  --remote-subdir real_high_quality \
  --replace-remote-subdir
```

替换模式需要 `git` 和 `git-lfs`。权重仍通过 ModelScope API 上传；旧文件通过
临时浅克隆中的 Git 提交删除，不会下载远端 LFS 权重内容。
