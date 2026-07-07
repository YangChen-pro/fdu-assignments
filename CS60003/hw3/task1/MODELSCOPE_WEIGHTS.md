# Task 1 ModelScope 权重

仓库：<https://www.modelscope.cn/models/youngchen/CS60003>

目录：

```text
hw3/task1/real_high_quality/
```

该目录包含 `task1_real_quality_v2` 的六个模型文件。清单于 2026-06-21
通过 ModelScope 公开文件接口核验，仓库提交为 `195b638`，旧版 checkpoint
已删除。

| 类型 | 路径 | 大小（字节） |
|---|---|---:|
| background 3DGS splat | `exports/background/splat/splat.ply` | 202,327,681 |
| object A 3DGS splat | `exports/object_a/splat/splat.ply` | 19,231,512 |
| background Nerfstudio checkpoint | `nerfstudio/background/background/splatfacto/2026-06-10_223702/nerfstudio_models/step-000029999.ckpt` | 631,279,954 |
| object A Nerfstudio checkpoint | `nerfstudio/object_a/object_a/splatfacto/2026-06-11_141848/nerfstudio_models/step-000029999.ckpt` | 66,368,530 |
| object B SDS checkpoint | `object_b_threestudio/object_b/sds@20260608-021639/ckpts/last.ckpt` | 151,453,724 |
| object C Zero123 checkpoint | `object_c_zero123/object_c/zero123@20260608-022916/ckpts/last.ckpt` | 151,547,428 |

总大小：1,222,208,829 字节。

视频、图片、日志、配置、COLMAP 中间文件和 mesh 导出不上传到模型仓库。

```bash
python hw3/task1/upload_modelscope.py \
  --run-dir hw3/task1/outputs/task1_real_quality_v2 \
  --remote-subdir real_high_quality \
  --replace-remote-subdir
```

替换模式需要 `git` 和 `git-lfs`，并使用同一个
`MODELSCOPE_API_TOKEN` 完成 Git 身份验证。
