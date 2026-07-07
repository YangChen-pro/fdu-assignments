# HW1：EuroSAT 三层 MLP 分类器

本目录包含 HW1 的代码与实验文件。作业要求是在不使用现成自动微分框架的前提下，手写一个三层 MLP 完成 EuroSAT 十分类任务。本实现使用 `CuPy` 作为数组后端，前向传播、交叉熵损失、反向传播、SGD、学习率衰减和 L2 正则均为手写实现。

## 目录结构

```text
hw1/
├── EuroSAT_RGB/              # 数据集
├── mlp_hw1/
│   ├── backend.py            # CuPy 后端封装
│   ├── config.py             # 训练与搜索配置
│   ├── data.py               # 数据加载、缓存、归一化、划分
│   ├── metrics.py            # 准确率与混淆矩阵
│   ├── model.py              # 手写三层 MLP 与反向传播
│   ├── trainer.py            # 训练、评估、超参数搜索
│   └── visualization.py      # 曲线、权重、错例可视化
├── train.py                  # 训练入口
├── evaluate.py               # 测试集评估入口
├── search.py                 # 超参数搜索入口
├── download_weights.py       # ModelScope 权重下载与 SHA256 校验
├── tests/
│   └── test_core.py          # 不依赖数据集的基础单元测试
└── requirements.txt
```

## 环境依赖

基础依赖写在 `requirements.txt` 中，其中包含已在远端 GPU 环境验证的 `cupy-cuda13x==14.0.1`，以及用于下载和校验 ModelScope 权重的 `modelscope`：

```bash
python -m pip install -r "hw1/requirements.txt"
```

训练应放在支持 `CuPy` 的 GPU 环境中进行。本仓库仅支持 `CuPy + GPU` 环境复现，不提供 `NumPy` 训练回退；本地无 GPU 时只建议运行单元测试，完整训练与评估在远端 GPU 环境执行。

远端验证环境：

- Python：`3.12.13`
- CuPy：`cupy-cuda13x==14.0.1`
- GPU：`NVIDIA GeForce RTX 3090`
- 代码目录：`/data/yc/CS60003`
- 已验证解释器：`/data/yc/miniconda/envs/llm-26-gpu/bin/python`

在 135 上建议显式使用该解释器，避免误用系统默认 Python：

```bash
cd "/data/yc/CS60003"
PY="/data/yc/miniconda/envs/llm-26-gpu/bin/python"
"$PY" -X utf8 -c "import cupy; print(cupy.__version__)"
```

如需执行最小规模验证，可运行：

```bash
"$PY" -X utf8 "hw1/train.py" --preset quick
"$PY" -X utf8 -m unittest discover -s "hw1/tests"
```

## 仓库与权重

- GitHub 仓库：[https://github.com/YangChen-pro/CS60003](https://github.com/YangChen-pro/CS60003)
- ModelScope 权重仓库：[https://modelscope.cn/models/youngchen/CS60003/](https://modelscope.cn/models/youngchen/CS60003/)

仓库默认不保留 `best_model.npz`，`final_p` 与 `final_o` 的权重单独存放在 ModelScope 的 `hw1/` 目录下。当前仓库中用于复核实验结论的摘要文件、训练曲线、混淆矩阵、权重图和错例图保留在以下路径：

- `hw1/outputs/runs/final_a/`
- `hw1/outputs/runs/final_k/`
- `hw1/outputs/runs/final_l/`
- `hw1/outputs/runs/final_n/`
- `hw1/outputs/runs/final_p/`
- `hw1/outputs/runs/final_o/`
- `hw1/outputs/runs/manifest.csv`
- `hw1/outputs/search/20260409_100323/search_config.json`
- `hw1/outputs/search/20260409_100536/search_config.json`
- `hw1/outputs/search/20260419_045438/search_config.json`
- `hw1/outputs/search/20260419_045438/results.csv`
- `hw1/outputs/search/20260419_045438/results.json`
- `hw1/outputs/search/20260419_045438/best_result.json`

其中 `final_a`、`final_p` 和 `final_o` 是报告正文直接分析的核心实验；`final_k`、`final_l`、`final_n` 是围绕同一组隐藏层宽度与优化参数补跑的邻域实验，用于补强选模证据。`manifest.csv` 现在按“历史实验台账”整理，新增了 `source_group` 与 `tracked_in_repo` 两列：前者区分实验来源，后者标明该条记录是否在当前仓库中保留了可直接点击核验的产物。`20260409_100323` 与 `20260409_100536` 下的两份 `search_config.json` 是历史搜索配置快照，只用于保留搜索来源记录。

当前代码版本下重新执行的 `24` 组 full search 结果保留在 `hw1/outputs/search/20260419_045438/`。其中 `results.csv` 与 `results.json` 记录每个 trial 的超参数和验证/测试准确率，`best_result.json` 记录该次抽样搜索中的最优 trial。该最优 trial 使用 `1280 -> 768` 的隐藏层宽度、`learning_rate=0.012`、`weight_decay=0.0002`、`lr_decay=0.01`、`grad_clip=3.0` 和 `relu`，验证集准确率为 68.37%。为闭合这一条证据链，仓库额外保留了对应的 `hw1/outputs/runs/trial_04/config.json`、`history.json`、`summary.json` 和 `confusion_matrix.json`；这组文件与 `best_result.json`、`results.csv` 中的 `trial=4` 完全对应。后续邻域补跑在同一主配置附近加入 dropout 与轮数调整，最终得到验证集更高的 `final_p`。

实验列表：

| 实验 | 训练入口 | 验证集准确率 | 测试集准确率 | 说明 |
| --- | --- | --- | --- | --- |
| `final_a` | `"$PY" -X utf8 "hw1/train.py" --preset final_a` | 68.49% | 66.69% | 不加 dropout 的对照组 |
| `final_p` | `"$PY" -X utf8 "hw1/train.py" --preset final_p` | 69.01% | 67.58% | 正式提交模型 |
| `final_o` | `"$PY" -X utf8 "hw1/train.py" --preset final_o` | 68.77% | 68.10% | 扩展实验，测试集准确率高于 `final_p`，验证集准确率低于 `final_p` |

此外，仓库还保留了 3 组邻域补跑产物，用于说明正式模型并不是只在 `final_a / final_p / final_o` 这 3 个点之间做选择：

| 实验 | 验证集准确率 | 测试集准确率 | 说明 |
| --- | --- | --- | --- |
| `final_k` | 68.54% | 67.75% | `dropout=0.10`，`epochs=40` |
| `final_l` | 68.86% | 67.14% | `dropout=0.15`，`epochs=40` |
| `final_n` | 68.44% | 66.84% | `dropout=0.12`，`epochs=42` |

`final_a`、`final_k`、`final_l`、`final_n`、`final_p`、`final_o` 现在都保留了 `config.json` 与 `summary.json`。如需复核配置，既可以直接运行对应 preset，也可以对照对应目录下的配置和摘要文件。

## 运行方式

下面的命令默认在仓库根目录执行。正式训练、搜索和评估默认使用上一节定义的远端解释器变量 `$PY`；如只在本机运行单元测试，可使用文末给出的 `conda run -n nlp ...` 命令。

### 1. 快速检查

```bash
"$PY" -X utf8 "hw1/train.py" --preset quick
```

`quick` 预设使用 `limit_per_class=120` 并训练 `2` 个 epoch，用于验证训练、评估和输出流程是否可正常执行。

### 2. 训练正式模型

```bash
"$PY" -X utf8 "hw1/train.py" --preset best
```

如需运行默认训练配置，可执行：

```bash
"$PY" -X utf8 "hw1/train.py" --preset default
```

如需复现报告中列出的 3 组实验，请保持对应 preset、默认数据划分比例以及随机种子 `42` 不变：

```bash
"$PY" -X utf8 "hw1/train.py" --preset final_a
"$PY" -X utf8 "hw1/train.py" --preset final_p
"$PY" -X utf8 "hw1/train.py" --preset final_o
```

这 3 条命令会分别将结果输出到：

- `hw1/outputs/runs/final_a/`
- `hw1/outputs/runs/final_p/`
- `hw1/outputs/runs/final_o/`

其中 `best` 预设和 `final_p` 使用同一组训练配置。

如需复核报告中用于补强选模证据的邻域实验，也可以直接运行：

```bash
"$PY" -X utf8 "hw1/train.py" --preset final_k
"$PY" -X utf8 "hw1/train.py" --preset final_l
"$PY" -X utf8 "hw1/train.py" --preset final_n
```

### 3. 超参数搜索

```bash
"$PY" -X utf8 "hw1/search.py" --preset quick --max-trials 4
```

如需在完整数据上重新执行一次 24 组组合的抽样网格搜索，可运行：

```bash
"$PY" -X utf8 "hw1/search.py" --preset full --max-trials 24
```

`full` 表示在完整数据上执行一次包含 `24` 组组合的抽样网格搜索，而不是穷举全部组合；抽样过程使用固定随机种子，并优先覆盖学习率、两层隐藏层宽度和权重衰减这几类核心超参数。

运行后会在 `hw1/outputs/search/...` 下保存搜索结果。当前仓库保留了 `20260419_045438` 这次完整 `24` trial 搜索的 `results.csv`、`results.json` 和 `best_result.json`，同时保留 `20260409_100323` 与 `20260409_100536` 两份历史 `search_config.json` 配置快照。当前 full search 的主证据应以这三个结果文件为准；为便于直接核验最优 trial，仓库另外保留了 `hw1/outputs/runs/trial_04/` 下的 4 个 JSON 文件。`hw1/outputs/runs/manifest.csv` 只作为历史实验台账使用：其中 `source_group=full_search_20260419_045438` 的 `trial_04` 与本次 full search 一一对应，`archive_trial_*` 则是早期搜索留档别名，不能与当前 full search 的同名 trial 直接视为同一个 run。

### 4. 下载并校验已上传权重

```bash
"$PY" -X utf8 "hw1/download_weights.py" "hw1/model_weights"
```

该命令只下载 ModelScope 仓库中的两个权重文件，并自动校验 SHA256。ModelScope 仓库中的固定文件路径和下载后的本地路径如下：

| ModelScope 路径 | 下载后本地路径 | SHA256 |
| --- | --- | --- |
| `hw1/final_p/best_model.npz` | `hw1/model_weights/hw1/final_p/best_model.npz` | `1d33521419a060a4670b86be58926522c5febffc5a64a1d63ec2d30793325d2a` |
| `hw1/final_o/best_model.npz` | `hw1/model_weights/hw1/final_o/best_model.npz` | `e1295599ece6be23a20016c110dbca63359ebc82cb715d7db4a154b30b2457f5` |

### 5. 评估已上传模型

```bash
"$PY" -X utf8 "hw1/evaluate.py" --preset best --checkpoint "hw1/model_weights/hw1/final_p/best_model.npz"
```

将下载后的权重路径传给 `--checkpoint`。在 ModelScope 仓库中，正式提交模型对应 `hw1/final_p/best_model.npz`，扩展实验对应 `hw1/final_o/best_model.npz`。

若使用扩展实验权重，则将 `--preset` 设置为 `final_o`。

## 简要说明

- 模型结构是 `input -> hidden -> hidden -> output`
- 支持 `relu`、`tanh`、`sigmoid` 三种激活函数
- 训练时实现了 SGD、交叉熵损失、学习率衰减和 L2 正则
- 数据按类别分层划分为 train / val / test，并使用训练集统计量做归一化
- 数据缓存文件名会把 `seed`、`val_ratio`、`test_ratio`、`limit_per_class` 一起编码进去，避免在划分比例变化后复用不匹配的旧缓存
- 验证集准确率最高时会自动保存权重
- 训练后会在 `hw1/outputs/runs/<run_name>/` 下生成本次实验结果

## 测试

不依赖真实数据集的基础回归测试：

```bash
conda run -n nlp python -X utf8 -m unittest discover -s "hw1/tests"
```

## 备注

- 本实现只使用 `CuPy`，不保留 `NumPy` 训练回退
- 所有 HW1 相关文件位于 `hw1/` 目录下
- 报告以 PDF 形式提交
- `summary.json` 里如果出现绝对路径，表示训练时的运行路径记录
