# CALVIN simulator probe

## 结论

这次已经把官方 CALVIN simulator 的代码和关键运行依赖在 135 上探通到可导入状态，但还不能对当前 LeRobot 版 `splitD` 直接跑真实 Success Rate。

原因不是模型或 GPU，而是数据格式：官方 Success Rate 评估入口 `calvin_agent.evaluation.evaluate_policy.make_env(dataset_path)` 会读取 `${dataset_path}/validation/.hydra/merged_config.yaml` 以及原始 CALVIN validation 环境状态；当前作业使用的 `xiaoma26/calvin-lerobot` 数据是 LeRobot parquet/meta 结构，`splitD` 下没有 `validation/` 目录，也没有 `.hydra/merged_config.yaml`。

## 135 上已完成的探测

- 官方仓库已克隆到 `/data/yc/tools/calvin_sim/calvin`。
- 官方环境仓库已克隆到 `/data/yc/tools/calvin_sim/calvin_env`。
- 原仓库 README 的评估入口是 `calvin_models/calvin_agent/evaluation/evaluate_policy.py`。
- 在 `/data/yc/miniconda/envs/llm-26-gpu` 中已补装官方评估入口所需的轻量依赖：`gym`、`hydra-core`、`omegaconf`、`pybullet`、`numpy-quaternion`、`pytorch-lightning`、`termcolor` 等。
- `calvin_env.envs.play_table_env` 和 `calvin_agent.evaluation.evaluate_policy` 已能通过 `PYTHONPATH` 导入。
- `pyhash` 对 Python 3.12 不兼容，官方包安装失败；已在 `/data/yc/tools/calvin_sim/compat/pyhash.py` 放了 FNV-1 32-bit 兼容 stub，仅用于让官方评估代码导入。

## 阻塞证据

对当前数据目录做最小环境构造测试：

```text
make_env FAIL FileNotFoundError [Errno 2] No such file or directory: '/data/yc/CS60003/hw3/task2/data/calvin_lerobot/splitD/validation/.hydra/merged_config.yaml'
dataset_exists True validation_exists False
```

官方 `task_D_D.zip` 的 HTTP 头显示大小约 165 GiB：

```text
Content-Length: 177379436142
```

135 的 `/data` 当前还有约 423 GiB 可用空间，理论上可以下载 zip，但解压后可能接近磁盘安全边界；而且这会引入另一套与作业当前 LeRobot 数据并行的原始 CALVIN 数据，不适合作为“精简版”交付默认动作。

## 当前可交付口径

本次结果中的 `Action L1` 是离线 imitation-learning 指标，不等价于 simulator Success Rate。真实 Success Rate 需要原始 CALVIN `task_D_D` 或至少带 `validation/.hydra/merged_config.yaml` 的官方数据目录，再为 ACT 写官方 `CustomModel.step(obs, goal)` 适配器。
