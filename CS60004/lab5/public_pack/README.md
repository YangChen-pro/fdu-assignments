# 实验五公开包说明

本目录包含《实验课五：智能体个体模拟》的公开数据、起步代码和验证集评估脚本。实验任务说明以 PDF 为准，本文档主要解释数据文件结构、字段含义和脚本使用方式。

## 1. 目录结构

```text
data/
├── individual_simulation_data/
│   ├── characters/
│   ├── scene/
│   ├── dialogue/
│   └── interview/
├── role_interview_eval.jsonl
└── instruction_following/
    ├── dev.jsonl
    └── test.jsonl
code/
├── task0_memory.py
├── instruction_eval_dev.py
└── instruction_eval/
```

## 2. 角色个体模拟数据

角色个体模拟数据位于：

```text
data/individual_simulation_data/
```

数据覆盖 68 个角色。`characters/`、`scene/`、`dialogue/` 和 `interview/` 中的文件均按角色名对应。

### 2.1 `characters/`

该目录包含角色 Persona 文本文件。每个文件对应一个角色：

```text
wiki_{character_name}.txt
```

文件内容为纯文本，通常包含：

- 角色身份、年龄、性别和职业；
- 性格特征、说话风格和行为习惯；
- 兴趣、偏好、价值观和长期目标；
- 社会关系、背景经历和个人设定。

### 2.2 `scene/`

该目录包含角色相关场景。每个文件对应一个角色：

```text
generated_agent_scene_{character_name}.json
```

每个文件是 JSON list，list 中每个元素是一条场景记录：

```json
{
  "type": "Chat",
  "location": "...",
  "background": "...",
  "source": "...",
  "profile": "..."
}
```

字段说明：

- `type`：场景类型，例如 `Chat`。
- `location`：场景发生地点。
- `background`：场景背景描述，包括当前事件、环境或互动上下文。
- `source`：场景来源标识，可用于追踪或去重。
- `profile`：该场景对应角色的 Persona 摘要。

### 2.3 `dialogue/`

该目录包含多轮角色对话。每个文件对应一个角色：

```text
generated_agent_dialogue_{character_name}.json
```

每个文件是 JSON list，list 中每个元素是一段对话场景：

```json
{
  "setting": ["..."],
  "emotion": "...",
  "topic": ["..."],
  "dialogue": [
    {
      "role": "...",
      "action": "(speaking)",
      "content": "..."
    }
  ],
  "location": "...",
  "background": "...",
  "source": "..."
}
```

字段说明：

- `setting`：对话整体设定，通常描述参与者、环境和互动背景。
- `emotion`：当前场景中的情绪状态或情绪标签。
- `topic`：对话主题标签。
- `dialogue`：多轮对话内容。
- `location`：对话发生地点。
- `background`：对话背景描述。
- `source`：对话来源标识，可用于追踪或去重。

`dialogue` 中每一轮包含：

- `role`：该轮内容所属角色。
- `action`：动作类型，常见取值包括 `(speaking)` 和 `(thinking)`。
- `content`：该轮具体内容。

### 2.4 `interview/`

该目录包含缩减后的角色采访问题。每个文件对应一个角色：

```text
generated_agent_interview_{character_name}.json
```

每个文件是 JSON list，list 中每个元素是一条采访问题对象：

```json
{
  "question_id": "role_eval_000000",
  "question": "How did your multilingual skills help you in your career as a secret agent?"
}
```

字段说明：

- `question_id`：问题唯一标识，与 `data/role_interview_eval.jsonl` 中的 `id` 对应。
- `question`：采访问题文本，已去掉原始数字编号。

本公开包中，采访问题共 500 道，覆盖全部 68 个角色。每个角色保留 7-8 道问题。

## 3. 角色采访问答评测索引

统一评测索引文件为：

```text
data/role_interview_eval.jsonl
```

每行是一条 JSON 样本：

```json
{
  "id": "role_eval_000000",
  "character_id": "Alessandra Rossi",
  "question_index": 1,
  "question": "How did your multilingual skills help you in your career as a secret agent?"
}
```

字段说明：

- `id`：评测样本唯一标识。提交预测时必须原样保留。
- `character_id`：目标角色名，与角色数据文件名对应。
- `question_index`：该问题在原始采访问题列表中的位置，仅用于追踪问题来源。
- `question`：需要模型以该角色身份回答的采访问题。

建议角色模拟推理时以 `data/role_interview_eval.jsonl` 为主，而不是逐个扫描 `interview/` 文件。

角色模拟预测文件格式：

```json
{"id": "role_eval_000000", "character_id": "Alessandra Rossi", "prediction": "..."}
```

## 4. 通用指令能力数据

通用指令能力数据位于：

```text
data/instruction_following/
```

### 4.1 `dev.jsonl`

验证集，共 100 条。每行是一条 JSON 样本：

```json
{
  "key": 3156,
  "prompt": "...",
  "instruction_id_list": ["keywords:existence"],
  "kwargs": [{"keywords": ["trust", "brand", "customer", "law", "policy", "unusable"]}]
}
```

字段说明：

- `key`：样本唯一标识。
- `prompt`：需要模型完成的自然语言指令。
- `instruction_id_list`：该 prompt 中需要检查的指令约束列表。
- `kwargs`：与 `instruction_id_list` 一一对应的检查参数。

`dev.jsonl` 保留完整评估字段，可用于本地验证模型是否满足指令约束。

### 4.2 `test.jsonl`

测试集，共 441 条。每行只包含公开输入字段：

```json
{"key": 251, "prompt": "..."}
```

字段说明：

- `key`：样本唯一标识。提交预测时必须原样保留。
- `prompt`：需要模型完成的自然语言指令。

`test.jsonl` 不包含 `instruction_id_list` 和 `kwargs`。最终测试集分数以平台端评测为准。

通用指令预测文件格式：

```json
{"key": 251, "prediction": "..."}
```

## 5. 起步代码与本地检查

### 5.1 Task 0

补全：

```text
code/task0_memory.py
```

运行检查：

```bash
python code/task0_memory.py
```

若实现正确，检查项应全部显示 `PASS`。

### 5.2 通用指令验证集评估

`code/instruction_eval/` 中包含通用指令能力验证集的规则评估代码，用于检查模型是否满足格式、长度、关键词、大小写、标点等可规则化判断的指令约束。通常不需要直接调用该目录下的模块，推荐使用外层入口脚本 `code/instruction_eval_dev.py`。

准备预测文件，例如：

```text
predictions/instruction_dev_predictions.jsonl
```

每行格式：

```json
{"key": 3156, "prediction": "..."}
```

从公开包根目录运行：

```bash
python code/instruction_eval_dev.py \
  --dev data/instruction_following/dev.jsonl \
  --predictions predictions/instruction_dev_predictions.jsonl
```

脚本会输出 strict 和 loose 两种统计结果。验证集结果只用于本地调试，最终 leaderboard 分数以平台端评测为准。

## 6. 注意事项

- 所有生成、训练和推理必须遵守实验文档中的模型使用限制。
- 不得修改评测输入文件。
- 上传报告和脚本时不需要上传模型权重。
