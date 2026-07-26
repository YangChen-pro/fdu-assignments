# 实验一：SFT课程实验——CS60004 人工智能研究生课程

## 1. 实验概述

本实验围绕监督微调（Supervised Fine-Tuning, SFT）展开，目标是让大家完整经历一次从数据构造、模型训练、评测分析到结果汇报的实验流程。基础模型统一使用 **Qwen2.5-1.5B-Instruct**，如需使用 teacher model 进行数据构造或辅助标注，统一使用 **Qwen3.5-9B**。所有训练实验须在**单张 32GB 显存 GPU**上完成。在线平台最多上传 **5 次**结果。

本次实验分为两个层次：

- **主线任务**：围绕一个专业领域单项选择题任务完成 Task 1–4，并形成最终实验报告。
- **附加任务**：在课程实验平台上尝试逻辑推理题 leaderboard，作为 Task 5，鼓励大家基于前面的尝试继续深挖。这个部分只提交平台结果，不作为主报告的主体。

## 2. 论文阅读与分享

需要阅读与 SFT 相关的经典论文，课堂分享内容建议包括：

- 论文试图解决什么问题；
- 核心方法是什么；
- 关键实验结论是什么；
- 这篇工作对本次课程实验有什么启发。

### 小组分享 1：指令微调范式的建立与数据自举

#### Finetuned Language Models Are Zero-Shot Learners (FLAN)
- 作者：Jason Wei et al. (Google Research)
- 发表：ICLR 2022
- arXiv：2109.01652
- 核心主题：指令微调奠基工作

#### Self-Instruct: Aligning Language Models with Self-Generated Instructions
- 作者：Yizhong Wang et al. (University of Washington / Allen Institute for AI / University of Washington NLP)
- 发表：ACL 2023
- arXiv：2212.1056
- 核心主题：用模型自生成 instruction 数据做指令对齐

### 小组分享 2：参数高效微调：从低秩适配到量化微调

#### LoRA: Low-Rank Adaptation of Large Language Models
- 作者：Edward J. Hu et al. (Microsoft)
- 发表：ICLR 2022
- arXiv：2106.09685
- 核心主题：参数高效微调经典方法

#### QLoRA: Efficient Finetuning of Quantized LLMs
- 作者：Tim Dettmers et al. (University of Washington)
- 发表：NeurIPS 2023
- arXiv：2305.14314
- 核心主题：量化 + LoRA，实现低显存高效微调

### 小组分享 3：reasoning-enhanced SFT：从拒绝采样到自举推理

#### Scaling Relationship on Learning Mathematical Reasoning with Large Language Models (RFT)
- 作者：Zheng Yuan et al. (Alibaba Group)
- 发表：arXiv 2023
- arXiv：2308.01825
- 核心主题：拒绝采样微调与 reasoning 数据增强

#### STaR: Bootstrapping Reasoning With Reasoning
- 作者：Eric Zelikman et al. (Stanford University)
- 发表：NeurIPS 2022
- arXiv：2203.14465
- 核心主题：reasoning bootstrapping / 自举式推理训练

### 小组分享 4：领域对齐与高质量监督数

#### DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation
- 作者：Zhijie Bao et al. (Fudan University 等)
- 发表：arXiv 2023
- arXiv：2308.14346
- 核心主题：通用 SFT 方法在医疗问诊场景中的落地

#### LIMA: Less Is More for Alignment
- 作者：Chunting Zhou et al. (Meta AI / University of Washington 等)
- 发表：NeurIPS 2023
- arXiv：2305.11206
- 核心主题：少量高质量 SFT 数据也能实现强对齐

## 3. 实验数据任务说明

本次主线实验围绕一个**医疗领域单项选择题问答任务**展开。每条样本包含题干、选项和标准答案。你们需要将原始样本**训练集 train.jsonl** 加工为适合 SFT 的结构化训练数据，并训练模型输出标准化结果。

### 核心点

- 这是一个医疗专业领域单项选择题任务；
- 每道题都有明确标准答案；
- 任务目标是比较不同 SFT 数据设计与训练方法的效果。

在这个主线任务中，大家需要重点关注：

- 什么样的数据监督格式更适合小模型；
- 全参数微调和参数高效微调的差异；
- 带 reasoning 的监督是否一定优于只监督最终答案；
- 准确率、输出格式稳定性与训练开销之间的关系。

## 4. Task 1：SFT 数据构造

围绕主线任务，需要至少设计并比较以下几类数据构造方案。下面以一条专业领域单项选择题样本为例说明。

### 原始样本示例

```json
{"question": "患者，男性，15岁，自幼有出血倾向。实验室检查:出血时间延长，凝血时间正常，PLT200×109/L，血小板黏附聚集功能障碍。其父也有类似病史。最可能的诊断为", "option": {"A": "血友病", "B": "血小板减少性紫癜", "C": "遗传性出血性毛细血管扩张症", "D": "血管性假血友病", "E": "维生素K缺乏"}, "answer": "D"}
````

在构造 SFT 数据时，可以统一写成 Alpaca 格式：

```json
{
  "instruction": "...",
  "input": "",
  "output": "..."
}
```

其中 `instruction` 负责描述题目和输出要求，`output` 则对应不同方案下的目标输出文本。

### Plan A：直接答案监督

只监督最终答案，不包含推理过程，作为基线方案。

#### 示例

```json
{
  "instruction": "请回答下面的单项选择题，并以 JSON 格式输出最终答案。\n\n题目类型：专业知识考试-基础医学-病理生理学\n题目：患者，男性，15岁，自幼有出血倾向。实验室检查:出血时间延长，凝血时间正常，PLT200×109/L，血小板黏附聚集功能障碍。其父也有类似病史。最可能的诊断为\n选项：\nA. 血友病\nB. 血小板减少性紫癜\nC. 遗传性出血性毛细血管扩张症\nD. 血管性假血友病\nE. 维生素K缺乏\n\n输出要求：仅输出一个合法 JSON 对象，包含字段 answer。",
  "input": "",
  "output": "{\"answer\": \"D\"}"
}
```

这个方案的特点是目标最简单，训练信号集中在“从题目到正确选项”的映射上。

### Plan B：推理增强监督

借助 teacher model（Qwen3.5-9B）为样本生成推理过程，输出中同时包含 `reasoning` 和 `answer`。推理必须基于正确答案生成，且不能与标准答案矛盾。

#### 示例

```json
{
  "instruction": "请回答下面的单项选择题，并以 JSON 格式输出推理过程和最终答案。\n\n题目类型：专业知识考试-基础医学-病理生理学\n题目：患者，男性，15岁，自幼有出血倾向。实验室检查:出血时间延长，凝血时间正常，PLT200×109/L，血小板黏附聚集功能障碍。其父也有类似病史。最可能的诊断为\n选项：\nA. 血友病\nB. 血小板减少性紫癜\nC. 遗传性出血性毛细血管扩张症\nD. 血管性假血友病\nE. 维生素K缺乏\n\n输出要求：仅输出一个合法 JSON 对象，包含字段 reasoning 和 answer。",
  "input": "",
  "output": "{\"reasoning\": [\"患者自幼有出血倾向，且其父也有类似病史，提示可能为遗传性出血性疾病。\", \"实验室检查显示出血时间延长，但凝血时间正常，说明主要问题不在凝血因子缺乏。\", \"血小板计数正常，但血小板黏附聚集功能障碍，提示血小板功能异常而非数量减少。\", \"血管性假血友病的典型表现包括出血时间延长、血小板黏附功能异常以及家族遗传倾向。\"], \"answer\": \"D\"}"
}
```

这个方案的目标是让模型不仅输出答案，还学习一种较规范的分析路径。

### Plan C：知识增强监督

在 Plan B 基础上加入额外背景知识，输出中增加 `knowledge` 字段，再给出 `reasoning` 和 `answer`，用于探索更丰富 supervision 是否带来收益。

#### 示例

```json
{
  "instruction": "请回答下面的单项选择题，并以 JSON 格式输出相关知识、推理过程和最终答案。\n\n题目类型：专业知识考试-基础医学-病理生理学\n题目：患者，男性，15岁，自幼有出血倾向。实验室检查:出血时间延长，凝血时间正常，PLT200×109/L，血小板黏附聚集功能障碍。其父也有类似病史。最可能的诊断为\n选项：\nA. 血友病\nB. 血小板减少性紫癜\nC. 遗传性出血性毛细血管扩张症\nD. 血管性假血友病\nE. 维生素K缺乏\n\n输出要求：仅输出一个合法 JSON 对象，包含字段 knowledge、reasoning 和 answer。",
  "input": "",
  "output": "{\"knowledge\": [\"血管性假血友病（von Willebrand disease）是一种常见的遗传性出血性疾病。\", \"其主要机制是 von Willebrand 因子异常，导致血小板黏附功能受损，并可伴有出血时间延长。\", \"该病常表现为血小板计数正常，但血小板功能检查异常，可有家族史。\"], \"reasoning\": [\"患者自幼出血且有家族史，首先考虑遗传性出血性疾病。\", \"凝血时间正常，不支持典型凝血因子缺乏所致的血友病。\", \"血小板计数正常但黏附聚集功能障碍，说明问题在血小板功能。\", \"结合上述特征，最符合血管性假血友病。\"], \"answer\": \"D\"}"
}
```

这个方案相比 Plan B 多了一层显式知识注入，适合探索“小模型是否能从额外背景信息中获益”。

### 所有方案都应满足以下基本要求

* `output` 必须能够被 `json.loads()` 正确解析；
* `answer` 必须与标准答案一致；
* 同一方案中的字段结构应保持统一；
* 若使用 teacher model 生成 `reasoning` 或 `knowledge`，应进行答案一致性检查和必要的格式清洗。

## 5. Task 2：全参数微调

Task 2 的代码必须在**不使用任何 Trainer 类**的情况下完成训练。

基于 Task 1 构造的数据，对 **Qwen2.5-1.5B-Instruct** 进行全参数监督微调。

本任务的目标是让大家理解 SFT 的基本训练流程，因此需要自行实现核心训练逻辑，而不是直接调用封装好的训练器。

### 允许使用的工具

可以使用以下基础组件：

* PyTorch
* Transformers

  * 允许加载模型（`AutoModelForCausalLM`）
  * 允许加载 tokenizer（`AutoTokenizer`）
* datasets

  * 用于数据读取与预处理
* accelerate / torch.cuda

  * 用于设备管理
* 常规 Python 工具库

允许直接使用：

* tokenizer
* model forward
* optimizer（AdamW 等）
* scheduler

### 必须自己实现的部分

以下内容需要自行实现：

* `instruction / input / output` 的 label masking
* 数据预处理 pipeline
* data collator（动态 padding）
* 训练循环

训练循环至少包含：

* forward
* loss 计算
* backward
* optimizer step
* scheduler step
* gradient clipping
* logging
* validation
* checkpoint 保存

### 不允许使用的工具

以下工具**禁止使用**：

* `transformers.Trainer`
* `trl.SFTTrainer`
* 任何封装好的 SFT 训练框架
* 一键 finetune pipeline

这些工具会隐藏训练细节，不利于理解 SFT 的核心流程。

## 6. Task 3：参数高效微调（PEFT）

在 Task 2 的基础上，实现参数高效微调方法，并与全参数微调进行对比。

本任务的目标是理解**参数效率、显存占用与性能之间的权衡**。

### 实现方法

至少实现以下方法：

* LoRA
* QLoRA
* RFT（拒绝采样微调）

### 允许使用的工具

在 Task 3 中，可以使用成熟的 PEFT 工具：

* PEFT library

  * `LoraConfig`
  * `get_peft_model`
* bitsandbytes

  * 用于 4-bit / 8-bit 量化
* transformers
* datasets

在 Task 3 中：

* 可以使用 Trainer
* 可以使用 SFTTrainer
* 可以使用 PEFT 官方接口

因为本任务的重点不再是实现训练循环，而是比较不同方法。

### 需要报告的内容

对于每种方法，需要记录：

* Answer accuracy
* JSON parse rate
* GPU 显存占用
* 训练时间
* 可训练参数规模
* 推理速度

同时需要说明：

* LoRA 的 `r`
* `lora_alpha`
* `lora_dropout`
* `target modules`

以及 QLoRA 的量化设置。

## 7. Task 4：评测、对比分析与最终报告

完成前面任务后，需要对所有实验方案在**验证集 val.jsonl**上做统一评测，并提交最终报告。

主线任务至少应包含以下评测指标：

* **Answer Accuracy**：最终答案正确率；
* **JSON Parse Rate**：输出可解析 JSON 的比例；
* **Reasoning Completeness（可选）**：推理步骤是否覆盖足够信息，可以通过部署模型实行 LLM-as-a-Judge 方案实现。

### 最终报告需要包含

* 尝试了哪些数据构造方案；
* 实现了哪些训练方法；
* 各实验结果对比表格；
* 训练开销记录；
* 对结果的分析与结论。

主报告的重点是完整呈现实验过程和对比分析，而不是只给一个最终分数。

## 8. Task 5：挑战最优 SFT 方案

在完成了上述微调实验后，需要同学们总结训练经验，探索最优的训练方案。在前面 Task 1–4 的基础上继续深挖 SFT 设计，并在实验平台上提交测试集的结果尝试，挑战 leaderboard。

这个任务的要求是：

* 在测试集 `test.jsonl` 上推理；
* 在课程实验平台提交；
* 测试集只提供题目，不公布标准答案；
* 大家可以基于前面的方法自行探索更优设计；
* 至多可以提交 5 次，取最优结果。

## 9. 需要上传到实验平台的内容

### 9.1 最终版本的实验报告、脚本、结果合集

#### 上传要求

提交截止时间：**第五周周日 23:59 之前（4 月 5 日 23:59 前）**

* **PDF 实验报告（不超过四页）（组长姓名_学号.pdf）**

  * 内容结构自由组织，包含但不限于实验方案设计、结果对比、分析结论与关键思考；
* **主线任务过程脚本（/task_scripts）**

  * Task 1–5 的过程脚本；
  * Task 1–5 实验过程记录（如训练 log、loss 曲线）
* **训练集合集（/sft_train_data） & 验证集推理合集（/sft_val_data）**

  * 即提供 Task 1 & Task 5 对应的训练集数据、在验证集上的推理结果合集。
* **上传内容说明（readme.md）**

  * 即上传的脚本结构、各个脚本功能、任务实行情况说明。

#### 注意

* 上传文件的大小不得超过 **200MB**；
* **不需要上传模型的权重文件，只上传可复现的过程脚本**；
* 具体内容及格式请参考实验结果上传示例包。

#### 提交入口

在实验详情——上传结果——**上传实验报告及脚本集** 这里提交。

### 9.2 Task 5 的 leaderboard 推理结果

#### 上传要求

截止时间：**第五周上课前（3 月 31 日 18:30 前）**

在实验平台上上传 Task 5 最终产生的推理文件 `task_5_test.jsonl`。

每条样本格式如下（即增加模型推理结果的 `answer` 字段，要求输出是一个解析的 JSON，且至少包含 `answer` 字段，可以根据方案自由添加 `thinking` 等额外字段）：

```json
{"question": "患者，男性，15岁，自幼有出血倾向。实验室检查:出血时间延长，凝血时间正常，PLT200×109/L，血小板黏附聚集功能障碍。其父也有类似病史。最可能的诊断为", "option": {"A": "血友病", "B": "血小板减少性紫癜", "C": "遗传性出血性毛细血管扩张症", "D": "血管性假血友病", "E": "维生素K缺乏"}, "answer": "{\"answer\": \"D\"}"}
```

可以在平台上下载参考的上传格式模版。

#### 提交入口

在实验详情——上传结果——**上传 leaderboard 测试结果** 这里提交。

### 9.3 leaderboard 测试结果页面补充说明

根据文档最后一页页面内容，`task_5_test.jsonl` 结果文件需满足：

* 每条记录的 `answer` 字段必须是一个 **JSON**；
* 对应内容中也必须包含 `answer` 键；
* 可以附带额外保留 `thinking` 等字段，但系统**只读取其中的 `answer` 作为最终选项**；
* 系统会基于隐藏参考答案自动计算准确率，并实时更新榜单；
* 该接口**最多可提交 5 次**；
* 若 leaderboard 结果包括格式不符合要求，系统会直接删除该次上传文件并提示错误；
* 页面中还显示：**jsonl 文件大小不能超过 50MB**。

页面表单区域包含：

* 上传实验名
* 选择 leaderboard 结果文件
* 上传并验证 leaderboard 结果


