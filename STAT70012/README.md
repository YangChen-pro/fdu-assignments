# 知识图谱概念与技术

> 2025202601STAT70012.01

Neo4j + Flask 的四大名著人物关系图谱示例项目。

运行步骤：

1) 启动 Neo4j（Docker，无密码，仅本机访问）

`bash ./start_docker.sh`

打开 Neo4j 自带的 Web UI（Neo4j Browser）：
- `http://localhost:7474`（脚本默认 `AUTH_MODE=none`，无需账号密码）

2) 安装 Python 依赖

`pip install -r requirement.txt`

3) 导入图谱数据

`python kgqa/neo_db/create_graph.py`

数据格式说明（`kgqa/raw_data/relation.txt`，逗号分隔、无表头）：

- 兼容两种格式：
  - v1（原格式，5 列）：`head,tail,relation,head_cate,tail_cate`
    - 默认 `head_type=Person`、`tail_type=Person`
  - v2（扩展格式，7+ 列）：`head,tail,relation,head_cate,tail_cate,head_type,tail_type,(...)`
- `*_cate`：用于分组/着色（原项目是家族/阵营：贾家荣国府、贾家宁国府、王家、史家、薛家、林家、其他）
- `*_type`：实体类型（用于 Neo4j 标签），建议用英文枚举：`Person/Place/Org/Item/Event/Text`

导入后节点会带 `:Entity` 标签，并额外带一个类型标签（例如 `:Person`、`:Place`），并写入属性：
- `Name`：实体名
- `etype`：实体类型
- `cate`：分组

从原始书籍文本抽取并“增量扩充”关系（避免重复）建议做法：

1) 用 vLLM(OpenAI 兼容) 从 `kgqa/data/input/hlm.txt` 抽取候选关系（输出 `.jsonl`）：

- 编辑 `kgqa/scripts/extract_relations_llm.py` 顶部常量：`VLLM_BASE_URL / VLLM_MODEL / VLLM_API_KEY`
- 默认开启“二次模型验真”（`VERIFY_WITH_LLM=True`），会把明显靠推断/共现的关系过滤掉，结果更适合入库
- 运行：`python kgqa/scripts/extract_relations_llm.py`
- 输出：`kgqa/data/output/new_relations.jsonl`（已做源头去重，不会重复旧库）

也可以自行抽取，输出为 `.jsonl` 或 `.txt/.csv`（建议每条关系带 `head/tail/relation/head_cate/tail_cate/head_type/tail_type`）。
2) 用合并脚本去重后写回 `kgqa/raw_data/relation.txt`（本项目用“脚本顶部常量”配置路径）：

- 编辑 `kgqa/scripts/merge_relations.py` 顶部的 `NEW_RELATIONS_PATH`（必要时也改 `EMIT_MODE`）
- 默认不会覆盖原文件：会输出到 `kgqa/data/output/relation.txt`（可在脚本里改 `OUTPUT_PATH`）
- 运行：`python kgqa/scripts/merge_relations.py`


4) 启动 Web

`python kgqa/app.py`

访问：
- `http://localhost:5000`


得分：A