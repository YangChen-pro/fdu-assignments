# -*- coding: utf-8 -*-
"""
兼容 Python 3.12 的问句解析实现。

原项目依赖 pyltp + LTP 3.x 模型（较老，常在新 Python 上无法安装/运行）。
这里改为：基于 Neo4j 图谱中已有的人名 + neo_db/config.py 的关系同义词表做解析，
不再需要额外模型文件。
"""

from __future__ import annotations

from dataclasses import dataclass

from neo_db.config import graph, similar_words


@dataclass(frozen=True)
class ParsedQuestion:
    subject: str
    relations: list[str]


def _extract_subject(question: str) -> str | None:
    question = (question or "").strip()
    if not question:
        return None

    # 直接用图谱里的人名做匹配：选最长命中的名字，避免“宝玉”命中“贾宝玉”的问题
    data = graph.run(
        "MATCH (e:Entity) "
        "WHERE $q CONTAINS e.Name "
        "RETURN e.Name AS name "
        "ORDER BY size(e.Name) DESC "
        "LIMIT 1",
        q=question,
    )
    record = data.evaluate()
    return record


def _extract_relations(question: str) -> list[str]:
    question = (question or "").strip()
    if not question:
        return []

    hits: list[tuple[int, str]] = []
    for key in similar_words.keys():
        idx = question.find(key)
        if idx != -1:
            hits.append((idx, key))

    hits.sort(key=lambda x: x[0])

    # 去重：同一个 key 只保留一次，按出现顺序
    relations: list[str] = []
    for _, key in hits:
        if key not in relations:
            relations.append(key)
    return relations


def parse_question(question: str) -> ParsedQuestion | None:
    subject = _extract_subject(question)
    if not subject:
        return None

    relations = _extract_relations(question)
    if not relations:
        return None

    return ParsedQuestion(subject=subject, relations=relations)


def get_target_array(words: str):
    """
    给 neo_db/query_graph.get_KGQA_answer() 的兼容输出：
    - array[0]：起始实体名
    - array[1..n]：关系词（必须是 neo_db/config.py:similar_words 的 key）
    - 最后补一个占位符，满足原逻辑的 len(array)-2
    """
    parsed = parse_question(words)
    if not parsed:
        return []

    return [parsed.subject, *parsed.relations, ""]


