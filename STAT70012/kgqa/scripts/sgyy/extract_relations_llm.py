"""
Extract relations from 《西游记》 txt via vLLM (OpenAI-compatible), optionally verify via LLM,
and export to a 7-column txt format compatible with kgqa/raw_data/relation_gemini.txt:

  head,tail,relation,head_cate,tail_cate,head_type,tail_type

Design goals (西游记版):
- Restartable: chunk-level logs; does not overwrite intermediate JSONL.
- Two-pass: extract candidates -> (optional) normalize -> (optional) verify.
- Open relation set: relation is not whitelisted, but strictly cleaned/validated to avoid garbage.
- Strong evidence: keep evidence + chunk_id in JSONL for auditing, but NOT in final 7-col txt.
- Global dedup at export stage with light alias normalization for high-frequency entities.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm  # type: ignore


# =========================
# Config (edit these)
# =========================
BOOK_PATH = Path("kgqa/data/input/sgyy.txt")

OUTPUT_DIR = Path("kgqa/data/output")
OUT_RELATIONS_JSONL = OUTPUT_DIR / "sgyy_new_relations.jsonl"
EXTRACT_LOG_JSONL = OUTPUT_DIR / "sgyy_extract_log.jsonl"
NORMALIZE_LOG_JSONL = OUTPUT_DIR / "sgyy_normalize_log.jsonl"
VERIFY_LOG_JSONL = OUTPUT_DIR / "sgyy_verify_log.jsonl"

# Final export: 7 columns like relation_gemini.txt
EXPORT_TXT_PATH = Path("kgqa/raw_data/sgyy_relation_llm.txt")

# Quick test switches (set to None to disable limits)
CHAPTER_NO_START: int | None = None  # e.g. 1
CHAPTER_NO_END: int | None = None  # e.g. 5 (inclusive) - None means all chapters
MAX_CHUNKS_PER_CHAPTER: int | None = None  # e.g. 2

# Keep the same model calling method and sampling params as HLM script
VLLM_BASE_URL = "http://157.66.255.40:8000/v1"
VLLM_API_KEY = "none"  # set to "none" to disable Authorization header
VLLM_MODEL = "summary"

# Chunk sizing
TOKEN_PER_CHAR = 0.66
CHUNK_TOKEN_LIMIT_EST = 1200  # Reduced to fit server request size limit (~10KB)
CHUNK_OVERLAP_CHARS = 100  # Reduced overlap to save space

# Change this when chunking / parsing / prompts change, to avoid skipping old done chunks.
RUN_SIGNATURE = f"sgyy_v1|limit={CHUNK_TOKEN_LIMIT_EST}|overlap={CHUNK_OVERLAP_CHARS}|multipass=1"

# Multi-pass extraction: increase recall while keeping each response small (less truncation).
# Each pass MUST obey: head/tail appear in evidence verbatim; relation is short and not generic.
EXTRACT_PASSES: list[dict[str, Any]] = [
    {
        "name": "pp_social",
        "max_relations": 15,
        "focus": "优先抽取 人物-人物 的社会关系与互动：结义/兄弟/主公/部将/谋士/师徒/敌对/追杀/擒拿/斩杀/救助/招降/投奔/效忠/背叛/联盟/对峙等。"
    },
    {
        "name": "title_office",
        "max_relations": 15,
        "focus": "优先抽取 封号/官职/任命/受封/册封/加封/封为/拜为/任命为/自称/称帝/让位 等制度性关系。"
    },
    {
        "name": "battle_event",
        "max_relations": 15,
        "focus": "优先抽取 战役/战斗/事件 相关：参与战役/发动战役/指挥战役/战败于/战胜/单挑/斩将/攻打/防守/夺取/丢失/火烧等。"
    },
    {
        "name": "item_place",
        "max_relations": 12,
        "focus": "优先抽取 人物-兵器/宝物 以及 人物-地点/城池 的关系：使用兵器/赠送/夺取/镇守/驻扎/占据/攻占/前往/出生于/死于等。"
    },
]

# Retries
JSON_RETRY_MAX = 3  # if response is not valid JSON, re-request up to 3 times
HTTP_RETRY_MAX = 3
HTTP_RETRY_SLEEP_SEC = 2.0

# vLLM sampling params (keep unchanged)
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0.0

# LLM normalization batch size
NORMALIZE_BATCH_SIZE = 10

# Second-pass LLM verification (to improve precision)
VERIFY_WITH_LLM = True
VERIFY_BATCH_SIZE = 5


TYPE_ENUM = ["Person", "Place", "Org", "Item", "Event", "Text"]
CATE_ENUM = [
    "蜀汉阵营",
    "曹魏阵营",
    "东吴阵营",
    "董卓势力",
    "袁绍势力",
    "袁术势力",
    "刘表势力",
    "马腾势力",
    "吕布势力",
    "汉室朝廷",
    "其他诸侯",
    "其他",
]


try:
    # Optional: reuse synonym normalization if available (may be empty).
    from neo_db.config import similar_words  # type: ignore
except Exception:
    similar_words = {}


DEFAULT_ALIAS_MAP: dict[str, str] = {
    # 三国演义核心人物别名
    "玄德": "刘备",
    "刘皇叔": "刘备",
    "先主": "刘备",
    "云长": "关羽",
    "关公": "关羽",
    "美髯公": "关羽",
    "翼德": "张飞",
    "孔明": "诸葛亮",
    "卧龙": "诸葛亮",
    "孟德": "曹操",
    "曹阿瞒": "曹操",
    "魏武帝": "曹操",
    "仲谋": "孙权",
    "子龙": "赵云",
    "孟起": "马超",
    "汉升": "黄忠",
    "文长": "魏延",
    "奉先": "吕布",
    "仲德": "夏侯惇",
    "妙才": "夏侯渊",
    "文远": "张辽",
    "仲康": "许褚",
    "公瑾": "周瑜",
    "子敬": "鲁肃",
    "子明": "吕蒙",
    "伯言": "陆逊",
    "汉献帝": "献帝",
}

_REVERSE_ALIAS_MAP: dict[str, set[str]] = {}
for alias, canonical in DEFAULT_ALIAS_MAP.items():
    _REVERSE_ALIAS_MAP.setdefault(canonical, set()).add(alias)


@dataclass(frozen=True)
class RelationKey:
    head: str
    relation: str
    tail: str
    head_type: str
    tail_type: str


def _jsonl_iter(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def _estimate_tokens(chars: int) -> int:
    return int(chars * TOKEN_PER_CHAR)


def _clean_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ").replace("\u3000", " ")
    s = s.strip()
    # strip wrapping quotes / book title marks
    if len(s) >= 2:
        pairs = [
            ("“", "”"),
            ("\"", "\""),
            ("'", "'"),
            ("《", "》"),
            ("「", "」"),
            ("『", "』"),
            ("(", ")"),
            ("（", "）"),
            ("[", "]"),
            ("【", "】"),
        ]
        for l, r in pairs:
            if s.startswith(l) and s.endswith(r):
                s = s[1:-1].strip()
                break
    # remove control chars
    s = "".join(ch for ch in s if ord(ch) >= 32)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


_STOP_REL_WORDS = {
    "相关",
    "有关",
    "提到",
    "出现",
    "认为",
    "知道",
    "看到",
    "说",
    "讲",
    "对话",
    "发生",
    "关系",
    # overly generic / functional relations (open set, but these add little value)
    "是",
    "为",
    "在",
    "有",
    "叫",
    "名为",
    "称为",
    "成为",
    "属于",
    "关于",
    "对",
    "向",
    "从",
    "把",
    "将",
    "给",
    "让",
}


def _norm_relation(rel: str) -> str:
    rel = _clean_text(rel)
    if not rel:
        return rel
    rel = rel.replace("，", "").replace(",", "").replace("。", "")
    rel = rel.replace("：", "").replace(":", "").strip()
    # common particles trimming
    rel = re.sub(r"(了|着|过|起来|下去|出来|回去|进去)$", "", rel)
    rel = rel.strip()
    # keep it short-ish (open set, but avoid sentences)
    if len(rel) > 12:
        rel = rel[:12]
    if len(rel) < 2:
        return ""
    if rel in _STOP_REL_WORDS:
        return ""
    # optional synonym map (usually for relation words)
    return similar_words.get(rel, rel)


def _aliases_for_entity(name: str) -> list[str]:
    name = _clean_text(name)
    if not name:
        return []
    aliases = set(_REVERSE_ALIAS_MAP.get(name, set()))
    aliases.add(name)
    return sorted(aliases, key=len, reverse=True)


def _strip_entity_fragments_from_relation(rel: str, *, head: str, tail: str) -> str:
    rel = _clean_text(rel)
    if not rel:
        return ""

    # Strip head/tail and common aliases if they leaked into the relation name.
    for frag in _aliases_for_entity(head) + _aliases_for_entity(tail):
        if frag and frag in rel:
            rel = rel.replace(frag, "")

    # Only do very conservative cleanup here; do NOT strip meaningful prepositions like “于”.
    rel = rel.strip()
    rel = re.sub(r"^[，。！？；:：、】【】\\-]+", "", rel).strip()
    rel = re.sub(r"[，。！？；:：、】【】\\-]+$", "", rel).strip()
    rel = re.sub(r"\s+", " ", rel).strip()
    return rel


def _clean_open_relation(rel: str, *, head: str, tail: str) -> str:
    """
    Open relation set, but with strict cleaning:
    - forbid embedding head/tail (or their aliases) inside relation
    - keep it short and non-generic via _norm_relation()
    """
    rel0 = _clean_text(rel)
    if not rel0:
        return ""
    rel0 = rel0.replace("，", "").replace(",", "").replace("。", "")
    rel0 = rel0.replace("：", "").replace(":", "").strip()
    rel0 = _strip_entity_fragments_from_relation(rel0, head=head, tail=tail)
    rel1 = _norm_relation(rel0)
    if not rel1:
        return ""
    for frag in _aliases_for_entity(head) + _aliases_for_entity(tail):
        if frag and frag in rel1:
            return ""
    # block trivial patterns (pure function words / too empty)
    if rel1 in _STOP_REL_WORDS:
        return ""
    return rel1


def _norm_type(t: str) -> str:
    t = _clean_text(t)
    if not t:
        return "Person"
    mapping = {
        "Location": "Place",
        "Place": "Place",
        "Person": "Person",
        "Org": "Org",
        "Organization": "Org",
        "Item": "Item",
        "Object": "Item",
        "Event": "Event",
        "Text": "Text",
    }
    t = mapping.get(t, t)
    return t if t in TYPE_ENUM else "Person"


def _norm_cate(c: str) -> str:
    c = _clean_text(c)
    return c if c in CATE_ENUM else "其他"


def _norm_entity(name: str) -> str:
    name = _clean_text(name)
    if not name:
        return ""
    # exact alias normalization
    return DEFAULT_ALIAS_MAP.get(name, name)


def _find_all(haystack: str, needle: str) -> list[int]:
    if not haystack or not needle:
        return []
    out: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return out
        out.append(idx)
        start = idx + max(1, len(needle))


def _repair_evidence_in_chunk(*, chunk_text: str, head: str, tail: str, evidence: str) -> str | None:
    """
    Select a continuous substring from chunk_text that contains BOTH head and tail.
    Prefer a short window, then (optionally) expand to sentence boundaries.
    """
    chunk_text = chunk_text or ""
    head = _clean_text(head)
    tail = _clean_text(tail)
    evidence = _clean_text(evidence)
    if not chunk_text or not head or not tail:
        return None

    if evidence and evidence in chunk_text and head in evidence and tail in evidence and len(evidence) <= 120:
        return evidence

    head_pos = _find_all(chunk_text, head)
    tail_pos = _find_all(chunk_text, tail)
    if not head_pos or not tail_pos:
        return None

    best_s = None
    best_e = None
    best_span = None
    for hp in head_pos:
        for tp in tail_pos:
            s = min(hp, tp)
            e = max(hp + len(head), tp + len(tail))
            span = e - s
            if best_span is None or span < best_span:
                best_span = span
                best_s, best_e = s, e
    if best_s is None or best_e is None:
        return None

    s0, e0 = best_s, best_e

    # Expand to nearest sentence boundary (Chinese punctuation or newline).
    left = -1
    for p in ["。", "！", "？", "；", "\n"]:
        left = max(left, chunk_text.rfind(p, 0, s0))
    s1 = left + 1 if left >= 0 else 0

    right_candidates = [chunk_text.find(p, e0) for p in ["。", "！", "？", "；", "\n"]]
    right_candidates = [x for x in right_candidates if x >= 0]
    e1 = (min(right_candidates) + 1) if right_candidates else len(chunk_text)

    ev = chunk_text[s1:e1].strip()
    if head not in ev or tail not in ev:
        ev = chunk_text[s0:e0].strip()

    if len(ev) > 120:
        ev = chunk_text[s0:e0].strip()
    if len(ev) > 120:
        # Clip around minimal span with a small margin.
        clip_s = max(0, s0 - 20)
        clip_e = min(len(chunk_text), e0 + 20)
        ev = chunk_text[clip_s:clip_e].strip()

    if head in ev and tail in ev and ev in chunk_text:
        return ev[:120]
    return None


def _looks_like_non_entity(name: str) -> bool:
    if not name:
        return True
    if len(name) > 40:
        return True
    if re.fullmatch(r"[0-9]+", name):
        return True
    if re.fullmatch(r"[\W_]+", name):
        return True
    return False


_GENERIC_TITLES = {
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "她们",
    "大家",
    "众人",
    "主公",
    "丞相",
    "将军",
    "太守",
    "刺史",
    "大王",
    "陛下",
    "皇上",
    "先生",
    "老者",
}


def _looks_like_generic_title(name: str) -> bool:
    name = _clean_text(name)
    if not name:
        return True
    if name in _GENERIC_TITLES:
        return True
    return False


_CATE_FIXED_BY_ENTITY: dict[str, str] = {
    # 蜀汉阵营
    "刘备": "蜀汉阵营",
    "关羽": "蜀汉阵营",
    "张飞": "蜀汉阵营",
    "诸葛亮": "蜀汉阵营",
    "赵云": "蜀汉阵营",
    "马超": "蜀汉阵营",
    "黄忠": "蜀汉阵营",
    "魏延": "蜀汉阵营",
    "姜维": "蜀汉阵营",
    "庞统": "蜀汉阵营",
    # 曹魏阵营
    "曹操": "曹魏阵营",
    "曹丕": "曹魏阵营",
    "曹植": "曹魏阵营",
    "司马懿": "曹魏阵营",
    "夏侯惇": "曹魏阵营",
    "夏侯渊": "曹魏阵营",
    "张辽": "曹魏阵营",
    "许褚": "曹魏阵营",
    "典韦": "曹魏阵营",
    "郭嘉": "曹魏阵营",
    "荀彧": "曹魏阵营",
    # 东吴阵营
    "孙权": "东吴阵营",
    "孙策": "东吴阵营",
    "周瑜": "东吴阵营",
    "鲁肃": "东吴阵营",
    "吕蒙": "东吴阵营",
    "陆逊": "东吴阵营",
    "黄盖": "东吴阵营",
    "甘宁": "东吴阵营",
    # 董卓势力
    "董卓": "董卓势力",
    "吕布": "吕布势力",
    "貂蝉": "董卓势力",
    # 汉室朝廷
    "汉献帝": "汉室朝廷",
    "献帝": "汉室朝廷",
    "何进": "汉室朝廷",
}


def _infer_cate(*, name_norm: str, name_raw: str, ent_type: str) -> str:
    """
    Rule-first category inference for 三国演义.
    Keep it conservative; return '其他' if unsure.
    """
    name_norm = _clean_text(name_norm)
    name_raw = _clean_text(name_raw)
    ent_type = _norm_type(ent_type)

    if name_norm in _CATE_FIXED_BY_ENTITY:
        return _CATE_FIXED_BY_ENTITY[name_norm]

    n = name_norm or name_raw
    if not n:
        return "其他"

    # Keyword cues for faction classification
    # Shu-Han faction
    if any(k in n for k in ["蜀", "汉中", "益州", "成都", "白帝城"]):
        if ent_type == "Person":
            return "蜀汉阵营"
        if ent_type == "Place":
            return "蜀汉阵营"

    # Wei faction
    if any(k in n for k in ["曹", "魏", "许都", "邺城", "洛阳"]):
        if ent_type == "Person" and "曹" in n:
            return "曹魏阵营"
        if ent_type == "Person" and any(k in n for k in ["夏侯", "司马", "荀", "郭"]):
            return "曹魏阵营"

    # Wu faction
    if any(k in n for k in ["孙", "吴", "江东", "建业", "柴桑"]):
        if ent_type == "Person" and "孙" in n:
            return "东吴阵营"

    # Dong Zhuo faction
    if any(k in n for k in ["董卓", "吕布"]):
        return "董卓势力"

    # Yuan Shao faction
    if any(k in n for k in ["袁绍", "河北", "冀州"]):
        if ent_type == "Person":
            return "袁绍势力"

    # Yuan Shu faction
    if "袁术" in n:
        return "袁术势力"

    # Han Imperial Court
    if any(k in n for k in ["汉献帝", "献帝", "皇帝", "朝廷", "何进"]):
        return "汉室朝廷"

    # Other warlords/factions
    if any(k in n for k in ["刘表", "刘璋", "马腾", "公孙瓒", "吕布"]):
        if "吕布" in n:
            return "吕布势力"
        return "其他诸侯"

    return "其他"


def _is_fixed_cate(name_norm: str) -> bool:
    return _clean_text(name_norm) in _CATE_FIXED_BY_ENTITY


def _align_entity_to_evidence(entity_raw: str, *, entity_norm: str, evidence: str) -> str:
    """
    Ensure entity surface form matches evidence substring.
    If entity_raw not in evidence but an alias of entity_norm is, switch to that alias.
    """
    entity_raw = _clean_text(entity_raw)
    entity_norm = _clean_text(entity_norm)
    evidence = _clean_text(evidence)
    if not evidence:
        return entity_raw
    if entity_raw and entity_raw in evidence:
        return entity_raw
    for alias in _aliases_for_entity(entity_norm):
        if alias and alias in evidence:
            return alias
    return entity_raw


def _make_key(obj: dict[str, Any]) -> RelationKey:
    head_raw = _clean_text(obj.get("head", ""))
    tail_raw = _clean_text(obj.get("tail", ""))
    head_norm = _clean_text(obj.get("head_norm")) or _norm_entity(head_raw)
    tail_norm = _clean_text(obj.get("tail_norm")) or _norm_entity(tail_raw)
    return RelationKey(
        head=head_norm,
        relation=_clean_open_relation(obj.get("relation", ""), head=head_norm, tail=tail_norm),
        tail=tail_norm,
        head_type=_norm_type(obj.get("head_type", "")),
        tail_type=_norm_type(obj.get("tail_type", "")),
    )


def _load_done_chunks() -> set[str]:
    done: set[str] = set()
    for obj in _jsonl_iter(EXTRACT_LOG_JSONL):
        # Only treat as done if it matches current run signature.
        # Older logs (no signature) will be ignored so changes take effect on rerun.
        if obj.get("run_signature") != RUN_SIGNATURE:
            continue
        if obj.get("status") == "ok" and obj.get("chunk_id"):
            done.add(str(obj["chunk_id"]))
    return done


def _score_record(obj: dict[str, Any]) -> int:
    """
    Higher score means better metadata; used to allow reruns to 'upgrade' records.
    """
    head_cate = _norm_cate(obj.get("head_cate", ""))
    tail_cate = _norm_cate(obj.get("tail_cate", ""))
    evidence = _clean_text(obj.get("evidence"))
    head = _clean_text(obj.get("head"))
    tail = _clean_text(obj.get("tail"))
    head_norm = _clean_text(obj.get("head_norm")) or _norm_entity(head)
    tail_norm = _clean_text(obj.get("tail_norm")) or _norm_entity(tail)
    head_type = _norm_type(obj.get("head_type", ""))
    tail_type = _norm_type(obj.get("tail_type", ""))
    score = 0
    score += 2 if head_cate != "其他" else 0
    score += 2 if tail_cate != "其他" else 0
    score += 1 if evidence and head and tail and (head in evidence) and (tail in evidence) else 0
    # prefer shorter evidence when present
    if evidence:
        score += 1 if len(evidence) <= 80 else 0
    # reward agreement with rule-based cate (so corrected cate can replace earlier wrong cate)
    inferred_h = _infer_cate(name_norm=head_norm, name_raw=head, ent_type=head_type)
    inferred_t = _infer_cate(name_norm=tail_norm, name_raw=tail, ent_type=tail_type)
    score += 1 if inferred_h != "其他" and head_cate == inferred_h else 0
    score += 1 if inferred_t != "其他" and tail_cate == inferred_t else 0
    return score


def _load_existing_best_scores_from_output() -> dict[RelationKey, int]:
    best: dict[RelationKey, int] = {}
    for obj in _jsonl_iter(OUT_RELATIONS_JSONL):
        try:
            k = _make_key(obj)
        except Exception:
            continue
        if k.head and k.tail and k.relation:
            s = _score_record(obj)
            prev = best.get(k)
            if prev is None or s > prev:
                best[k] = s
    return best


def _http_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY and VLLM_API_KEY.lower() != "none":
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=120) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8", errors="replace"))


def _vllm_chat(messages: list[dict[str, str]]) -> str:
    url = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
    }

    last_err: str | None = None
    for attempt in range(1, HTTP_RETRY_MAX + 1):
        try:
            data = _http_post_json(url, payload)
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("Empty choices in response")
            msg = (choices[0].get("message") or {}).get("content")
            if not isinstance(msg, str):
                raise ValueError("Missing message.content")
            return msg
        except HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            last_err = f"HTTPError {getattr(e, 'code', '')}: {err_body[:500]}"
        except URLError as e:
            last_err = f"URLError: {e}"
        except Exception as e:
            last_err = f"Error: {e}"

        if attempt < HTTP_RETRY_MAX:
            time.sleep(HTTP_RETRY_SLEEP_SEC)
            continue
        raise RuntimeError(last_err or "HTTP request failed")


def _parse_llm_json_with_retry(messages: list[dict[str, str]]) -> Any:
    def try_load(s: str) -> Any | None:
        try:
            return json.loads(s)
        except Exception:
            return None

    def salvage_array(s: str) -> list[Any] | None:
        s = (s or "").strip()
        if not s.startswith("["):
            return None
        objs: list[str] = []
        in_str = False
        esc = False
        depth = 0
        arr_depth = 0
        obj_start: int | None = None
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == "\"":
                    in_str = False
                continue

            if ch == "\"":
                in_str = True
                continue
            if ch == "[":
                arr_depth += 1
                continue
            if ch == "]":
                arr_depth = max(0, arr_depth - 1)
                continue
            if ch == "{":
                if arr_depth >= 1:
                    if depth == 0:
                        obj_start = i
                    depth += 1
                continue
            if ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        objs.append(s[obj_start : i + 1])
                        obj_start = None
                continue

        if not objs:
            return None
        salvaged = "[" + ",".join(objs) + "]"
        out = try_load(salvaged)
        return out if isinstance(out, list) else None

    def salvage_relations(s: str) -> dict[str, Any] | list[Any] | None:
        s = (s or "").strip()
        if not s:
            return None
        if s.startswith("["):
            arr = salvage_array(s)
            if arr is not None:
                return arr
        # try to salvage an inner relations array from an object
        idx = s.find("\"relations\"")
        if idx >= 0:
            arr_start = s.find("[", idx)
            if arr_start >= 0:
                arr = salvage_array(s[arr_start:])
                if arr is not None:
                    return {"relations": arr, "_salvaged": True}
        return None

    last: str | None = None
    for attempt in range(1, JSON_RETRY_MAX + 1):
        out = _vllm_chat(messages)
        last = out
        loaded = try_load(out)
        if loaded is not None:
            return loaded
        salvaged = salvage_relations(out)
        if salvaged is not None:
            return salvaged
        try:
            raise ValueError("json_parse_failed")
        except Exception:
            fix_msgs = [
                {
                    "role": "system",
                    "content": "你是 JSON 修复器。只输出合法 JSON，不要代码块，不要解释，不要多余文字。",
                },
                {
                    "role": "user",
                    "content": (
                        "上一次输出不是合法 JSON，请将其修复为合法 JSON 并保持原意不变。\n"
                        "注意：只能输出 JSON。\n"
                        f"原输出：\n{out}"
                    ),
                },
            ]
            messages = fix_msgs
    raise ValueError(f"Failed to parse JSON after retries. Last output: {last[:500] if last else ''}")


def _build_extract_messages(
    *,
    chapter_no: int,
    chunk_id: str,
    text: str,
    pass_name: str,
    focus: str,
    max_relations: int,
    already_keys: list[str],
) -> list[dict[str, str]]:
    type_list = "/".join(TYPE_ENUM)
    cate_list = "、".join(CATE_ENUM)
    system = (
        "你是信息抽取助手。你的任务是从给定《三国演义》文本中抽取关系三元组。"
        "只能输出合法 JSON（不要代码块、不要解释、不要多余文字）。"
    )
    already = ""
    if already_keys:
        # Keep prompt short: provide only a few existing triples to reduce duplicates.
        sample = already_keys[:30]
        already = "已抽取过的三元组（本次不要重复）：\n" + "\n".join(f"- {x}" for x in sample) + "\n\n"
    user = (
        "请从下面《三国演义》文本中抽取关系三元组，输出严格 JSON。\n"
        f"- chunk_id: {chunk_id}\n"
        f"- chapter_no: {chapter_no}\n"
        f"- 本轮抽取重点({pass_name}): {focus}\n"
        f"- 本轮最多输出 {max_relations} 条关系（宁缺毋滥；不要为了凑数输出低质量）。\n"
        f"- type 枚举（必须从中选择）：{type_list}\n"
        f"- cate 枚举（必须从中选择，否则填'其他'）：{cate_list}\n"
        "\n"
        "字段要求：\n"
        "- head 与 tail 必须是具体专名（人名/地名/城池/官职/兵器等），不要用代词或泛称（他/她/主公/大王/将军/丞相等）。不确定就不要输出。\n"
        "- 重要：head 与 tail 必须是 evidence 中**原样出现**的连续子串（完全一致），不要把'玄德'改写成'刘备'，不要补全未出现在 evidence 的字。\n"
        "- relation 为'谓词短语'，开放集合：\n"
        "  * 必须简短（2~8 个中文为佳，最长不超过 12 个字）\n"
        "  * 不要包含逗号/换行/冒号/引号\n"
        "  * 不要输出整句或'相关/有关/提到/出现/发生'等泛词\n"
        "  * relation 里不要包含 head/tail 的名字或别名（例如不要输出'斩杀华雄'，应输出'斩杀'）\n"
        "- head_type/tail_type 必须选择枚举值。\n"
        "- head_cate/tail_cate 必须选择枚举值；尽量按《三国演义》体系归类（蜀汉/曹魏/东吴/汉室朝廷等）。\n"
        "- evidence 必须是原文中的连续子串(<=80字)，并且 evidence 中必须同时出现 head 与 tail（否则不要输出该条）。\n"
        "- 同一条关系不要重复输出。\n"
        "- relations 可以为空数组。\n"
        "\n"
        f"{already}"
        "输出 JSON 结构：\n"
        "{\n"
        '  "chunk_id": "...",\n'
        '  "chapter_no": 12,\n'
        '  "relations": [\n'
        "    {\n"
        '      "head": "...",\n'
        '      "relation": "...",\n'
        '      "tail": "...",\n'
        '      "head_type": "Person",\n'
        '      "tail_type": "Person",\n'
        '      "head_cate": "其他",\n'
        '      "tail_cate": "其他",\n'
        '      "evidence": "从原文摘取的一句或半句短证据"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "文本如下：\n"
        f"{text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_normalize_messages(*, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    type_list = "/".join(TYPE_ENUM)
    cate_list = "、".join(CATE_ENUM)
    system = "你是数据规范化助手。只输出合法 JSON（不要代码块、不要解释、不要多余文字）。"
    user = (
        "请将下面抽取结果规范化（必要时修复字段），严格遵守枚举约束。\n"
        f"- type 必须从：{type_list}\n"
        f"- cate 必须从：{cate_list}（否则填'其他'）\n"
        "\n"
        "类目参考（尽量按此纠偏）：\n"
        "- 刘备/关羽/张飞/诸葛亮/赵云/马超/黄忠/魏延：蜀汉阵营\n"
        "- 曹操/曹丕/曹植/司马懿/夏侯惇/张辽/许褚/典韦/郭嘉/荀彧：曹魏阵营\n"
        "- 孙权/孙策/周瑜/鲁肃/吕蒙/陆逊/黄盖/甘宁：东吴阵营\n"
        "- 献帝/何进/董卓/吕布：董卓势力、汉室朝廷\n"
        "- 袁绍/袁术/刘表/马腾/公孙瓒：其他诸侯\n"
        "- 华雄/李儒/张角/张宝/张梁：其他势力\n"
        "\n"
        "规范化要求（不满足就输出 null）：\n"
        "- head/tail 必须是具体专名，不要代词/泛称。\n"
        "- head/tail 必须是 evidence 中原样出现的连续子串（完全一致），不要补全/改写。\n"
        "- relation 必须是简短谓词短语（2~12字），不能是整句，不能含逗号/换行/引号；泛词（相关/有关/提到/出现/发生等）直接输出 null。\n"
        "- relation 中不得包含 head/tail 的名字或别名；例如不要输出'斩杀华雄'，应输出'斩杀'。\n"
        "- head_type/tail_type 必须选枚举值。\n"
        "- head_cate/tail_cate 必须选枚举值。\n"
        "- evidence 必须保留原样或更短，并且尽量包含 head 与 tail。\n"
        "\n"
        "输入是一个 JSON 数组，每个元素包含 head/tail/relation/.../evidence。\n"
        "输入数组中每个元素都包含字段 id（整数）。\n"
        "输出必须是一个 JSON 数组；每个输出元素都必须包含相同 id：\n"
        "- 若能规范化，输出对象：{id,head,relation,tail,head_type,tail_type,head_cate,tail_cate,evidence}\n"
        "- 若无法可靠规范化，输出对象：{id, drop: true}\n"
        "注意：允许你改变输出顺序，但 id 必须正确。\n"
        "\n"
        "输入：\n"
        f"{json.dumps(items, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_verify_messages(
    *,
    chapter_no: int,
    chunk_id: str,
    chunk_text: str,
    items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "你是关系校验助手。你的任务是对候选关系逐条验真。"
        "只能输出合法 JSON（不要代码块、不要解释、不要多余文字）。"
    )
    user = (
        "请根据原文对候选关系逐条验真。\n"
        f"- chunk_id: {chunk_id}\n"
        f"- chapter_no: {chapter_no}\n"
        "\n"
        "校验规则（非常严格）：\n"
        "- 只能在原文中找到明确措辞时 keep=true；仅凭同段共现/常识推断，一律 keep=false。\n"
        "- 必须方向一致：head -[relation]-> tail。\n"
        "- 若 head/tail 是代词或泛称（师父/国王/大王/和尚/妖怪等），无法落到具体专名，则 keep=false。\n"
        "- 若 evidence 中没有同时出现 head 与 tail，则 keep=false。\n"
        "- relation 必须是短谓词（2~12字），不能是整句或泛词。\n"
        "\n"
        "输出格式：JSON 数组，长度与候选列表相同，每项包含：\n"
        '{ "id": 0, "keep": true/false, "reason": "<=30字" }\n'
        "\n"
        "原文（可在整段内找证据，不限于 evidence 一句）：\n"
        f"{chunk_text}\n"
        "\n"
        "候选关系列表（JSON）：\n"
        f"{json.dumps(items, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalize_rule_first(
    item: dict[str, Any],
    *,
    chunk_text: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    head = _clean_text(item.get("head"))
    tail = _clean_text(item.get("tail"))
    rel = _clean_text(item.get("relation"))
    head_type = _norm_type(item.get("head_type", ""))
    tail_type = _norm_type(item.get("tail_type", ""))
    raw_head_cate = _clean_text(item.get("head_cate"))
    raw_tail_cate = _clean_text(item.get("tail_cate"))
    head_cate = _norm_cate(raw_head_cate)
    tail_cate = _norm_cate(raw_tail_cate)
    evidence = _clean_text(item.get("evidence"))

    # keep surface form (for evidence matching) and normalized form (for dedup/export)
    head_norm = _clean_text(item.get("head_norm")) or _norm_entity(head)
    tail_norm = _clean_text(item.get("tail_norm")) or _norm_entity(tail)

    # If evidence exists, align head/tail surface form to evidence substring (悟空 vs 孙悟空).
    if evidence:
        head = _align_entity_to_evidence(head, entity_norm=head_norm, evidence=evidence)
        tail = _align_entity_to_evidence(tail, entity_norm=tail_norm, evidence=evidence)

    if _looks_like_non_entity(head) or _looks_like_non_entity(tail):
        return None, False
    if _looks_like_generic_title(head) or _looks_like_generic_title(tail):
        return None, False
    if head == tail:
        return None, False

    # Clean relation using normalized entity names (better at stripping aliases like 玉帝).
    rel_norm = _clean_open_relation(rel, head=head_norm, tail=tail_norm)
    if not rel_norm:
        return None, False

    needs_fix = False

    # evidence should be short and include both entities; attempt deterministic repair first
    if not evidence or len(evidence) > 120 or head not in evidence or tail not in evidence:
        repaired = None
        if chunk_text:
            repaired = _repair_evidence_in_chunk(
                chunk_text=chunk_text, head=head, tail=tail, evidence=evidence
            )
        if repaired:
            evidence = repaired
        else:
            needs_fix = True

    if chunk_text:
        if not evidence or evidence not in chunk_text:
            repaired2 = _repair_evidence_in_chunk(
                chunk_text=chunk_text, head=head, tail=tail, evidence=evidence
            )
            if repaired2:
                evidence = repaired2
                head = _align_entity_to_evidence(head, entity_norm=head_norm, evidence=evidence)
                tail = _align_entity_to_evidence(tail, entity_norm=tail_norm, evidence=evidence)
            else:
                needs_fix = True

    # Rule-first cate correction (only override when current cate is 其他 or invalid)
    inferred_head_cate = _infer_cate(name_norm=head_norm, name_raw=head, ent_type=head_type)
    inferred_tail_cate = _infer_cate(name_norm=tail_norm, name_raw=tail, ent_type=tail_type)
    # Rule-first: override confidently for fixed entities; otherwise only fill when unknown.
    if inferred_head_cate != "其他":
        if _is_fixed_cate(head_norm) and head_cate != inferred_head_cate:
            head_cate = inferred_head_cate
        elif head_cate == "其他":
            head_cate = inferred_head_cate
    if inferred_tail_cate != "其他":
        if _is_fixed_cate(tail_norm) and tail_cate != inferred_tail_cate:
            tail_cate = inferred_tail_cate
        elif tail_cate == "其他":
            tail_cate = inferred_tail_cate

    # If tail is an Item/Text and relation implies offering/giving, use head_cate as tail_cate when unknown.
    if tail_cate == "其他" and head_cate != "其他" and tail_type in ("Item", "Text"):
        if rel_norm in {"奉命献上", "献上", "奉上", "赐予", "赠与", "赏赐", "交付", "送与", "给予"}:
            tail_cate = head_cate

    out = {
        "head": head,
        "relation": rel_norm,
        "tail": tail,
        "head_type": head_type,
        "tail_type": tail_type,
        "head_cate": head_cate,
        "tail_cate": tail_cate,
        "evidence": evidence[:200],
        "head_norm": head_norm,
        "tail_norm": tail_norm,
    }
    # needs_fix if cate not in enum (except explicit '其他')
    if raw_head_cate and raw_head_cate != "其他" and head_cate == "其他":
        needs_fix = True
    if raw_tail_cate and raw_tail_cate != "其他" and tail_cate == "其他":
        needs_fix = True

    # Strict surface-evidence consistency (after deterministic repair):
    if not evidence or head not in evidence or tail not in evidence:
        needs_fix = True

    return out, needs_fix


_CHAPTER_HEADER_RE = re.compile(
    r"^第(?:[一二三四五六七八九十百千0-9]+部分)?第?\s*([0-9]{1,4}|[一二三四五六七八九十百千]+)\s*回(?:\s+(.*))?$"
)


def _cn_num_to_int(s: str) -> int | None:
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    mapping = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    unit = {"十": 10, "百": 100, "千": 1000}
    total = 0
    num = 0
    last_unit = 1
    for ch in s:
        if ch in mapping:
            num = mapping[ch]
        elif ch in unit:
            u = unit[ch]
            if num == 0:
                num = 1
            total += num * u
            num = 0
            last_unit = u
        else:
            return None
    total += num
    if total == 0 and last_unit == 10 and "十" in s:
        total = 10
    return total or None


def _preprocess_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        s = (line or "").rstrip("\n")
        s = s.replace("\u3000", " ").strip()
        cleaned.append(s)
    return cleaned


def _split_chapters(lines: list[str]) -> list[tuple[int, str, str]]:
    """
    Returns list of (chapter_no, chapter_title, chapter_text).
    Expects headers like:
      - 第001回
      - 第一部分第002回 悟彻菩提真妙理 断魔归本合元神
    """
    header_positions: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        m = _CHAPTER_HEADER_RE.match(line.strip())
        if not m:
            continue
        no = _cn_num_to_int(m.group(1))
        if no is None:
            continue
        header_positions.append((idx, no, line.strip()))

    if not header_positions:
        return [(1, "第001回(无标题)", "\n".join(lines).strip())]

    chapters: list[tuple[int, str, str]] = []
    for i, (start_idx, chapter_no, title) in enumerate(header_positions):
        end_idx = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(lines)
        body = "\n".join(lines[start_idx + 1 : end_idx]).strip()
        if body:
            chapters.append((chapter_no, title, body))
    return chapters


def _split_paragraphs(text: str) -> list[str]:
    paras: list[str] = []
    buf: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            if buf:
                paras.append("".join(buf).strip())
                buf = []
            continue
        buf.append(s)
    if buf:
        paras.append("".join(buf).strip())
    return [p for p in paras if p]


def _split_long_para(para: str, limit_chars: int) -> list[str]:
    para = para.strip()
    if len(para) <= limit_chars:
        return [para]
    out: list[str] = []
    start = 0
    while start < len(para):
        end = min(len(para), start + limit_chars)
        window = para[start:end]
        cut = None
        for m in re.finditer(r"[。！？；]", window):
            cut = m.end()
        if cut is None or cut < max(50, int(limit_chars * 0.5)):
            cut = len(window)
        out.append(window[:cut].strip())
        start += cut
    return [x for x in out if x]


def _build_chunks(chapter_text: str) -> list[str]:
    limit_chars = int((CHUNK_TOKEN_LIMIT_EST - 80) / TOKEN_PER_CHAR)
    paras = _split_paragraphs(chapter_text)

    expanded: list[str] = []
    for p in paras:
        expanded.extend(_split_long_para(p, limit_chars))

    chunks: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    for p in expanded:
        if not p:
            continue
        new_chars = buf_chars + (1 if buf else 0) + len(p)
        if buf and _estimate_tokens(new_chars) > CHUNK_TOKEN_LIMIT_EST:
            chunks.append("\n".join(buf).strip())
            buf = []
            buf_chars = 0
        buf.append(p)
        buf_chars = new_chars
    if buf:
        chunks.append("\n".join(buf).strip())

    capped: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if _estimate_tokens(len(c)) <= CHUNK_TOKEN_LIMIT_EST:
            capped.append(c)
            continue
        capped.extend(_split_long_para(c, limit_chars))

    base = [c for c in capped if c]
    if not base or CHUNK_OVERLAP_CHARS <= 0:
        return base

    overlapped: list[str] = []
    for i, c in enumerate(base):
        if i == 0:
            overlapped.append(c)
            continue
        prev = overlapped[-1]
        tail = prev[-CHUNK_OVERLAP_CHARS:] if len(prev) > CHUNK_OVERLAP_CHARS else prev
        merged = (tail + "\n" + c).strip()
        # ensure token cap: trim from the front (overlap side) if needed
        if _estimate_tokens(len(merged)) > CHUNK_TOKEN_LIMIT_EST:
            max_chars = int((CHUNK_TOKEN_LIMIT_EST - 80) / TOKEN_PER_CHAR)
            merged = merged[-max_chars:].strip()
        overlapped.append(merged)
    return overlapped


def _export_7col_txt() -> None:
    def csv_safe(v: Any) -> str:
        s = _clean_text(v)
        # keep strict 7-col CSV by removing separators/newlines
        return s.replace(",", "，").replace("\n", " ").replace("\r", " ").strip()

    rows: list[dict[str, Any]] = []
    for obj in _jsonl_iter(OUT_RELATIONS_JSONL):
        rows.append(obj)

    best_by_key: dict[RelationKey, dict[str, Any]] = {}
    for r in rows:
        k = _make_key(r)
        if not (k.head and k.tail and k.relation):
            continue
        # normalize stored fields for writing
        head_raw = _clean_text(r.get("head", ""))
        tail_raw = _clean_text(r.get("tail", ""))
        head = _clean_text(r.get("head_norm")) or _norm_entity(head_raw)
        tail = _clean_text(r.get("tail_norm")) or _norm_entity(tail_raw)
        relation = _clean_open_relation(r.get("relation", ""), head=head, tail=tail)
        if not head or not tail or not relation:
            continue
        head_type = _norm_type(r.get("head_type", ""))
        tail_type = _norm_type(r.get("tail_type", ""))
        head_cate = _norm_cate(r.get("head_cate", ""))
        tail_cate = _norm_cate(r.get("tail_cate", ""))

        candidate = {
            "head": csv_safe(head),
            "tail": csv_safe(tail),
            "relation": csv_safe(relation),
            "head_cate": csv_safe(head_cate),
            "tail_cate": csv_safe(tail_cate),
            "head_type": csv_safe(head_type),
            "tail_type": csv_safe(tail_type),
            "chapter_no": int(r.get("chapter_no") or 10**9),
        }

        prev = best_by_key.get(k)
        if prev is None:
            best_by_key[k] = candidate
            continue

        # keep earlier chapter, and prefer non-'其他' cate if tie
        if candidate["chapter_no"] < prev["chapter_no"]:
            best_by_key[k] = candidate
            continue
        if candidate["chapter_no"] == prev["chapter_no"]:
            prev_score = int(prev["head_cate"] != "其他") + int(prev["tail_cate"] != "其他")
            cand_score = int(head_cate != "其他") + int(tail_cate != "其他")
            if cand_score > prev_score:
                best_by_key[k] = candidate

    # stable ordering
    out_rows = sorted(
        best_by_key.values(),
        key=lambda x: (x["chapter_no"], x["head"], x["relation"], x["tail"]),
    )
    lines = [
        ",".join(
            [
                r["head"],
                r["tail"],
                r["relation"],
                r["head_cate"],
                r["tail_cate"],
                r["head_type"],
                r["tail_type"],
            ]
        )
        for r in out_rows
    ]

    EXPORT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    # final format check: every non-empty line must have 7 columns
    for line_no, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        if len(line.split(",")) != 7:
            raise ValueError(f"Export format error at line {line_no}: not 7 columns")
    EXPORT_TXT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_best_scores = _load_existing_best_scores_from_output()
    done_chunks = _load_done_chunks()

    raw = BOOK_PATH.read_text(encoding="utf-8", errors="ignore")
    lines = _preprocess_lines(raw.splitlines())
    chapters = _split_chapters(lines)

    selected_chapters: list[tuple[int, str, str]] = []
    for chapter_no, chapter_title, chapter_text in chapters:
        if CHAPTER_NO_START is not None and chapter_no < CHAPTER_NO_START:
            continue
        if CHAPTER_NO_END is not None and chapter_no > CHAPTER_NO_END:
            continue
        if not chapter_text.strip():
            continue
        selected_chapters.append((chapter_no, chapter_title, chapter_text))

    for chapter_no, chapter_title, chapter_text in tqdm(
        selected_chapters, desc="Chapters", unit="chapter"
    ):
        chunks = _build_chunks(chapter_text)
        if MAX_CHUNKS_PER_CHAPTER is not None:
            chunks = chunks[: MAX_CHUNKS_PER_CHAPTER]

        chunk_iter = tqdm(
            enumerate(chunks),
            total=len(chunks),
            desc=f"Chapter {chapter_no}",
            unit="chunk",
            leave=False,
        )
        for idx, chunk_text in chunk_iter:
            chunk_id = f"{chapter_no}-{idx:04d}"
            if chunk_id in done_chunks:
                continue

            started = time.time()
            extract_ok = False
            normalize_count = 0
            emitted_count = 0
            verified_kept = 0
            verified_dropped = 0
            uncertain_records: list[dict[str, Any]] = []
            raw_rel_count = 0
            err: str | None = None

            try:
                candidates: list[dict[str, Any]] = []
                pass_summaries: list[dict[str, Any]] = []
                already_keys: list[str] = []
                per_chunk_seen: set[RelationKey] = set()

                for p in EXTRACT_PASSES:
                    p_name = str(p.get("name") or "pass")
                    p_focus = str(p.get("focus") or "")
                    p_max = int(p.get("max_relations") or 12)
                    p_raw = 0
                    p_added = 0
                    p_uncertain = 0
                    p_err: str | None = None
                    try:
                        messages = _build_extract_messages(
                            chapter_no=chapter_no,
                            chunk_id=chunk_id,
                            text=chunk_text,
                            pass_name=p_name,
                            focus=p_focus,
                            max_relations=p_max,
                            already_keys=already_keys,
                        )
                        data = _parse_llm_json_with_retry(messages)
                        if isinstance(data, list):
                            rels = data
                        elif isinstance(data, dict):
                            rels = data.get("relations", [])
                        else:
                            raise ValueError("extract response is not an object/list")
                        if not isinstance(rels, list):
                            raise ValueError("relations is not a list")
                        p_raw = len(rels)
                        raw_rel_count += p_raw

                        for r in rels:
                            if not isinstance(r, dict):
                                continue
                            normalized, needs_fix = _normalize_rule_first(r, chunk_text=chunk_text)
                            if normalized is None:
                                continue
                            normalized["chapter_no"] = chapter_no
                            normalized["chunk_id"] = chunk_id

                            # Dedup within a chunk across passes early to save later work.
                            k0 = _make_key(normalized)
                            if not (k0.head and k0.tail and k0.relation):
                                continue
                            if k0 in per_chunk_seen:
                                continue
                            per_chunk_seen.add(k0)

                            already_keys.append(f"{k0.head} -[{k0.relation}]-> {k0.tail}")
                            if len(already_keys) > 60:
                                already_keys = already_keys[:60]

                            # evidence must be substring of chunk (hard rule)
                            ev = normalized.get("evidence") or ""
                            if not ev or ev not in chunk_text:
                                needs_fix = True

                            if needs_fix:
                                uncertain_records.append(normalized)
                                p_uncertain += 1
                                continue
                            candidates.append(normalized)
                            p_added += 1
                    except Exception as e:
                        p_err = str(e)[:200]

                    pass_summaries.append(
                        {
                            "pass": p_name,
                            "raw_relations": p_raw,
                            "added": p_added,
                            "uncertain": p_uncertain,
                            "error": p_err,
                        }
                    )

                # LLM normalization for uncertain items
                fixed: list[dict[str, Any]] = []
                if uncertain_records:
                    for batch_start in range(0, len(uncertain_records), NORMALIZE_BATCH_SIZE):
                        batch = uncertain_records[
                            batch_start : batch_start + NORMALIZE_BATCH_SIZE
                        ]
                        batch_with_id = []
                        for i, it in enumerate(batch):
                            obj = dict(it)
                            obj["id"] = batch_start + i
                            batch_with_id.append(obj)
                        nm = _build_normalize_messages(items=batch_with_id)
                        try:
                            nm_data = _parse_llm_json_with_retry(nm)
                            if not isinstance(nm_data, list):
                                raise ValueError("normalize response is not a list")
                        except Exception as e:
                            # Normalization failure should not kill the whole chunk; keep existing candidates.
                            _append_jsonl(
                                NORMALIZE_LOG_JSONL,
                                {
                                    "chunk_id": chunk_id,
                                    "chapter_no": chapter_no,
                                    "status": "normalize_batch_error_skip",
                                    "batch_start": batch_start,
                                    "batch_size": len(batch),
                                    "error": str(e)[:300],
                                },
                            )
                            continue

                        # Robust mapping: prefer id-based outputs; do not hard fail on length mismatch.
                        out_by_id: dict[int, dict[str, Any]] = {}
                        seq_outputs: list[Any] = []
                        for x in nm_data:
                            if isinstance(x, dict) and "id" in x:
                                try:
                                    out_by_id[int(x["id"])] = x
                                except Exception:
                                    continue
                            else:
                                seq_outputs.append(x)

                        for i, src_item in enumerate(batch):
                            _id = batch_start + i
                            out_item = out_by_id.get(_id)
                            if out_item is None and i < len(seq_outputs):
                                out_item = seq_outputs[i]

                            if out_item is None or (isinstance(out_item, dict) and out_item.get("drop") is True):
                                _append_jsonl(
                                    NORMALIZE_LOG_JSONL,
                                    {
                                        "chunk_id": chunk_id,
                                        "chapter_no": chapter_no,
                                        "status": "dropped",
                                        "id": _id,
                                        "input": src_item,
                                    },
                                )
                                continue
                            if not isinstance(out_item, dict):
                                _append_jsonl(
                                    NORMALIZE_LOG_JSONL,
                                    {
                                        "chunk_id": chunk_id,
                                        "chapter_no": chapter_no,
                                        "status": "invalid_normalize_item",
                                        "id": _id,
                                        "input": src_item,
                                        "output": out_item,
                                    },
                                )
                                continue
                            normalized2, needs_fix2 = _normalize_rule_first(
                                out_item, chunk_text=chunk_text
                            )
                            if normalized2 is None:
                                _append_jsonl(
                                    NORMALIZE_LOG_JSONL,
                                    {
                                        "chunk_id": chunk_id,
                                        "chapter_no": chapter_no,
                                        "status": "invalid_after_normalize",
                                        "id": _id,
                                        "input": src_item,
                                        "output": out_item,
                                    },
                                )
                                continue
                            normalized2["chapter_no"] = chapter_no
                            normalized2["chunk_id"] = chunk_id
                            ev2 = normalized2.get("evidence") or ""
                            if needs_fix2 or not ev2 or ev2 not in chunk_text:
                                _append_jsonl(
                                    NORMALIZE_LOG_JSONL,
                                    {
                                        "chunk_id": chunk_id,
                                        "chapter_no": chapter_no,
                                        "status": "needs_fix_after_normalize",
                                        "id": _id,
                                        "input": src_item,
                                        "output": normalized2,
                                    },
                                )
                                continue
                            fixed.append(normalized2)
                            normalize_count += 1
                            _append_jsonl(
                                NORMALIZE_LOG_JSONL,
                                {
                                    "chunk_id": chunk_id,
                                    "chapter_no": chapter_no,
                                    "status": "ok",
                                    "id": _id,
                                    "input": src_item,
                                    "output": normalized2,
                                },
                            )

                candidates.extend(fixed)

                # Optional second-pass verification (LLM)
                keep_mask: list[bool] | None = None
                if VERIFY_WITH_LLM and candidates:
                    keep_mask = [True] * len(candidates)
                    for batch_start in range(0, len(candidates), VERIFY_BATCH_SIZE):
                        batch = candidates[batch_start : batch_start + VERIFY_BATCH_SIZE]
                        vm = _build_verify_messages(
                            chapter_no=chapter_no,
                            chunk_id=chunk_id,
                            chunk_text=chunk_text,
                            items=[
                                {
                                    "id": batch_start + i,
                                    "head": x.get("head"),
                                    "relation": x.get("relation"),
                                    "tail": x.get("tail"),
                                    "head_type": x.get("head_type"),
                                    "tail_type": x.get("tail_type"),
                                    "head_cate": x.get("head_cate"),
                                    "tail_cate": x.get("tail_cate"),
                                    "evidence": x.get("evidence"),
                                }
                                for i, x in enumerate(batch)
                            ],
                        )
                        try:
                            vd = _parse_llm_json_with_retry(vm)
                            if not isinstance(vd, list):
                                raise ValueError("verify response is not a list")

                            id_to_decision: dict[int, dict[str, Any]] = {}
                            for d in vd:
                                if not isinstance(d, dict) or "id" not in d:
                                    continue
                                try:
                                    did = int(d.get("id"))
                                except Exception:
                                    continue
                                id_to_decision[did] = d

                            for i, candidate in enumerate(batch):
                                global_idx = batch_start + i
                                decision = id_to_decision.get(global_idx)
                                keep = keep_mask[global_idx]
                                reason = "missing_decision"
                                if isinstance(decision, dict):
                                    keep = bool(decision.get("keep") is True)
                                    reason = _clean_text(decision.get("reason"))[:50] or ""
                                keep_mask[global_idx] = keep
                                _append_jsonl(
                                    VERIFY_LOG_JSONL,
                                    {
                                        "chunk_id": chunk_id,
                                        "chapter_no": chapter_no,
                                        "id": global_idx,
                                        "keep": keep,
                                        "reason": reason,
                                        "candidate": candidate,
                                    },
                                )
                        except Exception as e:
                            _append_jsonl(
                                VERIFY_LOG_JSONL,
                                {
                                    "chunk_id": chunk_id,
                                    "chapter_no": chapter_no,
                                    "batch_start": batch_start,
                                    "batch_size": len(batch),
                                    "error": str(e),
                                    "status": "verify_error_keep_default",
                                },
                            )

                # Emit after optional verification + dedup (chunk-level)
                for i, item in enumerate(candidates):
                    if keep_mask is not None:
                        if not keep_mask[i]:
                            verified_dropped += 1
                            continue
                        verified_kept += 1

                    k = _make_key(item)
                    if not (k.head and k.tail and k.relation):
                        continue
                    s = _score_record(item)
                    prev = existing_best_scores.get(k)
                    if prev is not None and s <= prev:
                        continue
                    existing_best_scores[k] = s
                    _append_jsonl(OUT_RELATIONS_JSONL, item)
                    emitted_count += 1

                extract_ok = True
            except Exception as e:
                err = str(e)

            elapsed = time.time() - started
            _append_jsonl(
                EXTRACT_LOG_JSONL,
                {
                    "chunk_id": chunk_id,
                    "chapter_no": chapter_no,
                    "chapter_title": chapter_title,
                    "run_signature": RUN_SIGNATURE,
                    "status": "ok" if extract_ok else "error",
                    "raw_relations": raw_rel_count,
                    "passes": pass_summaries,
                    "uncertain_relations": len(uncertain_records),
                    "normalized_emitted": normalize_count,
                    "verified_kept": verified_kept,
                    "verified_dropped": verified_dropped,
                    "emitted": emitted_count,
                    "elapsed_sec": round(elapsed, 3),
                    "error": err,
                },
            )
            if extract_ok:
                done_chunks.add(chunk_id)

    # Export a deduped 7-column txt for downstream graph building
    _export_7col_txt()


if __name__ == "__main__":
    main()
