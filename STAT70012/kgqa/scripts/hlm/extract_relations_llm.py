"""
Extract relations from 红楼梦 txt via vLLM (OpenAI-compatible) and write JSONL.

Design goals:
- Simple, restartable, and safe (does not overwrite raw_data).
- Chunk size controlled by a rough char->token estimate: 1 char ~= 0.66 token.
- Source-level dedup: do not emit relations already in kgqa/raw_data/relation.txt.
- Hybrid normalization: deterministic rules first; use LLM only for uncertain records.

Output JSONL format (one relation per line):
{
  "head": "...",
  "relation": "...",
  "tail": "...",
  "head_type": "Person|Place|Org|Item|Event|Text",
  "tail_type": "Person|Place|Org|Item|Event|Text",
  "head_cate": "贾家荣国府|贾家宁国府|王家|史家|薛家|林家|其他",
  "tail_cate": "...",
  "chapter_no": 12,
  "chunk_id": "12-0003",
  "evidence": "..."
}
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
BOOK_PATH = Path("kgqa/data/input/hlm.txt")
BASE_RELATION_PATH = Path("kgqa/raw_data/relation.txt")

OUTPUT_DIR = Path("kgqa/data/output")
OUT_RELATIONS_JSONL = OUTPUT_DIR / "hlm_new_relations.jsonl"
EXTRACT_LOG_JSONL = OUTPUT_DIR / "hlm_extract_log.jsonl"
NORMALIZE_LOG_JSONL = OUTPUT_DIR / "hlm_normalize_log.jsonl"
VERIFY_LOG_JSONL = OUTPUT_DIR / "hlm_verify_log.jsonl"

# Quick test switches (set to None to disable limits)
CHAPTER_NO_START: int | None = None  # e.g. 1
CHAPTER_NO_END: int | None = None  # e.g. 5 (inclusive)
MAX_CHUNKS_PER_CHAPTER: int | None = None  # e.g. 2

VLLM_BASE_URL = "http://157.66.255.40:8000/v1"
VLLM_API_KEY = "none"  # set to "none" to disable Authorization header
VLLM_MODEL = "summary"

# Chunk sizing
TOKEN_PER_CHAR = 0.66
CHUNK_TOKEN_LIMIT_EST = 6500

# Retries
JSON_RETRY_MAX = 3  # if response is not valid JSON, re-request up to 3 times
HTTP_RETRY_MAX = 3
HTTP_RETRY_SLEEP_SEC = 2.0

# vLLM sampling params (user-provided "best" config)
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0.0

# LLM normalization batch size
NORMALIZE_BATCH_SIZE = 20

# Second-pass LLM verification (to improve precision)
VERIFY_WITH_LLM = True
VERIFY_BATCH_SIZE = 5


TYPE_ENUM = ["Person", "Place", "Org", "Item", "Event", "Text"]
CATE_ENUM = ["贾家荣国府", "贾家宁国府", "王家", "史家", "薛家", "林家", "其他"]

# Expanded relation whitelist (first pass)
REL_WHITELIST = [
    # person-person / identity
    "父亲",
    "母亲",
    "儿子",
    "女儿",
    "丈夫",
    "妻",
    "兄弟",
    "姐妹",
    "哥哥",
    "姐姐",
    "妹妹",
    "弟弟",
    "丫环",
    "丫头",
    "主人",
    "仆人",
    "朋友",
    "暧昧",
    "师徒",
    "雇佣",
    "养父",
    "养子",
    "婆婆",
    "嫂子",
    "孙子",
    "孙女",
    "岳母",
    "表兄妹",
    # cross-type (expanded)
    "居住于",
    "位于",
    "拥有",
    "赠与",
    "遗失",
    "发现",
    "使用",
    "发生在",
    "发生于",
    "参与",
    "主导",
    "受害",
    "作者",
    "作诗",
    "作词",
    "题词",
    "题赠对象",
]


# Optional: reuse existing relation normalization mapping
try:
    from neo_db.config import similar_words
except Exception:
    similar_words = {}


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
            line = line.strip()
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


def _norm_relation(rel: str) -> str:
    rel = _clean_text(rel)
    if not rel:
        return rel
    return similar_words.get(rel, rel)


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


def _looks_like_non_entity(name: str) -> bool:
    if not name:
        return True
    if len(name) > 30:
        return True
    if re.fullmatch(r"[0-9]+", name):
        return True
    if re.fullmatch(r"[\W_]+", name):
        return True
    return False


_TITLE_WORDS = [
    "老太太",
    "老爷",
    "太太",
    "小姐",
    "姑娘",
    "丫头",
    "丫鬟",
    "二爷",
    "三爷",
    "大爷",
    "奶奶",
    "嬷嬷",
    "婆子",
]


def _looks_like_title(name: str) -> bool:
    if len(name) > 6:
        return False
    return any(w == name or name.endswith(w) for w in _TITLE_WORDS)


def _make_key(obj: dict[str, Any]) -> RelationKey:
    return RelationKey(
        head=_clean_text(obj.get("head", "")),
        relation=_norm_relation(obj.get("relation", "")),
        tail=_clean_text(obj.get("tail", "")),
        head_type=_norm_type(obj.get("head_type", "")),
        tail_type=_norm_type(obj.get("tail_type", "")),
    )


def _load_servant_hints() -> tuple[set[str], set[str]]:
    """
    Infer canonical direction for 丫环/丫头 from the base dataset:
    - maid_set: entities appearing as head of 丫环/丫头
    - master_set: entities appearing as tail of 丫环/丫头
    """
    maid_set: set[str] = set()
    master_set: set[str] = set()
    if not BASE_RELATION_PATH.exists():
        return maid_set, master_set
    with BASE_RELATION_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            head, tail, rel = _clean_text(parts[0]), _clean_text(parts[1]), _norm_relation(parts[2])
            if rel in ("丫环", "丫头"):
                if head:
                    maid_set.add(head)
                if tail:
                    master_set.add(tail)
    return maid_set, master_set


def _swap_head_tail(item: dict[str, Any]) -> dict[str, Any]:
    swapped = dict(item)
    swapped["head"], swapped["tail"] = item.get("tail"), item.get("head")
    swapped["head_type"], swapped["tail_type"] = item.get("tail_type"), item.get("head_type")
    swapped["head_cate"], swapped["tail_cate"] = item.get("tail_cate"), item.get("head_cate")
    return swapped


def _canonicalize_servant_direction(
    item: dict[str, Any],
    *,
    maid_hint_set: set[str],
    master_hint_set: set[str],
    existing_keys: set[RelationKey],
) -> dict[str, Any]:
    """
    Unify direction for 丫环/丫头 to: maid -> master.
    If current direction looks reversed according to hint sets or existing keys, swap it.
    """
    rel = _norm_relation(item.get("relation", ""))
    if rel not in ("丫环", "丫头"):
        return item

    head = _clean_text(item.get("head", ""))
    tail = _clean_text(item.get("tail", ""))
    if not head or not tail:
        return item

    cur_key = _make_key(item)
    rev_key = RelationKey(
        head=tail,
        relation=rel,
        tail=head,
        head_type=_norm_type(item.get("tail_type", "")),
        tail_type=_norm_type(item.get("head_type", "")),
    )

    # If reverse already exists in the base graph (or extracted), prefer swapping to match.
    if rev_key in existing_keys and cur_key not in existing_keys:
        return _swap_head_tail(item)

    # Hint-based swap: prefer having known/likely maid on head side.
    head_is_maid = head in maid_hint_set
    tail_is_maid = tail in maid_hint_set
    head_is_master = head in master_hint_set
    tail_is_master = tail in master_hint_set

    if tail_is_maid and not head_is_maid:
        return _swap_head_tail(item)
    if head_is_master and not tail_is_master and not head_is_maid:
        return _swap_head_tail(item)

    return item


def _load_existing_keys() -> set[RelationKey]:
    keys: set[RelationKey] = set()

    # base relations (raw_data)
    if BASE_RELATION_PATH.exists():
        with BASE_RELATION_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                if len(parts) < 5:
                    continue
                head, tail, rel = parts[0], parts[1], parts[2]
                head_type = parts[5] if len(parts) >= 6 and parts[5].strip() else "Person"
                tail_type = parts[6] if len(parts) >= 7 and parts[6].strip() else "Person"
                keys.add(
                    RelationKey(
                        head=_clean_text(head),
                        relation=_norm_relation(rel),
                        tail=_clean_text(tail),
                        head_type=_norm_type(head_type),
                        tail_type=_norm_type(tail_type),
                    )
                )

    # already extracted output (restart-safe)
    for obj in _jsonl_iter(OUT_RELATIONS_JSONL):
        keys.add(_make_key(obj))

    return keys


def _load_done_chunks() -> set[str]:
    done: set[str] = set()
    for obj in _jsonl_iter(EXTRACT_LOG_JSONL):
        if obj.get("status") == "ok" and obj.get("chunk_id"):
            done.add(str(obj["chunk_id"]))
    return done


def _http_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY.strip().lower() != "none" and VLLM_API_KEY.strip():
        headers["Authorization"] = f"Bearer {VLLM_API_KEY.strip()}"
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)


def _call_chat_json(messages: list[dict[str, str]]) -> str:
    url = VLLM_BASE_URL.rstrip("/") + "/chat/completions"

    payload: dict[str, Any] = {
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


def _build_extract_messages(*, chapter_no: int, chunk_id: str, text: str) -> list[dict[str, str]]:
    rel_list = "、".join(REL_WHITELIST)
    type_list = "/".join(TYPE_ENUM)
    cate_list = "、".join(CATE_ENUM)
    relation_rules = (
        "关系语义/方向/类型硬约束（不满足就不要输出该条）：\n"
        "- 父亲/母亲：Person -> Person\n"
        "- 儿子/女儿：Person -> Person\n"
        "- 丈夫/妻：Person -> Person\n"
        "- 兄弟/姐妹/哥哥/姐姐/妹妹/弟弟/朋友/暧昧/师徒/雇佣/养父/养子/婆婆/嫂子/孙子/孙女/岳母/表兄妹：Person -> Person\n"
        "- 丫环/丫头/主人/仆人：Person -> Person（证据中需出现“丫头/丫鬟/小厮/仆/奴/侍/使唤/服侍/主子”等明确触发词之一）\n"
        "- 居住于：Person -> Place（证据中需出现“住/居/寄居/住在/居于/寓于/在…住/搬至/暂居”等明确用词之一）\n"
        "- 位于：Place/Org/Event/Text/Item -> Place（证据中需出现“在/位于/坐落/门前/旁/内/外”等明确定位表达）\n"
        "- 拥有/赠与/遗失/发现/使用：Person -> Item（或 Item -> Person 仅当原文明示“归某人所有/为某人之物”，否则不要倒置）\n"
        "- 发生在/发生于：Event -> Place（必须是事件作为 head；人物/地点不能用这两个关系）\n"
        "- 参与/主导/受害：Person -> Event（证据需能直接看出参与/主导/受害，不要推理）\n"
        "- 作者：Person -> Text（必须明确是“某人作/题/写/撰”）\n"
        "- 作诗/作词/题词：Person -> Text（必须明确“作诗/赋诗/诗云/词曰/填词/题词/题联/题额/题着”等；Text 可用标题/首句做名称）\n"
        "- 题赠对象：Text -> Person（必须明确“题赠/赠某人/为某人而作”）\n"
        "\n"
        "关系名规范化（必须输出到白名单中的规范名，不要自造）：\n"
        "- “妻子/夫人/内人/正室/侧室/老公/相公”等：映射为 丈夫/妻（按方向选择合适的一条）\n"
        "- “住在/居于/寄居/暂住/寓于”等：映射为 居住于\n"
        "- “发生在/发生于/在…发生”等：映射为 发生在 或 发生于（语义等价，任选其一，但必须 Event -> Place）\n"
        "- “写诗/赋诗/诗云/诗曰/作诗”等：映射为 作诗\n"
        "- “作词/填词/词云/词曰/作曲”等：映射为 作词\n"
        "- “题词/题联/题对/题额/题着”等：映射为 题词\n"
    )
    system = (
        "你是信息抽取助手。你的任务是从给定文本中抽取关系三元组，并严格按要求输出。"
        "只能输出合法 JSON（不要代码块、不要解释、不要多余文字）。"
    )
    user = (
        f"请从下面《红楼梦》文本中抽取关系，输出严格 JSON。\n"
        f"- chunk_id: {chunk_id}\n"
        f"- chapter_no: {chapter_no}\n"
        f"- 允许的 relation（必须从中选择）：{rel_list}\n"
        f"- 允许的 type（必须从中选择）：{type_list}\n"
        f"- 允许的 cate（必须从中选择，否则填“其他”）：{cate_list}\n"
        "\n"
        f"{relation_rules}\n"
        "\n"
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
        '      "evidence": "从原文摘取的一句或半句短证据(<=80字)"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "要求：\n"
        "- 只抽取文本中明确表达的关系，不要推理/脑补。\n"
        "- 必须保守：不确定就不输出该条，宁可少不要错。\n"
        "- head 与 tail 必须是具体姓名/专名；不要用代词或泛称（他/她/我们/老太太/二爷/姑娘/丫头等）。若无法消歧，直接不要输出。\n"
        "- evidence 必须是原文中的连续子串，并且 evidence 中必须同时出现 head 与 tail（否则不要输出该条）。\n"
        "- 不要把人物当“发生于/发生在”的 head；不要把地点当“发生于/发生在”的 head。\n"
        "- 同一条关系不要重复输出。\n"
        "- relations 可以为空数组。\n"
        "\n"
        "文本如下：\n"
        f"{text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_normalize_messages(*, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    rel_list = "、".join(REL_WHITELIST)
    type_list = "/".join(TYPE_ENUM)
    cate_list = "、".join(CATE_ENUM)
    system = (
        "你是数据规范化助手。只输出合法 JSON（不要代码块、不要解释、不要多余文字）。"
    )
    user = (
        "请将下面抽取结果规范化（必要时修复字段），严格遵守枚举约束。\n"
        f"- relation 必须从：{rel_list}\n"
        f"- type 必须从：{type_list}\n"
        f"- cate 必须从：{cate_list}（否则填“其他”）\n"
        "\n"
        "规范化要求（不满足就输出 null）：\n"
        "- relation 必须映射到白名单的规范名；例如“妻子/夫人/内人/正室/老公”等需映射为 丈夫/妻（按方向选）。\n"
        "- “写诗/赋诗/诗云/诗曰/作诗”映射为 作诗；“作词/填词/词云/词曰/作曲”映射为 作词；“题词/题联/题对/题额/题着”映射为 题词。\n"
        "- head_type/tail_type 必须选枚举值。\n"
        "- head/tail 不能是泛称或代词（如 老太太/二爷/她/他），无法指向具体实体则输出 null。\n"
        "- evidence 必须保留原样或更短，并且尽量包含 head 与 tail。\n"
        "\n"
        "输入是一个 JSON 数组，每个元素包含 head/tail/relation/.../evidence。\n"
        "输出也必须是一个 JSON 数组，长度与输入相同：\n"
        "- 若能规范化，输出对象：{head,relation,tail,head_type,tail_type,head_cate,tail_cate,evidence}\n"
        "- 若无法可靠规范化（例如称谓无法指向具体人名），输出 null\n"
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
    """
    Second-pass verification: keep only relations explicitly supported by the text.
    Output must be a JSON array with same length as input:
      [{"keep": true/false, "reason": "..."} ...]
    """
    system = (
        "你是关系校验助手。你的任务是对候选关系逐条验真。"
        "只能输出合法 JSON（不要代码块、不要解释、不要多余文字）。"
    )
    rel_list = "、".join(REL_WHITELIST)
    type_list = "/".join(TYPE_ENUM)
    verify_rules = (
        "校验规则（按本项目关系定义判断，严格但允许你在整段原文中找证据）：\n"
        "- 只能在原文中找到明确措辞时 keep=true；仅凭同段共现/座次/常识推断，一律 keep=false。\n"
        "- 必须方向一致：head -[relation]-> tail。\n"
        "- 丫环/丫头：必须能从原文直接看出“head 是 tail 的丫鬟/丫头/婢/与了 tail”等。\n"
        "- 主人/仆人/雇佣：必须有明确主仆/雇聘/门子/小厮/下人/西宾等措辞支撑；相交/邀饮/帮忙不算。\n"
        "- 拥有：更偏向“物的占有”（钱、玉、锁、匾额等）；若 tail 明显是地点(Place)通常不成立。\n"
        "- 作诗/作词/题词：必须明确“某人作/赋/题/写”与具体诗词/题词文本的对应；仅出现一段诗词但没指明作者，keep=false。\n"
        "- 嫂子/哥哥/姐姐/妹妹/弟弟/孙子/孙女等：必须有明确亲属关系表达；仅凭称呼默认不成立。\n"
        "- 若 head/tail 是泛称或代词（老太太/二爷/姑娘/她/他），无法落到具体专名，则 keep=false。\n"
    )
    user = (
        f"请根据原文对候选关系逐条验真。\n"
        f"- chunk_id: {chunk_id}\n"
        f"- chapter_no: {chapter_no}\n"
        f"- 候选 relation 来自白名单：{rel_list}\n"
        f"- type 枚举：{type_list}\n"
        "\n"
        f"{verify_rules}\n"
        "\n"
        "判定标准（非常严格）：\n"
        "- 只有当原文明确支持该 head -[relation]-> tail（方向一致、语义一致）时，keep=true。\n"
        "- 仅凭同段共现、称谓/代词、座次、你的常识推断，都算不支持（keep=false）。\n"
        "- 若无法从原文直接确定 head/tail 指代（例如“二爷/老太太/她/他”），keep=false。\n"
        "\n"
        "输出要求：\n"
        "- 只输出 JSON 数组，必须覆盖输入中的每个 id，且每个 id 只出现一次。\n"
        '- 每个元素是 {"id": 0, "keep": true/false, "reason": "...(<=20字)"}\n'
        "\n"
        "原文：\n"
        f"{chunk_text}\n"
        "\n"
        "候选关系(JSON数组)：\n"
        f"{json.dumps(items, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_llm_json_with_retry(messages: list[dict[str, str]]) -> dict[str, Any] | list[Any]:
    last_raw: str = ""
    for _ in range(JSON_RETRY_MAX):
        raw = _call_chat_json(messages)
        last_raw = raw
        try:
            return json.loads(raw)
        except Exception:
            continue
    raise ValueError(f"Model did not return valid JSON after {JSON_RETRY_MAX} tries: {last_raw[:500]}")


def _normalize_rule_first(obj: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    """
    Returns: (normalized_obj_or_none, needs_llm_fix)
    """
    head = _clean_text(obj.get("head"))
    tail = _clean_text(obj.get("tail"))
    rel = _norm_relation(obj.get("relation"))
    head_type = _norm_type(obj.get("head_type"))
    tail_type = _norm_type(obj.get("tail_type"))
    head_cate = _norm_cate(obj.get("head_cate"))
    tail_cate = _norm_cate(obj.get("tail_cate"))
    evidence = _clean_text(obj.get("evidence"))

    if _looks_like_non_entity(head) or _looks_like_non_entity(tail):
        return None, False

    needs_fix = False
    if rel not in REL_WHITELIST:
        needs_fix = True

    if _looks_like_title(head) or _looks_like_title(tail):
        needs_fix = True

    out = {
        "head": head,
        "relation": rel,
        "tail": tail,
        "head_type": head_type,
        "tail_type": tail_type,
        "head_cate": head_cate,
        "tail_cate": tail_cate,
        "evidence": evidence[:200],
    }
    return out, needs_fix


_CHAPTER_HEADER_RE = re.compile(r"^第[一二三四五六七八九十百千0-9]+卷\(\d+--\d+章\).+")


def _preprocess_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for i, line in enumerate(lines):
        s = line.rstrip("\n")
        if i < 50:
            low = s.lower()
            if "downbank" in low:
                continue
            if s.strip() and set(s.strip()) == {"-"}:
                continue
        cleaned.append(s)
    return cleaned


def _split_chapters(lines: list[str]) -> list[tuple[int, str, str]]:
    """
    Returns list of (chapter_no, chapter_title, chapter_text).
    chapter_no starts at 1; missing first header is treated as chapter 1.
    """
    header_positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if _CHAPTER_HEADER_RE.match(s):
            header_positions.append((idx, s))

    chapters: list[tuple[int, str, str]] = []

    # Chapter 1: before first header
    if header_positions:
        first_idx, _ = header_positions[0]
        text1 = "\n".join(lines[:first_idx]).strip()
        chapters.append((1, "第一回(缺失标题)", text1))
    else:
        chapters.append((1, "第一回(无标题)", "\n".join(lines).strip()))
        return chapters

    # Remaining chapters: assume headers correspond to chapter 2..N in order
    for i, (start_idx, title) in enumerate(header_positions):
        chapter_no = i + 2
        end_idx = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(lines)
        body = "\n".join(lines[start_idx + 1 : end_idx]).strip()
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
        # try to break at punctuation near the end
        cut = None
        for m in re.finditer(r"[。！？；]", window):
            cut = m.end()
        if cut is None or cut < max(50, int(limit_chars * 0.5)):
            cut = len(window)
        out.append(window[:cut].strip())
        start += cut
    return [x for x in out if x]


def _build_chunks(chapter_text: str) -> list[str]:
    # Keep a small safety margin: we also add newlines between paragraphs,
    # and the char->token estimate is rough.
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
        # joining uses '\n' between paragraphs
        new_chars = buf_chars + (1 if buf else 0) + len(p)
        if buf and _estimate_tokens(new_chars) > CHUNK_TOKEN_LIMIT_EST:
            chunks.append("\n".join(buf).strip())
            buf = []
            buf_chars = 0
        buf.append(p)
        buf_chars = new_chars
    if buf:
        chunks.append("\n".join(buf).strip())
    # final hard cap (just in case)
    capped: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if _estimate_tokens(len(c)) <= CHUNK_TOKEN_LIMIT_EST:
            capped.append(c)
            continue
        capped.extend(_split_long_para(c, limit_chars))
    return [c for c in capped if c]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_keys = _load_existing_keys()
    maid_hints, master_hints = _load_servant_hints()
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
                messages = _build_extract_messages(
                    chapter_no=chapter_no, chunk_id=chunk_id, text=chunk_text
                )
                data = _parse_llm_json_with_retry(messages)
                if not isinstance(data, dict):
                    raise ValueError("extract response is not an object")
                rels = data.get("relations", [])
                if not isinstance(rels, list):
                    raise ValueError("relations is not a list")
                raw_rel_count = len(rels)

                candidates: list[dict[str, Any]] = []
                for r in rels:
                    if not isinstance(r, dict):
                        continue
                    normalized, needs_fix = _normalize_rule_first(r)
                    if normalized is None:
                        continue
                    normalized = _canonicalize_servant_direction(
                        normalized,
                        maid_hint_set=maid_hints,
                        master_hint_set=master_hints,
                        existing_keys=existing_keys,
                    )
                    normalized["chapter_no"] = chapter_no
                    normalized["chunk_id"] = chunk_id
                    if needs_fix:
                        uncertain_records.append(normalized)
                        continue
                    candidates.append(normalized)

                # LLM normalization for uncertain items
                fixed: list[dict[str, Any]] = []
                if uncertain_records:
                    for batch_start in range(0, len(uncertain_records), NORMALIZE_BATCH_SIZE):
                        batch = uncertain_records[
                            batch_start : batch_start + NORMALIZE_BATCH_SIZE
                        ]
                        nm = _build_normalize_messages(items=batch)
                        nm_data = _parse_llm_json_with_retry(nm)
                        if not isinstance(nm_data, list):
                            raise ValueError("normalize response is not a list")
                        if len(nm_data) != len(batch):
                            raise ValueError("normalize response length mismatch")
                        for src_item, out_item in zip(batch, nm_data):
                            if out_item is None:
                                _append_jsonl(
                                    NORMALIZE_LOG_JSONL,
                                    {
                                        "chunk_id": chunk_id,
                                        "chapter_no": chapter_no,
                                        "status": "dropped",
                                        "input": src_item,
                                    },
                                )
                                continue
                            if not isinstance(out_item, dict):
                                continue
                            n2, needs_fix2 = _normalize_rule_first(out_item)
                            if n2 is None or needs_fix2:
                                _append_jsonl(
                                    NORMALIZE_LOG_JSONL,
                                    {
                                        "chunk_id": chunk_id,
                                        "chapter_no": chapter_no,
                                        "status": "invalid_after_normalize",
                                        "input": src_item,
                                        "output": out_item,
                                    },
                                )
                                continue
                            # carry metadata
                            n2 = _canonicalize_servant_direction(
                                n2,
                                maid_hint_set=maid_hints,
                                master_hint_set=master_hints,
                                existing_keys=existing_keys,
                            )
                            n2["chapter_no"] = chapter_no
                            n2["chunk_id"] = chunk_id
                            fixed.append(n2)

                candidates.extend(fixed)
                normalize_count = len(fixed)

                # Second-pass verification (optional)
                keep_mask: list[bool] | None = None
                if VERIFY_WITH_LLM and candidates:
                    # default keep-all; verification can only veto (set keep=false).
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

                            # Prefer id-based mapping (more robust than relying on length).
                            id_to_decision: dict[int, dict[str, Any]] = {}
                            seq_decisions: list[dict[str, Any]] = []
                            for d in vd:
                                if not isinstance(d, dict):
                                    continue
                                if "id" in d:
                                    try:
                                        did = int(d.get("id"))
                                    except Exception:
                                        continue
                                    id_to_decision[did] = d
                                else:
                                    seq_decisions.append(d)

                            for i, candidate in enumerate(batch):
                                global_idx = batch_start + i
                                decision = id_to_decision.get(global_idx)
                                if decision is None and i < len(seq_decisions):
                                    decision = seq_decisions[i]

                                # If missing, keep default (True) but log.
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
                            # Verification failure: keep default (True) for this batch, but log the error.
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

                # Emit after optional verification + dedup
                for i, item in enumerate(candidates):
                    if keep_mask is not None:
                        if not keep_mask[i]:
                            verified_dropped += 1
                            continue
                        verified_kept += 1

                    key = _make_key(item)
                    if key in existing_keys:
                        continue
                    existing_keys.add(key)
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
                    "status": "ok" if extract_ok else "error",
                    "raw_relations": raw_rel_count,
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


if __name__ == "__main__":
    main()
