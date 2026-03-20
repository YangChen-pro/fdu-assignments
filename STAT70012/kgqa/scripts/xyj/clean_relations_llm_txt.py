#!/usr/bin/env python3
"""
Clean LLM-extracted 7-column relation TXT for 《西游记》.

Input format (comma-separated, 7 columns):
  head,tail,relation,head_cate,tail_cate,head_type,tail_type

Outputs:
  - cleaned 7-col txt (same format, for downstream KGQA)
  - optional audit TSV with raw->clean mapping + notes (to avoid information loss)

Design principles:
  - Prefer normalization/fixing over dropping rows.
  - Only drop obviously broken rows (e.g., head==tail self-loop with non-reflexive relation).
  - Keep changes auditable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TYPE_ENUM = {"Person", "Place", "Org", "Item", "Event", "Text"}
CATE_ENUM = {
    "取经团队",
    "佛门（灵山/观音体系）",
    "天庭（天宫/星宿/天兵天将）",
    "道门仙家（三清/老君/洞天福地）",
    "地府冥界（十殿/鬼差）",
    "龙宫水族（四海/水府）",
    "人间大唐（朝廷）",
    "人间诸国（西域各国/地方政权）",
    "地方神祇（土地/山神/城隍/河神）",
    "妖魔势力（洞府妖王）",
    "其他",
}


DEFAULT_ALIAS_MAP: dict[str, str] = {
    # 取经团队核心别名（只做完全匹配替换）
    "悟空": "孙悟空",
    "行者": "孙悟空",
    "孙行者": "孙悟空",
    "齐天大圣": "孙悟空",
    "美猴王": "孙悟空",
    "弼马温": "孙悟空",
    "石猴": "孙悟空",
    # extra high-frequency surface forms observed in LLM output
    "孙大圣": "孙悟空",
    "大圣": "孙悟空",
    "唐僧": "唐三藏",
    "三藏": "唐三藏",
    "玄奘": "唐三藏",
    "御弟": "唐三藏",
    "八戒": "猪八戒",
    "悟能": "猪八戒",
    "猪悟能": "猪八戒",
    "天蓬元帅": "猪八戒",
    "沙僧": "沙悟净",
    "悟净": "沙悟净",
    "沙和尚": "沙悟净",
    "白龙马": "小白龙",
    # 常见神佛别名
    "观音": "观音菩萨",
    "观世音": "观音菩萨",
    "如来": "如来佛祖",
    "玉帝": "玉皇大帝",
    "玉皇大天尊": "玉皇大帝",
    "玉皇大天尊玄穹高上帝": "玉皇大帝",
    "老君": "太上老君",
    # small standardizations
    "太宗": "唐太宗",
    "八金刚": "八大金刚",
}


RELATION_MAP: dict[str, str] = {
    # travel / location
    "抵达": "到达",
    "送至东土": "送至",
    "奉佛旨送行": "送行",
    "奉命送行": "送行",
    "引往": "带往",
    # appoint
    "传旨封为": "封为",
    # item offering
    "奉命献上": "献上",
}


LOCATIVE_RELATIONS = {
    "位于",
    "居住于",
    "前往",
    "到达",
    "进入",
    "抵达",
    "返回",
    "离开",
    "出发",
    "驻扎",
    "送至",
    "带往",
}


PERSON_TITLE_SUFFIX = (
    "王",
    "公",
    "君",
    "帝",
    "佛",
    "菩萨",
    "罗汉",
    "将军",
    "天尊",
    "祖师",
    "星君",
    "娘娘",
    "大仙",
    "真人",
    "老君",
    "太子",
    "龙王",
)


PLACE_HINT_CHARS = set("山洞寺宫国城府岭河海塔关殿园观庙庄寨桥井州谷泉台馆楼峰")
TEXT_HINT_CHARS = set("经书咒卷令符诏牒文旨檄表")
ITEM_HINT_CHARS = set("棒刀剑杖锤珠铃环袍甲冠靴履扇鞭叉索网袋瓶葫芦炉钯")


def _clean_text(s: str) -> str:
    return (s or "").strip().replace("\u3000", " ")


def _norm_type(t: str) -> str:
    t = _clean_text(t)
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


def _norm_entity(name: str, alias_map: dict[str, str]) -> str:
    name = _clean_text(name)
    if not name:
        return ""
    return alias_map.get(name, name)


def _norm_relation(rel: str) -> str:
    rel = _clean_text(rel)
    if not rel:
        return ""
    rel = RELATION_MAP.get(rel, rel)
    return rel


def _guess_type(name: str) -> str | None:
    name = _clean_text(name)
    if not name:
        return None
    if any(ch in name for ch in PLACE_HINT_CHARS):
        return "Place"
    if any(ch in name for ch in TEXT_HINT_CHARS):
        return "Text"
    if any(ch in name for ch in ITEM_HINT_CHARS):
        return "Item"
    if name.endswith(PERSON_TITLE_SUFFIX):
        return "Person"
    return None


def _looks_like_place(name: str) -> bool:
    g = _guess_type(name)
    return g == "Place"


def _extract_embedded_place_from_relation(rel: str) -> tuple[str, str] | None:
    """
    Handle a narrow set of relations that embed a place:
      - 封压于五行山
      - 被镇压于五行山
      - 镇压于五行山
    Returns (action, place).
    """
    rel = _clean_text(rel)
    for prefix in ("被镇压于", "封压于", "镇压于"):
        if rel.startswith(prefix) and len(rel) > len(prefix):
            place = rel[len(prefix) :].strip()
            action = prefix[:-1]  # drop the trailing '于' for action label
            return (action, place)
    return None


@dataclass(frozen=True)
class Row:
    head: str
    tail: str
    relation: str
    head_cate: str
    tail_cate: str
    head_type: str
    tail_type: str


def _read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 7:
            raise ValueError(f"{path}:{line_no}: expected 7 columns, got {len(cols)}: {line[:160]}")
        head, tail, rel, hc, tc, ht, tt = cols
        rows.append(
            Row(
                head=_clean_text(head),
                tail=_clean_text(tail),
                relation=_clean_text(rel),
                head_cate=_norm_cate(hc),
                tail_cate=_norm_cate(tc),
                head_type=_norm_type(ht),
                tail_type=_norm_type(tt),
            )
        )
    return rows


def _build_entity_modes(
    rows: Iterable[Row],
) -> tuple[dict[str, str], dict[str, float], dict[str, str]]:
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cate_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if r.head:
            type_counts[r.head][r.head_type] += 1
            cate_counts[r.head][r.head_cate] += 1
        if r.tail:
            type_counts[r.tail][r.tail_type] += 1
            cate_counts[r.tail][r.tail_cate] += 1

    type_mode: dict[str, str] = {}
    type_conf: dict[str, float] = {}
    cate_mode: dict[str, str] = {}
    for ent, cnt in type_counts.items():
        t, n = cnt.most_common(1)[0]
        total = sum(cnt.values())
        type_mode[ent] = t
        type_conf[ent] = (n / total) if total else 0.0
    for ent, cnt in cate_counts.items():
        # prefer non-'其他' if it dominates reasonably
        total = sum(cnt.values())
        best_non_other = [(c, n) for c, n in cnt.items() if c != "其他"]
        best_non_other.sort(key=lambda x: -x[1])
        if best_non_other:
            c, n = best_non_other[0]
            if (n / total) >= 0.60 or n >= 3:
                cate_mode[ent] = c
                continue
        c, _ = cnt.most_common(1)[0]
        cate_mode[ent] = c
    return type_mode, type_conf, cate_mode


def clean_rows(
    rows: list[Row],
    *,
    alias_map: dict[str, str],
) -> tuple[list[Row], list[dict[str, str]]]:
    audit: list[dict[str, str]] = []
    out: list[Row] = []

    # 1) alias normalize entities + relation mapping
    stage1: list[Row] = []
    for r in rows:
        head_raw, tail_raw, rel_raw = r.head, r.tail, r.relation
        head = _norm_entity(head_raw, alias_map)
        tail = _norm_entity(tail_raw, alias_map)
        rel = _norm_relation(rel_raw)

        stage1.append(
            Row(
                head=head,
                tail=tail,
                relation=rel,
                head_cate=r.head_cate,
                tail_cate=r.tail_cate,
                head_type=r.head_type,
                tail_type=r.tail_type,
            )
        )

        audit.append(
            {
                "head_raw": head_raw,
                "tail_raw": tail_raw,
                "relation_raw": rel_raw,
                "head": head,
                "tail": tail,
                "relation": rel,
                "note": "",
            }
        )

    # 2) compute per-entity mode (after alias merge) and use it to fill missing cate/types cautiously
    type_mode, type_conf, cate_mode = _build_entity_modes(stage1)

    # 3) apply fixes + optional place-splitting (adds rows, avoids dropping info)
    dropped_self_loops = 0
    for r, a in zip(stage1, audit, strict=True):
        notes: list[str] = []
        head, tail, rel = r.head, r.tail, r.relation
        head_type, tail_type = r.head_type, r.tail_type
        head_cate, tail_cate = r.head_cate, r.tail_cate

        # fill cate if '其他' but entity has stable non-other cate
        if head and head_cate == "其他":
            cm = cate_mode.get(head)
            if cm and cm != "其他":
                head_cate = cm
                notes.append("fill_head_cate")
        if tail and tail_cate == "其他":
            cm = cate_mode.get(tail)
            if cm and cm != "其他":
                tail_cate = cm
                notes.append("fill_tail_cate")

        # enforce tail_type=Place for locative relations
        if rel in LOCATIVE_RELATIONS and tail:
            if tail_type != "Place":
                tail_type = "Place"
                notes.append("force_tail_place_by_relation")

        # '位于' is typically Place->Place
        if rel == "位于" and head and head_type != "Place":
            head_type = "Place"
            notes.append("force_head_place_by_rel_位于")

        # high-confidence type alignment (reduces type fragmentation without dropping rows)
        # do this after relation-specific constraints.
        if head and head_type != "Place":
            if type_conf.get(head, 0.0) >= 0.80:
                mt = type_mode.get(head)
                if mt and mt != head_type:
                    head_type = mt
                    notes.append("align_head_type_to_mode_hi_conf")
        if tail and tail_type != "Place":
            if type_conf.get(tail, 0.0) >= 0.80:
                mt = type_mode.get(tail)
                if mt and mt != tail_type:
                    tail_type = mt
                    notes.append("align_tail_type_to_mode_hi_conf")

        # heuristic type fixes when looks obvious
        if head:
            g = _guess_type(head)
            if g and g != head_type and (type_mode.get(head, head_type) != head_type):
                # only change when the entity is commonly seen as another type
                head_type = type_mode.get(head, head_type)
                notes.append("align_head_type_to_mode")
        if tail:
            if rel not in LOCATIVE_RELATIONS:
                g = _guess_type(tail)
                if g and g != tail_type:
                    # For person-title suffix, it's usually safe to flip to Person
                    if g == "Person" and tail_type != "Person":
                        tail_type = "Person"
                        notes.append("guess_tail_person_by_suffix")

        # split embedded place relations (narrow, to avoid over-editing)
        embedded = _extract_embedded_place_from_relation(rel)
        if embedded:
            action, place = embedded
            if place and _looks_like_place(place):
                # Rewrite current row's relation to action (actor-target)
                rel = action
                notes.append("split_embedded_place_action")

                # Add a new locative edge: (target, place, 被镇压于 / 封压于 / 镇压于)
                # Choose locative label based on original prefix.
                loc_rel = "被镇压于" if a["relation_raw"].startswith("被镇压于") else "镇压于"
                if a["relation_raw"].startswith("封压于"):
                    loc_rel = "被镇压于"

                # If the original row is in passive voice (被镇压于), it likely describes head->place.
                # We preserve actor tail in another edge: (tail, head, 镇压) if plausible.
                if a["relation_raw"].startswith("被镇压于"):
                    # convert current edge to locative: (head, place, 被镇压于)
                    out.append(
                        Row(
                            head=head,
                            tail=place,
                            relation="被镇压于",
                            head_cate=head_cate,
                            tail_cate="其他",
                            head_type="Person" if head_type == "Person" else head_type,
                            tail_type="Place",
                        )
                    )
                    notes.append("rewrite_passive_to_locative")

                    # add inferred actor-action edge: (tail_actor, head, 镇压) if tail exists and looks like person
                    if tail:
                        out.append(
                            Row(
                                head=tail,
                                tail=head,
                                relation="镇压",
                                head_cate=tail_cate,
                                tail_cate=head_cate,
                                head_type=tail_type,
                                tail_type=head_type,
                            )
                        )
                        notes.append("add_inferred_actor_action")

                    # skip adding the original (now ambiguous) actor-target row
                    a["note"] = "|".join(notes)
                    continue

                # active voice row: keep actor-target row + add target-location row
                if tail:
                    out.append(
                        Row(
                            head=tail,
                            tail=place,
                            relation="被镇压于",
                            head_cate=tail_cate,
                            tail_cate="其他",
                            head_type=tail_type,
                            tail_type="Place",
                        )
                    )
                    notes.append("add_target_locative")

        # drop obvious broken self-loops (very rare, and typically extraction error)
        if head and tail and head == tail:
            dropped_self_loops += 1
            notes.append("drop_self_loop")
            a["note"] = "|".join(notes)
            continue

        out.append(
            Row(
                head=head,
                tail=tail,
                relation=rel,
                head_cate=head_cate,
                tail_cate=tail_cate,
                head_type=head_type,
                tail_type=tail_type,
            )
        )
        a["note"] = "|".join(notes)

    # 4) de-duplicate exact rows
    seen: set[Row] = set()
    deduped: list[Row] = []
    for r in out:
        if r in seen:
            continue
        seen.add(r)
        deduped.append(r)

    # store global audit stats in the audit footer row (kept as a normal record in TSV)
    audit.append(
        {
            "head_raw": "",
            "tail_raw": "",
            "relation_raw": "",
            "head": "",
            "tail": "",
            "relation": "",
            "note": f"__stats__:in={len(rows)} stage1={len(stage1)} out={len(out)} dedup={len(deduped)} dropped_self_loops={dropped_self_loops}",
        }
    )

    return deduped, audit


def _write_clean_txt(path: Path, rows: list[Row]) -> None:
    def csv_safe(v: str) -> str:
        return _clean_text(v).replace(",", "，").replace("\n", " ").replace("\r", " ")

    lines = [
        ",".join(
            [
                csv_safe(r.head),
                csv_safe(r.tail),
                csv_safe(r.relation),
                csv_safe(r.head_cate),
                csv_safe(r.tail_cate),
                csv_safe(r.head_type),
                csv_safe(r.tail_type),
            ]
        )
        for r in rows
        if r.head and r.tail and r.relation
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_audit_tsv(path: Path, audit: list[dict[str, str]]) -> None:
    cols = ["head_raw", "tail_raw", "relation_raw", "head", "tail", "relation", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(cols)]
    for r in audit:
        lines.append("\t".join((r.get(c, "") or "").replace("\t", " ").replace("\n", " ") for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(path: str, rows: list[Row]) -> None:
    rel_cnt = Counter(r.relation for r in rows)
    head_other = sum(1 for r in rows if r.head_cate == "其他")
    tail_other = sum(1 for r in rows if r.tail_cate == "其他")
    ent = set()
    for r in rows:
        ent.add(r.head)
        ent.add(r.tail)
    print(f"[{path}] rows={len(rows)} entities={len(ent)} unique_rel={len(rel_cnt)}")  # noqa: T201
    print(f"[{path}] head_cate=其他 {head_other} ({head_other/len(rows):.3f})")  # noqa: T201
    print(f"[{path}] tail_cate=其他 {tail_other} ({tail_other/len(rows):.3f})")  # noqa: T201
    print(f"[{path}] top_rel {rel_cnt.most_common(10)}")  # noqa: T201


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="kgqa/raw_data/xyj_relation_llm.txt",
        help="Input 7-col txt",
    )
    ap.add_argument(
        "--output",
        default="kgqa/raw_data/xyj_relation_llm.cleaned.txt",
        help="Cleaned 7-col txt",
    )
    ap.add_argument(
        "--audit",
        default="kgqa/raw_data/xyj_relation_llm.cleaned_audit.tsv",
        help="Audit TSV (raw->clean + notes)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    audit_path = Path(args.audit) if args.audit else None

    rows = _read_rows(in_path)
    _summarize(str(in_path), rows)

    cleaned, audit = clean_rows(rows, alias_map=DEFAULT_ALIAS_MAP)
    _write_clean_txt(out_path, cleaned)
    if audit_path is not None:
        _write_audit_tsv(audit_path, audit)

    _summarize(str(out_path), cleaned)
    print(f"wrote: {out_path}")  # noqa: T201
    if audit_path is not None:
        print(f"wrote: {audit_path}")  # noqa: T201


if __name__ == "__main__":
    main()
