"""
Merge newly extracted relations into kgqa/raw_data/relation.txt without duplicates.

Base file format (comma-separated, no header):
  v1 (5 cols): head,tail,relation,head_cate,tail_cate
  v2 (7+ cols): head,tail,relation,head_cate,tail_cate,head_type,tail_type,(...)

New file formats supported:
  - .txt/.csv: same as above (comma-separated; 5 or 7+ cols)
  - .jsonl: one JSON object per line describing a relation

Dedup key (normalized):
  (head, relation, tail, head_type, tail_type)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    # Optional: reuse relation normalization from the QA config if available.
    from neo_db.config import similar_words
except Exception:
    similar_words = {}


DEFAULT_CATE = "其他"
DEFAULT_TYPE = "Person"

# =========================
# Config (edit these)
# =========================
# Existing relation file to extend
BASE_RELATION_PATH = Path("kgqa/raw_data/relation.txt")

# New relations extracted from the book txt (required)
# - Supported: .jsonl or .txt/.csv
NEW_RELATIONS_PATH = Path("kgqa/data/output/hlm_new_relations.jsonl")

# Output path:
# - None means overwrite BASE_RELATION_PATH
# - If set to a directory, will write a file under it with the same name as BASE_RELATION_PATH.
OUTPUT_PATH: Path | None = Path("kgqa/data/output")

# Output format:
# - "v1": always 5 cols
# - "v2": always 7 cols
# - "mixed": v1 for Person-Person, else v2
EMIT_MODE = "mixed"  # "v1" | "v2" | "mixed"


@dataclass(frozen=True)
class RelationRow:
    head: str
    tail: str
    relation: str
    head_cate: str = DEFAULT_CATE
    tail_cate: str = DEFAULT_CATE
    head_type: str = DEFAULT_TYPE
    tail_type: str = DEFAULT_TYPE

    def key(self) -> tuple[str, str, str, str, str]:
        return (
            _norm_text(self.head),
            _norm_relation(self.relation),
            _norm_text(self.tail),
            _norm_type(self.head_type),
            _norm_type(self.tail_type),
        )

    def to_v1_line(self) -> str:
        return ",".join(
            [
                self.head,
                self.tail,
                self.relation,
                self.head_cate or DEFAULT_CATE,
                self.tail_cate or DEFAULT_CATE,
            ]
        )

    def to_v2_line(self) -> str:
        return ",".join(
            [
                self.head,
                self.tail,
                self.relation,
                self.head_cate or DEFAULT_CATE,
                self.tail_cate or DEFAULT_CATE,
                self.head_type or DEFAULT_TYPE,
                self.tail_type or DEFAULT_TYPE,
            ]
        )


def _norm_text(s: str) -> str:
    return (s or "").strip()


def _norm_type(t: str) -> str:
    t = (t or "").strip()
    return t or DEFAULT_TYPE


def _norm_relation(r: str) -> str:
    r = (r or "").strip()
    if not r:
        return r
    # Use config mapping where applicable (e.g. 爸爸 -> 父亲).
    return similar_words.get(r, r)


def _parse_csv_line(line: str) -> RelationRow | None:
    parts = [p.strip() for p in (line or "").strip().split(",")]
    if len(parts) < 3:
        return None
    if len(parts) < 5:
        return None

    head, tail, relation, head_cate, tail_cate = parts[:5]
    if not head or not tail or not relation:
        return None

    head_type = parts[5] if len(parts) >= 6 and parts[5].strip() else DEFAULT_TYPE
    tail_type = parts[6] if len(parts) >= 7 and parts[6].strip() else DEFAULT_TYPE
    return RelationRow(
        head=head,
        tail=tail,
        relation=_norm_relation(relation),
        head_cate=head_cate or DEFAULT_CATE,
        tail_cate=tail_cate or DEFAULT_CATE,
        head_type=head_type,
        tail_type=tail_type,
    )


def _parse_json_obj(obj: dict) -> RelationRow | None:
    head = _norm_text(obj.get("head") or obj.get("h") or obj.get("subject") or obj.get("s"))
    tail = _norm_text(obj.get("tail") or obj.get("t") or obj.get("object") or obj.get("o"))
    relation = _norm_text(obj.get("relation") or obj.get("rel") or obj.get("predicate") or obj.get("p"))
    if not head or not tail or not relation:
        return None

    head_cate = _norm_text(obj.get("head_cate") or obj.get("h_cate") or obj.get("cate_h") or obj.get("hCate"))
    tail_cate = _norm_text(obj.get("tail_cate") or obj.get("t_cate") or obj.get("cate_t") or obj.get("tCate"))
    head_type = _norm_text(obj.get("head_type") or obj.get("h_type") or obj.get("type_h") or obj.get("hType"))
    tail_type = _norm_text(obj.get("tail_type") or obj.get("t_type") or obj.get("type_t") or obj.get("tType"))

    return RelationRow(
        head=head,
        tail=tail,
        relation=_norm_relation(relation),
        head_cate=head_cate or DEFAULT_CATE,
        tail_cate=tail_cate or DEFAULT_CATE,
        head_type=head_type or DEFAULT_TYPE,
        tail_type=tail_type or DEFAULT_TYPE,
    )


def _iter_relations_from_path(path: Path) -> Iterable[RelationRow]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
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
                    row = _parse_json_obj(obj)
                    if row:
                        yield row
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            row = _parse_json_obj(item)
                            if row:
                                yield row
        return

    # default: csv-like
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = _parse_csv_line(line)
            if row:
                yield row


def main() -> None:
    base_path: Path = BASE_RELATION_PATH
    new_path: Path = NEW_RELATIONS_PATH
    if not new_path.exists():
        raise FileNotFoundError(
            f"NEW_RELATIONS_PATH not found: {new_path}. "
            "Edit kgqa/scripts/merge_relations.py to set the correct path."
        )
    out_path: Path = OUTPUT_PATH or base_path
    if OUTPUT_PATH is not None:
        # Allow configuring OUTPUT_PATH as a directory for safety.
        if out_path.suffix == "" or out_path.exists() and out_path.is_dir():
            out_path = out_path / base_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_rows = list(_iter_relations_from_path(base_path))
    base_keys = {r.key() for r in base_rows}

    added_rows: list[RelationRow] = []
    seen_in_new: set[tuple[str, str, str, str, str]] = set()
    for r in _iter_relations_from_path(new_path):
        k = r.key()
        if k in base_keys:
            continue
        if k in seen_in_new:
            continue
        seen_in_new.add(k)
        added_rows.append(r)
        base_keys.add(k)

    def emit_line(r: RelationRow) -> str:
        if EMIT_MODE == "v1":
            return r.to_v1_line()
        if EMIT_MODE == "v2":
            return r.to_v2_line()
        # mixed
        if _norm_type(r.head_type) == "Person" and _norm_type(r.tail_type) == "Person":
            return r.to_v1_line()
        return r.to_v2_line()

    # Preserve existing base lines as-is by rewriting from parsed rows.
    # (This normalizes relation synonyms if similar_words provides mappings.)
    lines_out = [emit_line(r) for r in base_rows] + [emit_line(r) for r in added_rows]
    out_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

    print(
        json.dumps(
            {
                "base_rows": len(base_rows),
                "added_rows": len(added_rows),
                "output_rows": len(lines_out),
                "output_path": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
