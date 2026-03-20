import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo_db.config import graph
from tqdm import tqdm


_TYPE_TO_LABEL = {
    # canonical types
    "Person": "Person",
    "Place": "Place",
    "Location": "Place",
    "Org": "Org",
    "Organization": "Org",
    "Item": "Item",
    "Object": "Item",
    "Event": "Event",
    "Text": "Text",
}


def _parse_relation_line(line: str) -> tuple[str, str, str, str, str, str, str] | None:
    """
    relation.txt supports 2 formats (comma-separated):

    v1 (legacy, 5 fields):
      head,tail,relation,head_cate,tail_cate
      - head_type/tail_type default to Person

    v2 (extended, 7+ fields):
      head,tail,relation,head_cate,tail_cate,head_type,tail_type,(...)
    """
    parts = [x.strip() for x in (line or "").strip().split(",")]
    if len(parts) < 5:
        return None
    head, tail, relation, head_cate, tail_cate = parts[:5]
    if not head or not tail or not relation:
        return None

    head_type = parts[5].strip() if len(parts) >= 6 and parts[5].strip() else "Person"
    tail_type = parts[6].strip() if len(parts) >= 7 and parts[6].strip() else "Person"
    return head, tail, relation, head_cate, tail_cate, head_type, tail_type


def _merge_entity(*, name: str, cate: str, etype: str) -> None:
    label = _TYPE_TO_LABEL.get((etype or "").strip(), "")
    label_clause = f":{label}" if label else ""
    graph.run(
        f"MERGE (e:Entity{label_clause} {{Name: $name}}) "
        "SET e.cate = $cate, e.etype = $etype",
        name=name,
        cate=cate or "其他",
        etype=etype or "Entity",
    )


def _cypher_escape_ident(value: str) -> str:
    # For backtick-escaped identifiers in Cypher (labels / rel types).
    return (value or "").replace("`", "``")


def _clear_neo4j() -> None:
    # User-requested behavior: always start from a clean DB for repeatable imports.
    graph.run("MATCH (n) DETACH DELETE n")


def main() -> None:
    # relation_path = PROJECT_ROOT / "raw_data" / "relation.txt"
    relation_path = PROJECT_ROOT / "raw_data" / "tot_relation.txt"

    _clear_neo4j()

    with relation_path.open("r", encoding="utf-8") as file:
        total = sum(1 for _ in file)
        file.seek(0)
        it = file
        it = tqdm(it, total=total, desc="Importing relations", unit="line")

        for line in it:
            parsed = _parse_relation_line(line)
            if not parsed:
                continue
            head, tail, relation, head_cate, tail_cate, head_type, tail_type = parsed

            _merge_entity(name=head, cate=head_cate, etype=head_type)
            _merge_entity(name=tail, cate=tail_cate, etype=tail_type)

            rel_type = _cypher_escape_ident(relation)
            graph.run(
                f"MATCH (h:Entity {{Name: $head}}), (t:Entity {{Name: $tail}}) "
                f"MERGE (h)-[r:`{rel_type}`]->(t) "
                "SET r.relation = $relation "
                "RETURN r",
                head=head,
                tail=tail,
                relation=relation,
            )

    print("导入完成。Neo4j Browser UI: http://localhost:7474")


if __name__ == "__main__":
    main()
