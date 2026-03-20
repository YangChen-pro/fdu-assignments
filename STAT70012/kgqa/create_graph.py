import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo_db.config import graph
from tqdm import tqdm

def _clear_book_data(book: str) -> None:
    print(f"\n[CLEAN] 正在删除数据库中属于 [{book}] 的所有节点和关系...")
    graph.run("MATCH (n {book: $book}) DETACH DELETE n", book=book)

def main() -> None:
    if len(sys.argv) < 3:
        print("错误：必须提供书籍ID和数据文件名。")
        return

    book_id = sys.argv[1]
    file_name = sys.argv[2]
    
    # 绝对路径锁定，绝不默认
    relation_path = PROJECT_ROOT / "raw_data" / file_name
    
    if not relation_path.exists():
        print(f"致命错误：文件不存在 -> {relation_path}")
        return

    print(f"[START] 正在从 {relation_path} 导入数据到书籍 [{book_id}]...")

    _clear_book_data(book_id)

    with relation_path.open("r", encoding="utf-8") as file:
        lines = [l.strip() for l in file if l.strip()]
        print(f"[INFO] 文件行数: {len(lines)}")

        for line in tqdm(lines, desc="写入Neo4j"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 5: continue
            
            h, t, rel, h_cate, t_cate = parts[:5]
            
            # 1. 强制写入节点
            graph.run(
                "MERGE (e:Entity {Name: $name, book: $book}) "
                "SET e.cate = $cate",
                name=h, cate=h_cate, book=book_id
            )
            graph.run(
                "MERGE (e:Entity {Name: $name, book: $book}) "
                "SET e.cate = $cate",
                name=t, cate=t_cate, book=book_id
            )
            
            # 2. 强制写入关系
            graph.run(
                "MATCH (h:Entity {Name: $head, book: $book}), (t:Entity {Name: $tail, book: $book}) "
                "MERGE (h)-[r:RELATION {relation: $rel, book: $book}]->(t)",
                head=h, tail=t, rel=rel, book=book_id
            )

    print(f"\n[DONE] 书籍 [{book_id}] 导入完成。")

if __name__ == "__main__":
    main()