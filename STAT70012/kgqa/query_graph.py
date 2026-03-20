from neo_db.config import graph, BOOK_CONFIGS
from spider.show_profile import get_profile
import base64
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
_JSON_CACHE = {}

def _load_json_data(book):
    path = BASE_DIR / "static" / f"{book}_data.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"data": [], "links": []}

def _get_node_info_from_json(name, book):
    data = _load_json_data(book)
    for node in data.get("data", []):
        if node["name"] == name:
            return node.get("category", "其他"), node.get("karma_text", "")
    return "其他", ""

def query(name, book='dream'):
    name = str(name or "").strip()
    if not name:
        cypher = "MATCH (p {book: $book})-[r]->(n {book: $book}) RETURN p.Name, r.relation, n.Name, p.cate, n.cate LIMIT 1000"
        data = graph.run(cypher, book=book)
    else:
        cypher = """
            MATCH (p {book: $book})-[r]->(n {Name:$name, book: $book}) 
            RETURN p.Name, r.relation, n.Name, p.cate, n.cate 
            UNION ALL 
            MATCH (p {Name:$name, book: $book})-[r]->(n {book: $book}) 
            RETURN p.Name, r.relation, n.Name, p.cate, n.cate
        """
        data = graph.run(cypher, name=name, book=book)
    return get_json_data(list(data), book)

def get_json_data(data, book='dream'):
    json_data = {'data': [], "links": []}
    if not data: return json_data
    node_set = set()
    for i in data:
        for k in ['p', 'n']:
            nm = i.get(f'{k}.Name')
            if nm and nm not in node_set:
                node_set.add(nm)
                cate, _ = _get_node_info_from_json(nm, book)
                json_data['data'].append({'name': nm, 'category': cate})
        json_data['links'].append({'source': i.get('p.Name'), 'target': i.get('n.Name'), 'value': i.get('r.relation')})
    return json_data

def get_KGQA_answer(array, book='dream'):
    config = BOOK_CONFIGS.get(book, BOOK_CONFIGS['dream'])
    sim_words = config['similar_words']
    
    if len(array) < 2: return {"error": "无法解析问题"}
        
    start_node, relation_word = array[0], array[1]
    if relation_word not in sim_words: return {"error": f"不理解关系: {relation_word}"}
    rel = sim_words[relation_word]
    
    # 终极双向查询
    q1 = "MATCH (p)-[r:RELATION {relation:$rel, book:$book}]->(n {Name:$name, book:$book}) RETURN p, r, n"
    q2 = "MATCH (n {Name:$name, book:$book})-[r:RELATION {relation:$rel, book:$book}]->(p) RETURN p, r, n"
    
    all_data = list(graph.run(q1, book=book, rel=rel, name=start_node)) + list(graph.run(q2, book=book, rel=rel, name=start_node))
    
    ans_name, profile_html, b64 = "", "", ""
    results_for_json = []

    if all_data:
        ans_name = all_data[0]['p']['Name'] # 答案永远在 p
        for item in all_data:
            p_node, r_rel, n_node = item['p'], item['r'], item['n']
            results_for_json.append({
                'p.Name': p_node['Name'], 'n.Name': n_node['Name'], 'r.relation': r_rel['relation'],
                'p.cate': p_node.get('cate'), 'n.cate': n_node.get('cate')
            })

    if ans_name:
        cate, karma_text = _get_node_info_from_json(ans_name, book)
        profile_html = get_profile(ans_name, book, cate, karma_text)
        img_path = BASE_DIR / "spider" / "images" / (f"{ans_name}.jpg")
        try:
            with img_path.open("rb") as f: b64 = base64.b64encode(f.read()).decode("ascii")
        except: pass

    return [get_json_data(results_for_json, book), profile_html, b64]

def get_answer_profile(name, book='dream'):
    cate, karma_text = _get_node_info_from_json(name, book)
    b64 = ""
    img_path = BASE_DIR / "spider" / "images" / (f"{name}.jpg")
    try:
        with img_path.open("rb") as f: b64 = base64.b64encode(f.read()).decode("ascii")
    except: pass
    return [get_profile(name, book, cate, karma_text), b64]
