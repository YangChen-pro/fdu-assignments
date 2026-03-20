from neo_db.config import graph, CA_LIST, similar_words
from spider.show_profile import get_profile
import base64
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_category(cate: str | None) -> str:
    cate = (cate or "").strip() or "其他"
    return cate if cate in CA_LIST else "其他"


def _cypher_escape_ident(value: str) -> str:
    return (value or "").replace("`", "``")


def query(name):
    data = graph.run(
        "MATCH (p)-[r]->(n:Entity {Name:$name}) "
        "RETURN p.Name, r.relation, n.Name, p.cate, n.cate "
        "UNION ALL "
        "MATCH (p:Entity {Name:$name})-[r]->(n) "
        "RETURN p.Name, r.relation, n.Name, p.cate, n.cate",
        name=name,
    )
    data = list(data)
    return get_json_data(data)
def get_json_data(data):
    json_data={'data':[],"links":[]}
    d=[]
    
    
    for i in data:
        # print(i["p.Name"], i["r.relation"], i["n.Name"], i["p.cate"], i["n.cate"])
        d.append(i['p.Name']+"_"+_safe_category(i.get('p.cate')))
        d.append(i['n.Name']+"_"+_safe_category(i.get('n.cate')))
        d=list(set(d))
    name_dict={}
    count=0
    for j in d:
        j_array=j.split("_")
    
        data_item={}
        name_dict[j_array[0]]=count
        count+=1
        data_item['name']=j_array[0]
        data_item['category']=CA_LIST[_safe_category(j_array[1])]
        json_data['data'].append(data_item)
    for i in data:
   
        link_item = {}
        
        link_item['source'] = name_dict[i['p.Name']]
        
        link_item['target'] = name_dict[i['n.Name']]
        link_item['value'] = i['r.relation']
        json_data['links'].append(link_item)

    return json_data
def get_KGQA_answer(array):
    data_array=[]
    for i in range(len(array)-2):
        if i==0:
            name=array[0]
        else:
            name=data_array[-1]['p.Name']
           
        rel = similar_words[array[i+1]]
        rel_type = _cypher_escape_ident(rel)
        data = graph.run(
            f"MATCH (p)-[r:`{rel_type}` {{relation: $rel}}]->(n:Entity {{Name: $name}}) "
            "RETURN p.Name, n.Name, r.relation, p.cate, n.cate",
            rel=rel,
            name=name,
        )
       
        data = list(data)
        print(data)
        data_array.extend(data)
        
        print("==="*36)
    answer_name = str(data_array[-1]["p.Name"]) if data_array else ""

    b64 = ""
    if answer_name:
        image_path = PROJECT_ROOT / "spider" / "images" / (f"{answer_name}.jpg")
        try:
            with image_path.open("rb") as image:
                b64 = base64.b64encode(image.read()).decode("ascii")
        except FileNotFoundError:
            b64 = ""

    profile_html = get_profile(answer_name) if answer_name else ""
    return [get_json_data(data_array), profile_html, b64]
def get_answer_profile(name):
    name = str(name or "")
    b64 = ""
    image_path = PROJECT_ROOT / "spider" / "images" / (f"{name}.jpg")
    try:
        with image_path.open("rb") as image:
            b64 = base64.b64encode(image.read()).decode("ascii")
    except FileNotFoundError:
        b64 = ""
    return [get_profile(name), b64]
        
