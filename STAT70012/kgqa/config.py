import os

from py2neo import Graph

NEO4J_URL = os.environ.get("NEO4J_URL", "http://localhost:7474")
NEO4J_AUTH = os.environ.get("NEO4J_AUTH", "none")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "950302")

if NEO4J_AUTH.strip().lower() == "none":
    graph = Graph(NEO4J_URL)
else:
    graph = Graph(NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

BOOK_CONFIGS = {
    'dream': {
        'CA_LIST': {"荣国府":0,"宁国府":1,"王家":2,"史家":3,"薛家":4,"其他":5,"林家":6},
        'similar_words': {
            "爸爸": "父亲", "父亲": "父亲", "爹": "父亲",
            "妈妈": "母亲", "母亲": "母亲", "娘": "母亲",
            "儿子": "儿子", "女儿": "女儿", "丫环": "丫环", "丫鬟": "丫环",
            "兄弟": "兄弟", "哥哥": "哥哥", "弟弟": "弟弟",
            "妻": "夫妻", "妻子": "夫妻", "老婆": "夫妻", "丈夫": "夫妻", "老公": "丈夫", "爱人": "夫妻",
            "表妹": "表兄妹", "表兄": "表兄妹", "表弟": "表兄妹", "表姐": "表兄妹"
        }
    },
    'journey': {
        'CA_LIST': {"西行师徒":0, "满天神佛":1, "各路妖魔":2, "人间万象":3},
        'similar_words': {
            "徒弟": "徒弟", "师父": "师父", "师傅": "师父", "师徒": "师徒",
            "武器": "武器", "法宝": "法宝", "住所": "居住于", "居住": "居住于", "住在哪": "居住于",
            "主人": "主人", "坐骑": "坐骑", "保护": "奉命保护", "受职": "受职", "挑战": "挑战"
        }
    }
}

CA_LIST = BOOK_CONFIGS['dream']['CA_LIST']
similar_words = BOOK_CONFIGS['dream']['similar_words']
