import sys
from pathlib import Path
import os
from flask import Flask, render_template, request, jsonify

# 1. 基础设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

# 2. 导入后端逻辑
from neo_db.query_graph import query, get_KGQA_answer, get_answer_profile
from KGQA.ltp import get_target_array

# 3. DeepSeek 模型配置
import requests
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# --- 路由恢复：所有原始路由 + 新增路由 ---

@app.route('/', methods=['GET', 'POST'])
def cover():
    return render_template('cover.html')

@app.route('/index', methods=['GET', 'POST'])
def index(name=None):
    return render_template('index.html', name = name)

BOOK_CONFIG = {
    'dream': { 
        'title': '红楼梦 · 知识图谱', 
        'theme_class': 'theme-dream', 
        'search_placeholder': '寻访金陵十二钗，如：贾宝玉...', # 新增
        'tags': ['贾宝玉', '林黛玉', '薛宝钗', '王熙凤', '贾母'], # 补全
        'chat_welcome': '可问“贾宝玉的父亲是谁”', 
        'home_seal': '梦', 
        'intro_char': '空' 
    },
    'journey': { 
        'title': '西游记 · 知识图谱', 
        'theme_class': 'theme-journey', 
        'search_placeholder': '寻访三界圣贤，如：观音菩萨...', # 新增
        'tags': ['孙悟空', '唐三藏', '猪八戒', '沙悟净', '白龙马'], # 补全
        'chat_welcome': '可问“孙悟空的师父是谁”', 
        'home_seal': '行', 
        'intro_char': '悟' 
    }
}

@app.route('/project_main', methods=['GET', 'POST'])
def project_main():
    book_key = request.args.get('book', 'dream')
    config = BOOK_CONFIG.get(book_key, BOOK_CONFIG['dream'])
    return render_template('project_main.html', 
                         book=book_key,
                         book_title=config['title'],
                         theme_class=config['theme_class'],
                         search_placeholder=config['search_placeholder'], # 传递
                         tags=config['tags'], # 传递
                         chat_welcome=config['chat_welcome'],
                         home_seal=config['home_seal'],
                         intro_char=config['intro_char'])

# --- 搜索与问答接口 ---

@app.route('/search_name', methods=['GET', 'POST'])
def search_name_route():
    name = request.args.get('name')
    book = request.args.get('book', 'dream')
    return jsonify(query(name, book))

@app.route('/get_profile', methods=['GET', 'POST'])
def get_profile_route():
    name = request.args.get('character_name')
    book = request.args.get('book', 'dream')
    return jsonify(get_answer_profile(name, book))

@app.route('/KGQA_answer', methods=['GET', 'POST'])
def KGQA_answer_route():
    question = request.args.get('name')
    book = request.args.get('book', 'dream')
    target = get_target_array(question, book)
    if not target: return jsonify({"error": "无法解析问题"})
    return jsonify(get_KGQA_answer(target, book))

# --- DeepSeek 接口 ---

def call_deepseek_model(question: str, book: str) -> str:
    if not DEEPSEEK_API_KEY: return "API Key未配置"
    book_name = "《红楼梦》" if book == "dream" else "《西游记》"
    prompt = f"你是一位精通中国古典文学的专家。请仅根据{book_name}的原著内容，简洁地回答以下问题：\n\n{question}"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question}]}
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"调用失败: {e}"

@app.route('/deep_chat', methods=['POST'])
def deep_chat_route():
    data = request.json
    question = data.get('question')
    book = data.get('book', 'dream')
    if not question: return jsonify({"answer": "问题不能为空。"})
    answer = call_deepseek_model(question, book)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(port=5005, debug=True)
