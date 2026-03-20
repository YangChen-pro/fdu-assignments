import csv
import io

# =================配置区域=================
# 1. 实体映射表：把简称映射为全名
ENTITY_MAP = {
    "宝玉": "贾宝玉", "凤姐": "王熙凤", "凤姐儿": "王熙凤",
    "黛玉": "林黛玉", "宝钗": "薛宝钗", "湘云": "史湘云",
    "老祖宗": "贾母", "老爷": "贾政", "太太": "王夫人",
    "颦儿": "林黛玉", "宝二爷": "贾宝玉", "二爷": "贾宝玉",
    "琏二爷": "贾琏", "平儿": "平儿", "袭人": "花袭人",
    "宁府": "宁国府", "荣府": "荣国府", "贾府": "贾家",
}

# 2. 分类映射表：简化冗长的分类
CATE_MAP = {
    "贾家宁国府": "宁国府", "贾家荣国府": "荣国府",
    "史家": "史家", "王家": "王家", "薛家": "薛家",
    "林家": "林家", "其他": "其他",
}

# 3. 关系修正：修正模糊的关系描述
RELATION_MAP = {
    "妻": "夫妻", "丈夫": "夫妻",
    "父": "父亲", "母": "母亲",
    "子": "儿子", "女": "女儿",
    "爷": "祖父", "奶": "祖母",
}

# =================核心逻辑=================

def clean_line(line):
    parts = line.strip().split(',')
    if len(parts) < 3: return None  # 忽略无效行

    # 提取基础数据
    head = parts[0].strip()
    tail = parts[1].strip()
    relation = parts[2].strip()
    
    # 获取原始分类（如果不足5列，补全为"其他"）
    head_cate = parts[3].strip() if len(parts) > 3 else "其他"
    tail_cate = parts[4].strip() if len(parts) > 4 else "其他"
    
    # 获取原始类型（如果不足7列，默认为Person）
    head_type = parts[5].strip() if len(parts) > 5 else "Person"
    tail_type = parts[6].strip() if len(parts) > 6 else "Person"

    # --- 1. 实体归一化 ---
    head = ENTITY_MAP.get(head, head)
    tail = ENTITY_MAP.get(tail, tail)

    # --- 2. 分类标准化 ---
    head_cate = CATE_MAP.get(head_cate, head_cate)
    tail_cate = CATE_MAP.get(tail_cate, tail_cate)

    # --- 3. 关系标准化 ---
    relation = RELATION_MAP.get(relation, relation)

    # --- 4. 类型推断 (简单的规则修正) ---
    # 如果关系包含"居住"，尾实体通常是地点
    if "居住" in relation or "位于" in relation:
        tail_type = "Location"
    # 如果关系包含"拥有"或"使用"，尾实体通常是物品
    if "拥有" in relation or "使用" in relation or "赠与" in relation:
        if tail_type == "Person": # 修正错误的标注
            tail_type = "Item"
    
    # --- 5. 事件处理 ---
    # 原始数据中很多Event被标为Person，这里尝试修正
    if "事件" in relation or "参与" in relation or "主导" in relation:
        tail_type = "Event"

    return [head, tail, relation, head_cate, tail_cate, head_type, tail_type]

def process_data(raw_text):
    seen = set()
    cleaned_rows = []
    
    # 逐行处理
    lines = raw_text.strip().split('\n')
    print(f"原始行数: {len(lines)}")
    
    for line in lines:
        if not line or line.startswith('Iteration'): continue # 跳过空行和无关头
        
        row = clean_line(line)
        if not row: continue
        
        # --- 6. 去重策略 ---
        # 创建一个唯一键，防止完全重复的数据
        # 注意：这里我们保留了双向关系（如果你想保留原始数据量的话）
        unique_key = tuple(row)
        if unique_key in seen:
            continue
        
        seen.add(unique_key)
        cleaned_rows.append(row)

    return cleaned_rows

# =================执行部分=================

# 将你那1700行数据粘贴到这里 raw_data_str 中，或者从文件读取
with open('kgqa/raw_data/hlm_new_relation.txt', 'r', encoding='utf-8') as f:
    raw_data_str = f.read()

# 如果 raw_data_str 为空（因为代码块里没放），请确保你理解如何加载数据
if len(raw_data_str.strip()) > 10: 
    final_data = process_data(raw_data_str)

    # 写入 CSV
    output_file = 'kgqa/raw_data/relation_gemini.txt'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        # writer.writerow(['head', 'tail', 'relation', 'head_cate', 'tail_cate', 'head_type', 'tail_type'])
        writer.writerows(final_data)

    print(f"处理完成！清洗后行数: {len(final_data)}")
    print(f"数据已保存为: {output_file}")
else:
    print("请在脚本中填入数据或读取文件。")