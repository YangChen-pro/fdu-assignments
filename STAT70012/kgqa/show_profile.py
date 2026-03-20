# -*- coding: utf-8 -*-
import os

# ==========================================
# 1. 红楼梦 (Dream) 专属数据
# ==========================================
DREAM_DATA = {
    "JUDGMENT_POEMS": {
        "贾宝玉": ["无故寻愁觅恨", "有时似傻如狂", "纵然生得好皮囊", "腹内原来草莽", "潦倒不通世务", "愚顽怕读文章", "行为偏僻性乖张", "那管世人诽谤"],
        "林黛玉": ["两地生孤木", "两处各悬心", "玉带林中挂", "金簪雪里埋"],
        "薛宝钗": ["可叹停机德", "堪怜咏絮才", "玉带林中挂", "金簪雪里埋"],
        "王熙凤": ["凡鸟偏从末世来", "都知爱慕此生才", "一从二令三人木", "哭向金陵事更哀"]
    },
    "GROUP_POEMS": {
        "丫鬟组": ["落红成阵", "风露清愁", "千红一哭", "万艳同悲", "喜散不喜聚", "奈何薄命"],
        "王孙组": ["陋室空堂", "当年笏满床", "衰草枯杨", "曾为歌舞场", "金满箱", "银满箱", "展眼乞丐人皆谤"],
        "世外组": ["世上万般哀苦", "唯有回头是岸", "好便是了", "了便是好", "痴迷不悟", "终归荒野"],
        "长辈组": ["享尽人间富贵", "看透世情冷暖", "儿孙满堂", "终归一梦", "树倒枢狲散", "飞鸟各投林"]
    },
    "SPECIFIC_BIOS": {
        "贾宝玉": "衔玉而生的荣府公子。他不爱功名，唯爱红妆，在诗词女儿堆里寻求真情，终在白茫茫大地中寻得解脱。",
        "林黛玉": "绛珠草转世，入府还泪。她一生孤傲如诗，与宝玉虽心意相通，却终究敌不过世易时移，泪尽而逝。"
    }
}

# ==========================================
# 2. 西游记 (Journey) 专属数据
# ==========================================
JOURNEY_DATA = {
    "POEM_偈": {
        "孙悟空": ["混元体正果初真", "历劫磨写成圣身", "大闹天宫惊圣主", "五行山下脱凡尘"],
        "唐三藏": ["宏愿深藏证法门", "单身策马向西昆", "不辞万里程途远", "只要真经度众魂"],
        "猪八戒": ["卷脏贪淫贬世间", "错投猪胎志未全", "钉耙倒筑群魔散", "净坛使者位金莲"],
        "沙悟净": ["失手打破玻璃盏", "流沙河里受严寒", "一心保护取经去", "金身罗汉果位完"],
        "白龙马": ["西海神龙犯天条", "鹰愁涧下锁英豪", "化作白马承圣体", "八部天龙在云霄"],
        "观音菩萨": ["碧藕金莲现化身", "慈航普渡济迷津", "手里杨枝常遍洒", "落伽山上现法身"]
    },
    "GROUP_偈": {
        "西行师徒": ["西行路远万重山", "风霜雪雨共艰难", "一心只求真经在", "扫尽妖氛得生还"],
        "满天神佛": ["仙风道骨隐云霞", "万劫修来法力嘉", "俯瞰苍生皆因果", "灵山宝殿是归家"],
        "各路妖魔": ["占山为王弄魔风", "狡诈贪婪意不穷", "纵有神通遮白日", "终归金箍法网中"],
        "人间万象": ["滚滚红尘百态多", "是非成败付烟波", "虽无神力遮天手", "自有真情唱赞歌"]
    }
}

def _get_dream_profile(name):
    """构建红楼梦风格"""
    poems = DREAM_DATA["JUDGMENT_POEMS"]
    groups = DREAM_DATA["GROUP_POEMS"]
    bios = DREAM_DATA["SPECIFIC_BIOS"]
    has_real_poem = name in poems
    section_title = "【判词】" if has_real_poem else "【红楼考语】"
    
    poem_lines = poems.get(name)
    if not poem_lines:
        if any(x in name for x in ["丫", "儿", "官", "鹃", "画", "雯", "月", "人"]): poem_lines = groups["丫鬟组"]
        elif any(x in name for x in ["僧", "道", "甄士隐"]): poem_lines = groups["世外组"]
        else: poem_lines = groups["王孙组"]
    
    poem_html = "".join([f"<span>{line}</span>" for line in poem_lines])
    bio = bios.get(name, f"红楼一梦，众生皆苦。{name}处此局中，其身世起伏见证了家族的繁华与幻灭。")
    
    return f"""
    <div class="letter-paper theme-dream">
        <div class="character-avatar-placeholder"></div>
        <div class="character-title">{name}</div>
        <div class="section-title">{section_title}</div>
        <div class="poem-container-vertical">{poem_html}</div>
        <div class="section-title">【生平】</div>
        <div class="bio-text">{bio}</div>
        <div class="letter-footer">—— 红楼梦 · 脂砚斋重评</div>
    </div>
    """

def _get_journey_profile(name, cate, karma_text=""):
    """构建西游记风格"""
    p_spec = JOURNEY_DATA["POEM_偈"]
    p_gen = JOURNEY_DATA["GROUP_偈"]
    
    poem_lines = p_spec.get(name, p_gen.get(cate, p_gen["人间万象"]))
    poem_html = "".join([f"<span>{line}</span>" for line in poem_lines])
    
    display_bio = karma_text if karma_text else f"大千世界，神魔众生。{name}于西行路上一现真踪，共谱释厄奇传奇。"

    return f"""
    <div class="letter-paper theme-journey">
        <div class="character-avatar-placeholder"></div>
        <div class="character-title">{name}</div>
        <div class="section-title">【西行偈】</div>
        <div class="poem-container-vertical">{poem_html}</div>
        <div class="section-title">【因果录】</div>
        <div class="bio-text">{display_bio}</div>
        <div class="letter-footer">—— 西游记 · 释厄传</div>
    </div>
    """

def get_profile(name, book='dream', cate=None, karma_text=""):
    """统一入口"""
    name = str(name or "")
    if book == 'journey':
        return _get_journey_profile(name, cate or "人间万象", karma_text)
    else:
        # 红楼梦模式
        return _get_dream_profile(name)
