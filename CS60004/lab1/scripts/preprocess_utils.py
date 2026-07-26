import json
import os
import re
from openai import OpenAI


class TeacherEmptyContentError(ValueError):
    def __init__(self, finish_reason, max_tokens):
        self.finish_reason = finish_reason
        self.max_tokens = max_tokens
        super().__init__(
            f'teacher returned empty content (finish_reason={finish_reason}, max_tokens={max_tokens})'
        )


class JudgeOutputFormatError(ValueError):
    def __init__(
        self,
        message,
        missing_subscores=None,
        retryable=None,
        error_type='format'
    ):
        self.missing_subscores = missing_subscores or []
        if retryable is None:
            retryable = len(self.missing_subscores) == 1
        self.retryable = retryable
        self.error_type = error_type
        super().__init__(message)


class TeacherModel(object):
    def __init__(self, model_name, base_url):
        self.model_name = model_name
        self.client = OpenAI(
            api_key='EMPTY',
            base_url=base_url
        )

    def chat(self, messages, max_tokens=1024):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body={
                'top_k': 20,
            },
        )

        choice = response.choices[0]
        content = choice.message.content

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    texts.append(str(item.get('text', '')))
                elif hasattr(item, 'text'):
                    texts.append(str(item.text))
            joined_text = ''.join(texts).strip()
            if joined_text:
                return joined_text

        raise TeacherEmptyContentError(choice.finish_reason, max_tokens)
    

def build_plan_b_teacher_prompt(sample):
    question = sample['question'].strip()
    options = format_options(sample['option'])
    gold_answer = str(sample['answer']).strip().upper()

    prompt = (
        '下面是一道中文医学单项选择题。\n'
        '我已经给出标准答案，请你不要重新判断答案，不要修改答案，只需要围绕这个标准答案生成简洁、合理、可用于监督微调的推理过程。\n\n'
        f'题目：{question}\n'
        f'选项：\n{options}\n\n'
        f'标准答案：{gold_answer}\n\n'
        '请严格按照以下要求输出：\n'
        '1. 只输出一个合法的 JSON 对象，不要输出任何额外文字。\n'
        '2. JSON 中必须包含两个字段：reasoning 和 answer。\n'
        '3. reasoning 必须是字符串列表，包含 2 到 4 条简洁推理，每条一句话。\n'
        '4. answer 必须直接填写给定的标准答案，不能修改。\n'
        '5. 不要输出 markdown 代码块。\n\n'
        '输出示例：\n'
        '{"reasoning": ["第一条推理", "第二条推理"], "answer": "A"}'
    )
    return prompt


def parse_plan_b_output(output_str):
    """Parse a Plan B output JSON string."""
    output_obj = json.loads(output_str)
    if not isinstance(output_obj, dict):
        raise ValueError('plan b output is not a dict')
    return output_obj


def get_plan_b_gold_answer(sample):
    """Extract the gold answer from an existing Plan B sample."""
    output_obj = parse_plan_b_output(sample['output'])
    answer = str(output_obj.get('answer', '')).strip().upper()
    if answer not in ['A', 'B', 'C', 'D', 'E']:
        raise ValueError(f'invalid plan b answer: {answer}')
    return answer


def check_plan_b_clean_hard_rules(output_str):
    """Apply deterministic hard rules for Plan B cleaning."""
    reasons = set()

    try:
        output_obj = parse_plan_b_output(output_str)
    except Exception:
        return False, ['invalid_json']

    reasoning = output_obj.get('reasoning')
    answer = str(output_obj.get('answer', '')).strip().upper()

    if answer not in ['A', 'B', 'C', 'D', 'E']:
        reasons.add('invalid_answer')

    if not isinstance(reasoning, list):
        reasons.add('reasoning_not_list')
        return False, sorted(reasons)

    if len(reasoning) != 3:
        reasons.add('reasoning_len_not_3')

    meta_pattern = re.compile(
        r'本题|标准答案|给定标准答案|依据题目给出的标准答案|'
        r'根据本题给出的标准考点设定|结合医学常识及给定标准答案'
    )
    option_pattern = re.compile(r'[ABCDE][项）\)]|选项\s*[ABCDE]|故选\s*[ABCDE]')
    answer_pattern = re.compile(r'答案为\s*[ABCDE]|应选\s*[ABCDE]')

    for item in reasoning:
        if not isinstance(item, str):
            reasons.add('reasoning_item_not_string')
            continue

        text = item.strip()
        if not text:
            reasons.add('reasoning_item_empty')
            continue

        if meta_pattern.search(text):
            reasons.add('meta_phrase')
        if option_pattern.search(text):
            reasons.add('option_leakage')
        if answer_pattern.search(text):
            reasons.add('answer_leakage')

    return len(reasons) == 0, sorted(reasons)


def build_plan_b_clean_judge_prompt(sample, output_str):
    """Build the teacher-judge prompt for a Plan B sample."""
    gold_answer = get_plan_b_gold_answer(sample)

    prompt = (
        '下面是一道中文医学单项选择题，以及一个用于监督微调的候选输出。\n'
        '请你评估这份候选输出中的 reasoning 质量是否适合给小模型做 SFT。\n'
        '你不需要重新答题，只需要从训练数据质量角度进行打分。\n\n'
        f'{sample["input"]}\n\n'
        f'标准答案：{gold_answer}\n'
        f'候选输出：{output_str}\n\n'
        '评分维度（每项 0-2 分，总分 10 分）：\n'
        '1. key_point：是否抓住题干中的关键判别点。\n'
        '2. support_gold：是否真正解释了为什么标准答案成立。\n'
        '3. conciseness：是否简洁、具体、无明显空话。\n'
        '4. independence：是否没有依赖“标准答案已知”之类的后验表述。\n'
        '5. factual_risk：是否没有明显事实冲突、硬拗 gold 或可疑内容。\n\n'
        'decision 的含义：\n'
        '- keep：可直接保留用于训练。\n'
        '- regen：建议重写 reasoning 后再用。\n'
        '- drop：样本存在明显事实风险或质量过差，不建议进入 B_clean。\n\n'
        '请只输出一个合法 JSON 对象，不要输出额外文字，不要输出 markdown。\n'
        'JSON 格式如下：\n'
        '{"total_score": 0, "subscores": {"key_point": 0, "support_gold": 0, '
        '"conciseness": 0, "independence": 0, "factual_risk": 0}, '
        '"decision": "keep", "reason": "一句话说明"}'
    )
    return prompt


def parse_plan_b_clean_judge_result(raw_text):
    """Validate and normalize the teacher-judge result."""
    try:
        output_obj = json.loads(raw_text.strip())
    except json.JSONDecodeError as e:
        raise JudgeOutputFormatError(
            f'judge output invalid json: {e}',
            retryable=True,
            error_type='json_decode',
        )
    if not isinstance(output_obj, dict):
        raise JudgeOutputFormatError('judge output is not a dict')

    subscores = output_obj.get('subscores')
    if not isinstance(subscores, dict):
        raise JudgeOutputFormatError('judge output missing subscores')

    normalized_subscores = {}
    total_score = 0
    missing_subscores = []
    for key in ['key_point', 'support_gold', 'conciseness', 'independence', 'factual_risk']:
        if key not in subscores:
            missing_subscores.append(key)
            continue
        score = int(subscores[key])
        if score < 0 or score > 2:
            raise JudgeOutputFormatError(f'invalid subscore {key}: {score}')
        normalized_subscores[key] = score
        total_score += score

    if missing_subscores:
        if len(missing_subscores) == 1:
            raise JudgeOutputFormatError(
                f'judge output missing subscore: {missing_subscores[0]}',
                missing_subscores=missing_subscores,
                error_type='missing_subscore',
            )
        raise JudgeOutputFormatError(
            'judge output missing subscores: ' + ', '.join(missing_subscores),
            missing_subscores=missing_subscores,
            error_type='missing_subscores',
        )

    output_total_score = int(output_obj.get('total_score', total_score))
    if output_total_score != total_score:
        raise JudgeOutputFormatError(
            f'judge total score mismatch: output={output_total_score}, computed={total_score}'
        )

    decision = str(output_obj.get('decision', '')).strip().lower()
    if decision not in ['keep', 'regen', 'drop']:
        raise JudgeOutputFormatError(f'invalid judge decision: {decision}')

    reason = str(output_obj.get('reason', '')).strip()
    if not reason:
        raise JudgeOutputFormatError('judge output missing reason')

    return {
        'total_score': total_score,
        'subscores': normalized_subscores,
        'decision': decision,
        'reason': reason,
    }


def judge_plan_b_clean_output(sample, output_str, teacher, max_tokens=4096):
    """Use the teacher model to score a Plan B output."""
    prompt = build_plan_b_clean_judge_prompt(sample, output_str)
    messages = [
        {'role': 'system', 'content': '你是一个严格的医学 SFT 数据质检员。'},
        {'role': 'user', 'content': prompt}
    ]
    raw_text = teacher.chat(messages, max_tokens=max_tokens)
    return parse_plan_b_clean_judge_result(raw_text)


def build_plan_b_clean_regen_prompt(sample):
    """Build a stricter regeneration prompt for Plan B cleaning."""
    gold_answer = get_plan_b_gold_answer(sample)

    prompt = (
        '下面是一道中文医学单项选择题。\n'
        '我已经给出标准答案，请你不要重新判断答案，只围绕这个标准答案生成更干净、更适合小模型监督微调的 reasoning。\n\n'
        f'{sample["input"]}\n\n'
        f'标准答案：{gold_answer}\n\n'
        '请严格按照以下要求输出：\n'
        '1. 只输出一个合法 JSON 对象，不要输出任何额外文字。\n'
        '2. JSON 中必须包含 reasoning 和 answer 两个字段。\n'
        '3. reasoning 必须是长度固定为 3 的字符串列表，每条只写一句话。\n'
        '4. 第 1 条写题干中的关键判别点。\n'
        '5. 第 2 条写为什么标准答案对应的选项成立。\n'
        '6. 第 3 条可简要排除一个最强干扰项，或总结最终判别依据。\n'
        '7. 不要出现“本题”“标准答案”“给定标准答案”“依据题目给出的标准答案”。\n'
        '8. 不要出现“A项”“B项”“选项A”“故选A”“答案为A”这类表述。\n'
        '9. 不要写“根据医学常识”“根据标准答案”等空泛后验话术。\n'
        '10. answer 必须直接填写给定的标准答案，不能修改。\n'
        '11. 不要输出 markdown 代码块。\n\n'
        '输出示例：\n'
        '{"reasoning": ["第一条推理", "第二条推理", "第三条推理"], "answer": "A"}'
    )
    return prompt


def regenerate_plan_b_clean_output(sample, teacher, max_tokens=4096):
    """Regenerate a cleaner Plan B output with stricter constraints."""
    prompt = build_plan_b_clean_regen_prompt(sample)
    messages = [
        {'role': 'system', 'content': '你是一个严谨的医学考试题解析助手。'},
        {'role': 'user', 'content': prompt}
    ]

    raw_text = teacher.chat(messages, max_tokens=max_tokens)
    output_obj = parse_plan_b_output(raw_text.strip())
    output_str = json.dumps(output_obj, ensure_ascii=False)
    gold_answer = get_plan_b_gold_answer(sample)

    ok, msg = validate_output(output_str, gold_answer, plan='B')
    if not ok:
        raise ValueError(f'invalid regenerated sample: {msg}')

    return output_str


def build_plan_c_knowledge_prompt(sample):
    question = sample['question'].strip()
    options = format_options(sample['option'])

    prompt = (
        '下面是一道中文医学单项选择题。\n'
        '请先抽取“解答本题可能用到的客观背景知识”，供后续推理使用。\n'
        '这里只能写背景知识，不能写解题过程，不能比较选项，不能给出最终答案。\n\n'
        f'题目：{question}\n'
        f'选项：\n{options}\n\n'
        '请严格按照以下要求输出：\n'
        '1. 只输出一个合法的 JSON 对象，不要输出任何额外文字。\n'
        '2. JSON 中只能包含一个字段：knowledge。\n'
        '3. knowledge 必须是字符串列表，包含 2 到 4 条内容。\n'
        '4. 每条 knowledge 必须是“客观、通用、可复用”的医学事实或概念，不得写成解题提示。\n'
        '5. 不要出现“本题”“题干”“选项”“应选”“答案”“故选”“因此选”等表述。\n'
        '6. 不要直接比较 A/B/C/D/E，不要点名哪个选项正确或错误。\n'
        '7. 不要写推理链，不要写“根据……可知……”，不要写结论句。\n'
        '8. 每条 knowledge 只写一句话，尽量短，避免重复。\n'
        '9. 不要输出 markdown 代码块。\n\n'
        '好的 knowledge 示例：\n'
        '{"knowledge": ["Z检验通常用于大样本均数比较或总体方差已知的情形。", "小样本均数比较通常采用t检验。"]}\n\n'
        '不好的 knowledge 示例：\n'
        '{"knowledge": ["本题考查Z检验的适用条件。", "根据题干可知应选D。", "A项不对，D项正确。"]}'
    )
    return prompt


def build_plan_c_reasoning_prompt(sample, knowledge):
    question = sample['question'].strip()
    options = format_options(sample['option'])
    gold_answer = str(sample['answer']).strip().upper()

    knowledge_text = '\n'.join([f'{idx + 1}. {item}' for idx, item in enumerate(knowledge)])

    prompt = (
        '下面是一道中文医学单项选择题。\n'
        '我会提供题目、选项、背景知识 knowledge，以及标准答案。\n'
        '请基于题目信息和给定 knowledge，生成“用于支持该标准答案”的简洁推理。\n'
        '不要修改答案，不要重新判断答案；你的任务是把知识落实到题目和选项上，形成判别过程。\n\n'
        f'题目：{question}\n'
        f'选项：\n{options}\n\n'
        f'背景知识：\n{knowledge_text}\n\n'
        f'标准答案：{gold_answer}\n\n'
        '请严格按照以下要求输出：\n'
        '1. 只输出一个合法的 JSON 对象，不要输出任何额外文字。\n'
        '2. JSON 中必须包含两个字段：reasoning 和 answer。\n'
        '3. reasoning 必须是字符串列表，包含 2 到 4 条内容。\n'
        '4. reasoning 必须体现“题目关键点 -> 对照 knowledge -> 落到选项判别 -> 得出答案”的过程。\n'
        '5. reasoning 不能只是重复或改写 knowledge 原句，必须结合题干或选项进行判断。\n'
        '6. reasoning 中至少有一条要明确指出题干的关键判别点，至少有一条要明确说明为什么标准答案对应的选项成立。\n'
        '7. 可以简要排除 1 到 2 个明显不符合的干扰项，但不要逐项展开成长篇解析。\n'
        '8. 不要出现与标准答案矛盾的内容。\n'
        '9. answer 必须直接填写给定的标准答案，不能修改。\n'
        '10. 不要输出 markdown 代码块。\n\n'
        '好的 reasoning 示例：\n'
        '{"reasoning": ["题干考查的是两个样本均数差别进行Z检验的适用条件。", "根据给定知识，Z检验主要用于大样本情形，因此关键条件是样本含量足够大。", "选项中只有D直接对应这一条件，所以答案为D。"], "answer": "D"}\n\n'
        '不好的 reasoning 示例：\n'
        '{"reasoning": ["Z检验通常用于大样本。", "小样本通常用t检验。", "中心极限定理成立。"], "answer": "D"}'
    )
    return prompt


def read_jsonl(path):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception as e:
                raise ValueError(f'Failed to parse line {line_id} in {path}: {e}')
    return samples


def write_jsonl(path, samples):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def append_jsonl(path, sample):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def format_options(option_dict):
    lines = []
    for key in ['A', 'B', 'C', 'D', 'E']:
        value = option_dict.get(key, '')
        lines.append(f'{key}. {value}')
    return '\n'.join(lines)


def build_instruction(plan='A'):
    if plan == 'A':
        return (
            '请回答下面的单项选择题，并只输出一个合法的 JSON 对象，不要输出任何额外解释。\n'
            'JSON 中只包含 1 个字段：answer。\n'
            'answer 的取值只能是 A、B、C、D、E 中的一个。'
        )

    elif plan == 'B':
        return (
            '请回答下面的单项选择题，并只输出一个合法的 JSON 对象，不要输出任何额外解释。\n'
            'JSON 中包含 2 个字段：reasoning 和 answer。\n'
            'reasoning 必须是字符串列表，列表中的每一项是一句简洁的推理。\n'
            'answer 的取值只能是 A、B、C、D、E 中的一个。'
        )

    elif plan == 'C':
        return (
            '请回答下面的单项选择题，并只输出一个合法的 JSON 对象，不要输出任何额外解释。\n'
            'JSON 中包含 3 个字段：knowledge、reasoning 和 answer。\n'
            'knowledge 必须是字符串列表，表示与解题相关的背景知识。\n'
            'reasoning 必须是字符串列表，表示基于题目与 knowledge 的简洁推理。\n'
            'answer 的取值只能是 A、B、C、D、E 中的一个。'
        )

    else:
        raise ValueError(f'Unknown plan: {plan}')


def build_input(sample, plan='A'):
    question = sample['question'].strip()
    options = format_options(sample['option'])

    if plan in ['A', 'B', 'C']:
        return (
            f'题目：{question}\n'
            f'选项：\n{options}'
        )
    
    else:
        raise ValueError(f'Unknown plan: {plan}')


def build_output(sample, plan='A', teacher=None):
    gold_answer = str(sample['answer']).strip().upper()

    if plan == 'A':
        output_obj = {
            'answer': gold_answer
        }
        return json.dumps(output_obj, ensure_ascii=False)

    elif plan == 'B':
        if teacher is None:
            raise ValueError('teacher is required for Plan B')

        prompt = build_plan_b_teacher_prompt(sample)
        messages = [
            {'role': 'system', 'content': '你是一个严谨的医学考试题解析助手。'},
            {'role': 'user', 'content': prompt}
        ]

        raw_text = teacher.chat(messages, max_tokens=4096)
        # 调试用
        # print(f'[Debug] raw teacher output: {raw_text}')
        output_obj = json.loads(raw_text.strip())

        return json.dumps(output_obj, ensure_ascii=False)

    elif plan == 'C':
        if teacher is None:
            raise ValueError('teacher is required for Plan C')
        
        # Step 1: 获取知识
        knowledge_prompt = build_plan_c_knowledge_prompt(sample)
        knowledge_messages = [
            {'role': 'system', 'content': '你是一个严谨的医学考试题知识抽取助手。'},
            {'role': 'user', 'content': knowledge_prompt}
        ]
        knowledge_text = teacher.chat(knowledge_messages, max_tokens=8192)
        knowledge_obj = json.loads(knowledge_text.strip())
        knowledge = knowledge_obj.get('knowledge', [])

        # Step 2: 基于知识进行推理
        reasoning_prompt = build_plan_c_reasoning_prompt(sample, knowledge)
        reasoning_messages = [
            {'role': 'system', 'content': '你是一个严谨的医学考试题解析助手。'},
            {'role': 'user', 'content': reasoning_prompt}
        ]
        reasoning_text = teacher.chat(reasoning_messages, max_tokens=8192)
        reasoning_obj = json.loads(reasoning_text.strip())
        reasoning = reasoning_obj.get('reasoning', [])
        answer = reasoning_obj.get('answer', '')

        # Step 3: 组织最终输出
        output_obj = {
            'knowledge': knowledge,
            'reasoning': reasoning,
            'answer': answer
        }
        return json.dumps(output_obj, ensure_ascii=False)

    else:
        raise ValueError(f'Unknown plan: {plan}')


def validate_output(output_str, gold_answer, plan='A'):
    gold_answer = str(gold_answer).strip().upper()
    output_obj = json.loads(output_str)

    if plan == 'A':
        return True, ''
    
    elif plan == 'B':
        if 'reasoning' not in output_obj:
            return False, 'missing reasoning field'
        if 'answer' not in output_obj:
            return False, 'missing answer field'

        reasoning = output_obj['reasoning']
        if not isinstance(reasoning, list):
            return False, 'reasoning is not a list'
        if len(reasoning) == 0:
            return False, 'reasoning is empty'

        for item in reasoning:
            if not isinstance(item, str):
                return False, 'reasoning item is not a string'
            if not item.strip():
                return False, 'reasoning item is empty'

        pred = str(output_obj['answer']).strip().upper()
        if pred not in ['A', 'B', 'C', 'D', 'E']:
            return False, f'invalid answer: {pred}'

        if pred != gold_answer:
            return False, f'answer mismatch: pred={pred}, gold={gold_answer}'

        return True, ''

    elif plan == 'C':
        if 'knowledge' not in output_obj:
            return False, 'missing knowledge field'
        if 'reasoning' not in output_obj:
            return False, 'missing reasoning field'
        if 'answer' not in output_obj:
            return False, 'missing answer field'
        
        knowledge = output_obj['knowledge']
        reasoning = output_obj['reasoning']

        if not isinstance(knowledge, list):
            return False, 'knowledge is not a list'
        if len(knowledge) == 0:
            return False, 'knowledge is empty'
        for item in knowledge:
            if not isinstance(item, str):
                return False, 'knowledge item is not a string'
            if not item.strip():
                return False, 'knowledge item is empty'

        if not isinstance(reasoning, list):
            return False, 'reasoning is not a list'
        if len(reasoning) == 0:
            return False, 'reasoning is empty'
        for item in reasoning:
            if not isinstance(item, str):
                return False, 'reasoning item is not a string'
            if not item.strip():
                return False, 'reasoning item is empty'

        pred = str(output_obj['answer']).strip().upper()
        if pred not in ['A', 'B', 'C', 'D', 'E']:
            return False, f'invalid answer: {pred}'

        if pred != gold_answer:
            return False, f'answer mismatch: pred={pred}, gold={gold_answer}'

        return True, ''

    else:
        return False, f'unknown plan: {plan}'


def convert_one_sample(sample, plan='A', teacher=None):
    instruction = build_instruction(plan=plan)
    input_text = build_input(sample, plan=plan)
    output = build_output(sample, plan=plan, teacher=teacher)

    ok, msg = validate_output(output, sample['answer'], plan=plan)
    if not ok:
        raise ValueError(f'Invalid converted sample: {msg}')

    new_sample = {
        'instruction': instruction,
        'input': input_text,
        'output': output
    }
    return new_sample


def process_one_sample(i, sample, plan='A', teacher=None):
    try:
        new_sample = convert_one_sample(sample, plan=plan, teacher=teacher)
        return i, new_sample, None
    except Exception as e:
        return i, None, e
