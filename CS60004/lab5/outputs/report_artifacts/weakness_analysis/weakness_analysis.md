# Lab5 最终模型回答弱点分析（7 个 subagent 并行审阅）

## 覆盖范围

- 审阅对象：当前 `lab5/outputs/submission_task5/` 的最终角色采访与通用指令预测。
- 并行子代理：7 个，分别审阅 role 事实/记忆、role 情绪声线、role 随机样本、role 改写审计、role 表面质量、instruction 严格格式、instruction 随机样本。
- 抽样规模：290 条模型回答；其中 role 分片 180 条，instruction 分片 110 条。
- 注意：分析只读模型回答、题目、检索记忆和可见元数据；不把 eval/test 原文作为训练数据。

## 总体结论

当前提交的硬性格式和中英混杂问题已经比早期版本稳定，但剩余失分主要来自两个方向：role 任务的“听起来像角色但不够忠实于记忆”，以及 instruction 任务的“自然语言回答可以，但 exact checker 级别的表层约束仍会漏”。

## 主要弱点

### 1. role：具体事实题仍会编造未被记忆支持的实体/事件

- role_eval_000002：编造 midnight cafe、translator、local artist 等恋爱冒险细节；可见检索记忆是 Mediterranean cruise、Jerusalem peace conference、New Orleans jazz club。
- role_eval_000005：回答 Rome 是最美任务地点并扩写 basilicas/colonnades；检索记忆不支持 Rome 作为该问题的任务地点。
- role_eval_000053：点名 Caravaggio 影响短篇小说；检索只出现 Renaissance/classical museum/modern art gallery/book launch，没有 Caravaggio。
- role_eval_000322：把 breakthrough role 写成 independent film 小角色；检索只支持 film festival / premiere after-party / date 场景。

### 2. role：召回到的记忆没有稳定转成答案证据

- role_eval_000000：检索到 Moscow/Russian agents、Monte Carlo/arms dealer、Geneva/Spanish 等强证据，但回答只讲多语言能力的一般好处。
- role_eval_000009：检索到 Instagram roast、podcast、Instagram Live 事故，但回答只说 friends/funny people。
- role_eval_000016：检索到摄影 workshop、lighting、digital vs film 讨论，回答却是泛泛 teaching reward。
- role_eval_000023：检索有 environmental-law auditorium speech 和 high-profile case，回答仍偏抽象法律价值。

### 3. role：角色声线被“成熟自省小作文”压平

- role_eval_000028、000140、000143、000144：不同职业角色都落到 compassion/respect/meaningful/balance 等通用反思。
- role_eval_000120、000123：Dr. Crumble 的 lisp、高亢、火箭科学怪人式表达没有真正被表演出来。
- role_eval_000420：Rex 应该 short/clipped/gravelly/sarcastic，却输出流畅哲理段落。
- role_eval_000333：演员角色的 theatrical/booming stage 感不足，更像通用职业访谈。

### 4. role：问题类型遵循不足，长答案掩盖未直接回答

- role_eval_000031：问 recently discovered book，只答 a new translation of a classic novel，没有书名。
- role_eval_000037：问 recently finished book，只说 a novel / clever cat，没有落实到具体作品。
- role_eval_000486：问 favorite video game，154 词里没有命名任何游戏。
- role_eval_000047：space-related joke 类问题容易变成解释/感悟，而不是先给一个真正 joke。

### 5. role：改写补丁有效，但带来统一化和 judge-friendly 风险

- 两轮 Qwen3.5-35B-A3B 改写基本消除了明显 CJK 污染，回答更直接、更稳定。
- role_eval_000095、000199、000298、000323：多条改写呈现“直接回答 + 一个整齐故事 + 最后一句人生箴言”的相似结构。
- role_eval_000038、000258、000339：改写后仍会 invent partner / favorite book / exact fusion dishes。

### 6. role：表面质量已改善，但英文层面的异常和模板仍存在

- 当前抽样未发现 CJK 泄漏，但仍有 awkward phrasing、拼写/造词、粘连、奇怪语义。
- role_artifact_length 分片中 30 条长度集中在 136-160 词，平均约 142 词，说明输出长度控制不够随题目变化。
- 长答案更容易引入 unsupported detail，例如 role_eval_000316 的 thermal mismatch/prototype failure、role_eval_000186 的 lighting investigation。

### 7. instruction：strict 约束仍然需要确定性校验

- key 332：要求逐字重复完整请求，输出只重复部分请求。
- key 374：要求逐字重复第一行，但把 youngins/damn 改成 [redacted]，破坏 exact repeat。
- key 1082、1236：要求用 markdown divider `***` 分段，输出用了空行而不是 `***`。
- key 1130：要求字母 t 最多出现一次，输出中 Chesterfield 出现两次导致超限。

### 8. instruction：满足关键词/计数时会牺牲自然度和信息质量

- key 1203：peace/war 为满足出现次数被连续堆叠，历史解释自然度下降。
- key 2139：强塞 `rte` 成 “the rte of love survive”，语义异常。
- key 2169：`tianjin` 被作为不相关旁枝插入 Winter Olympics ratings 解释。
- key 322、2787：现实/技术问题存在时间背景或技术细节限定不足，答案看起来顺但不够精确。

## 可能根因

- role 生成把 retrieved_memories 当作主题提示，而不是先抽取“可支持事实 / 声线特征 / 禁止推断”。
- role prompt 更奖励流畅和角色感，缺少实体级 grounding、问题类型优先、无证据时保守回答的硬约束。
- Qwen3.5-35B-A3B 改写显著提升了可读性，但容易形成统一的 100 词左右反思型模板。
- instruction 侧缺少生成后的 deterministic verifier：exact repeat、分隔符、禁用字符、大小写、词/句/段计数都不应只靠模型自觉。
- 当前 judge/winner 指标适合筛候选，但不能替代专项诊断；role 的 hallucination、directness、voice distinctiveness 需要单独量化。

## 优先修复方向（保持训练/评估隔离）

- 给 role 推理加 evidence extraction：先从检索记忆抽取 1-3 个事实锚点、2 个声线锚点和禁止新增的实体类型，再生成最终回答；最终只输出答案。
- 按问题类型加模板约束：book/game/film/favorite 必须给名称或明确说记忆只支持概括；joke 必须先给 joke；experience/moment 必须用已检索场景，不足时保守概括。
- 加入 role 输出后处理/重采样：非英文字符、过长、未回答具体名词题、明显 unsupported named entity、重复模板句命中时重采样。
- instruction 侧做规则级 validator 和 repair：exact substring、`***`、start/end phrase、JSON schema、大小写、禁用字符、计数都用代码检查；失败时最小改写，不重新发挥内容。
- 构建不泄露的诊断集：从训练侧/独立合成题覆盖 fact grounding、joke、favorite entity、角色声线、strict format，不使用 leaderboard/test 题面做训练。

## 子代理原始结果

- JSON：`lab5/outputs/report_artifacts/weakness_analysis/subagent_weakness_results.json`
- Markdown：`lab5/outputs/report_artifacts/weakness_analysis/weakness_analysis.md`
