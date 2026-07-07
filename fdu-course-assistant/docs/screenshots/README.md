# 截图说明

这个目录只放真实程序运行截图，不放 mock 图、生成示意图或手工绘制的终端图。

当前截图均来自 macOS Terminal 实际执行脚本过程。命令行没有显示原始 Cookie；输出中的 Cookie 和 `csrfToken` 只保留截断后的脱敏形式。

当前截图：

- `01-program-check-cookie.png`：真实运行 `python3 -m fdu_course_assistant.cli check-cookie --config configs/local.yaml`，显示命中域名和截断 token。
- `02-program-inspect.png`：真实运行 `python3 -m fdu_course_assistant.cli inspect --config configs/local.yaml`，显示页面标题、HTML 长度和页面片段。
- `03-program-run-once.png`：真实运行 `python3 -m fdu_course_assistant.cli run --config configs/local.yaml --once`，显示单轮提交过程、接口返回和最终统计。
