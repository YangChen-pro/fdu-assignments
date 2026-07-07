# 代码结构说明

这份文档面向想改代码的人。只想使用脚本时，先看 `README.md` 和 `docs/capture.md`。

## 选课请求链路

脚本使用浏览器登录后的 Cookie 访问选课页，再从页面中提取 `csrfToken`。后续提交课程时，使用同一个域名和 token。

常见域名：

- `yjsxk.fudan.edu.cn`
- `yjsxk.fudan.sh.cn`

实际请求路径：

- 获取选课页和 token：`/yjsxkapp/sys/xsxkappfudan/xsxkHome/gotoChooseCourse.do`
- 提交教学班：`/yjsxkapp/sys/xsxkappfudan/xsxkCourse/choiceCourse.do?_={timestamp}`

提交表单字段：

- `bjdm`：教学班代码，即网页里某一门课对应的具体教学班编号。
- `lx`：课程类别数字编码。
- `bqmc`：课程类别中文名。
- `csrfToken`：从选课页提取。

## 模块分工

- `session.py`：处理 Cookie 清洗、Cookie 脱敏、请求头生成、`csrfToken` 提取。
- `client.py`：发送真实 HTTP 请求、探测可用域名、执行只读检查、解析提交响应。
- `runner.py`：处理开始时间等待、结束时间停止、循环提交、`--once` 单轮验证、成功后移除目标。
- `config.py`：读取 YAML/JSON 配置，解析时间窗、候选域名和课程配置。
- `models.py`：定义课程目标、运行窗口、提交结果等数据结构。
- `cli.py`：提供 `check-cookie`、`inspect`、`run` 三个命令。

## 命令边界

- `check-cookie`：只访问选课页，验证 Cookie、域名和 `csrfToken`。
- `inspect`：只读检查选课页，输出脱敏 Cookie、命中域名、页面标题和页面片段。
- `run --once`：真实提交一轮，用于验证请求链路。
- `run`：按配置时间窗循环真实提交。

## 安全处理

真实 Cookie 只从环境变量或命令行临时参数读取，不写入模板和文档。日志和终端输出只展示脱敏 Cookie 与截断后的 `csrfToken`。
