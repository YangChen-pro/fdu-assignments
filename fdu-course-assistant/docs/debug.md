# 常见问题排查

遇到问题时，先按这个顺序排查：Cookie 是否存在，域名是否命中，`csrfToken` 是否能提取，教学班代码和课程类别是否匹配。

## `未找到 Cookie`

原因：没有设置配置里的 `cookie_env` 环境变量。默认变量名是 `FDU_XK_COOKIE`。

处理：重新执行下面的命令，把浏览器 Request Headers 里的整行 Cookie 放进去。

```bash
export FDU_XK_COOKIE='浏览器 Request Headers 里的整行 Cookie'
```

## `所有候选域名都无法提取 csrfToken`

常见原因：

- Cookie 已过期，需要重新登录后复制。
- Cookie 来自另一个域名，例如浏览器在 `sh.cn`，配置却只保留了 `edu.cn`。
- 复制的是首页或验证码请求，还没有真正进入登录后的选课页。
- 选课页结构变化，脚本没有从页面里匹配到 token。

处理：

```bash
fdu-course inspect --config configs/local.yaml
```

看输出里的命中域名、页面标题和页面片段。如果页面片段像登录页，说明 Cookie 或域名不对。

## 返回非 JSON

`choiceCourse.do` 正常情况下应返回 JSON。非 JSON 通常意味着请求被重定向到了 HTML 页面。

常见原因：

- 登录态失效。
- 域名不匹配。
- 请求路径或接口参数变化。

处理：先跑只读检查，确认页面仍是登录后的选课页。

```bash
fdu-course check-cookie --config configs/local.yaml
fdu-course inspect --config configs/local.yaml
```

## 课程类别错误

配置里的 `category` 必须是脚本支持的中文类别：

- 学位基础课
- 专业选修课
- 学位专业课
- 公共选修课
- 第一外国语
- 政治理论课
- 专业外语

如果写错，脚本无法确定接口字段 `lx`。

## 教学班代码错误

配置里的 `ids` 填教学班代码，也就是接口字段 `bjdm`。它不是课程名称，也不是培养方案里的课程号。

处理：到浏览器 Network 里找 `choiceCourse.do` 的 Payload 或 Form Data，确认这三个字段是否对应：

- `bjdm`：教学班代码。
- `bqmc`：课程类别中文名。
- `lx`：课程类别数字编码。

## 页面频繁提示过期

脚本运行期间，网页端可能出现页面过期提示。停止脚本后通常恢复。实战时不要同时在网页端和脚本端反复提交同一教学班。

## 请求频率

`poll_interval_seconds` 默认是 `0.3` 秒。调试阶段建议更保守，例如 `1.0`；正式运行再根据实际情况调整。不要把间隔设为 `0`。

## 实战案例记录区

真实排查记录只写脱敏结果，不粘贴原始 Cookie、姓名、学号或完整 token。

### 2026-07-07 单轮验证

环境：登录后页面为 `gotoChooseCourse.do`，命中域名 `yjsxk.fudan.edu.cn`，`check-cookie` 和 `inspect` 均可提取 `csrfToken`。

`run --once` 对 `政治理论课` 的第一个教学班代码发起真实提交，请求返回：教学班容量已满或退选席位暂未释放，提示退选席位每天 13:00 定时释放。随后第二个请求遇到 `Connection reset by peer`，脚本会把这类连接中断归一化为 `network_error`，避免直接抛 traceback。

结论：Cookie、`csrfToken`、域名、`choiceCourse.do` 表单链路是通的；当前失败原因不是登录态，而是选课时间、容量状态或服务器连接中断。
