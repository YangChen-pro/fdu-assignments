# 浏览器抓包说明

这份文档说明怎样从浏览器里拿到脚本需要的信息。新手只需要按顺序找三类内容：Cookie、教学班代码、课程类别字段。

## 打开 Network 面板

1. 在浏览器里登录选课系统。
2. 按 `F12` 或右键选择“检查”。
3. 打开 DevTools 里的 Network 面板。
4. 刷新选课页，让 Network 里出现新的请求。
5. 只看 `yjsxk.fudan.edu.cn` 或 `yjsxk.fudan.sh.cn` 这两个域名下的请求。

## 复制 Cookie

在 Network 中点击任意一个选课系统请求，打开 Headers，找到 Request Headers 里的 `Cookie`。

复制整行 Cookie 后，放到终端环境变量里：

```bash
export FDU_XK_COOKIE='复制出的整行 Cookie'
```

不要把 Cookie 写进 `configs/local.yaml`、README 或任何会提交到 Git 的文件。

## 找教学班代码

配置里的 `ids` 填“教学班代码”。它是网页里某一个教学班的编号，在提交接口里字段名叫 `bjdm`。

常见格式类似：

```text
2025202601GEIP40013.02
2025202601GEIP40013.03
```

可靠来源有两个：

1. 在网页上点选目标课程后，看 Network 里的 `choiceCourse.do` 请求。
2. 打开这个请求的 Payload 或 Form Data，找到 `bjdm`。

不要把课程名称当教学班代码。脚本真正提交的是 `bjdm`。

## 确认课程类别

同一个 `choiceCourse.do` 请求里通常还能看到：

- `bjdm`：教学班代码，填到配置的 `ids`。
- `lx`：课程类别数字编码，脚本会根据 `category` 自动填写。
- `bqmc`：课程类别中文名，填到配置的 `category`。
- `csrfToken`：页面 token，脚本会自动提取，不需要手填。

如果网页请求里的 `bqmc` 是 `政治理论课`，配置就写：

```yaml
courses:
  - category: 政治理论课
    ids:
      - "2025202601GEIP40013.02"
```

## 只读验证

抓到 Cookie 和教学班代码后，先运行只读检查：

```bash
fdu-course check-cookie --config configs/local.yaml
fdu-course inspect --config configs/local.yaml
```

这两条命令不会提交课程。只有 `run` 和 `run --once` 会调用真实提交接口。

## 两个域名

选课系统常见入口有两个域名，当前项目默认同时尝试：

```yaml
targets:
  - yjsxk.fudan.edu.cn
  - yjsxk.fudan.sh.cn
```

如果 `check-cookie` 报 Cookie 过期，但确认 Cookie 是刚复制的，优先看另一个域名是否能提取 `csrfToken`。
