# Tongyi DeepResearch 前端

基于 Vue 3 + Vite 的轻量控制台，用于与远程 Tongyi DeepResearch Agent 进行多轮对话与工具调用调试。

## 快速开始

1. **安装依赖（建议使用 pnpm）**
   ```bash
   pnpm install
   ```
2. **配置环境变量**
   - 复制 `.env.example` 为 `.env.development` 或 `.env.production`
   - 根据实际接口填写 `VITE_AGENT_API_BASE`（或设置 `VITE_AGENT_PROXY_TARGET` 通过 Vite 代理）
   - 若接口需要认证，设置 `VITE_AGENT_API_KEY`
3. **启动开发服务器**
   ```bash
   pnpm dev
   ```
   默认监听 `http://127.0.0.1:4173`，可通过 `--host 0.0.0.0` 暴露给局域网。若远端 Agent API 未开放 CORS，可在 `.env` 设置 `VITE_AGENT_PROXY_TARGET`（指向你的 Agent 服务，例如 `http://157.66.255.40:9200`），开发阶段会把 `/api-proxy/*` 请求自动转发到真实接口；生产环境请由后端或网关暴露 `/v1/chat/agent` 等接口。

4. **打包发布**
   ```bash
   pnpm build
   pnpm preview
   ```

## 目录结构

```
src/
├── api/            # axios 封装
├── components/     # ChatInput、ChatMessages 等基础组件
├── composables/    # 状态逻辑（如 useAgentChat）
├── router/         # 路由配置
├── views/          # 页面级组件
└── assets/         # 全局样式
```

## 注意

- 该前端假设后端为 OpenAI Chat Completions 协议；如接口路径或参数不同，请在 `src/api/client.ts` 调整。
- 若浏览器无法直接访问远端模型端口，可在 `.env` 中设置 `VITE_AGENT_PROXY_TARGET`，Vite 开发服务器会自动转发 `/v1` 请求。生产环境部署时，请使用 Nginx/自建后端完成转发。*** End Patch
