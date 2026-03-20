import { ref } from 'vue';
import type { AgentMessage, AgentRequest } from '@/api/client';

export function useAgentChat() {
  const history = ref<AgentMessage[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const useTools = ref(true);
  const currentAssistantIndex = ref<number | null>(null);
  const AVAILABLE_TOOLS = ['search', 'visit', 'google_scholar', 'PythonInterpreter'];
  const toolUsage = ref<Record<string, { used: boolean; count: number; lastMessage?: string; messageIndex?: number }>>(
    AVAILABLE_TOOLS.reduce((acc, tool) => {
      acc[tool] = { used: false, count: 0 };
      return acc;
    }, {} as Record<string, { used: boolean; count: number; lastMessage?: string; messageIndex?: number }>)
  );
  const eventLogs = ref<Array<{ type: string; content: string; timestamp: string; messageIndex?: number }>>([]);

  async function ask(question: string) {
    if (!question.trim()) return;
    history.value.push({ role: 'user', content: question });
    loading.value = true;
    error.value = null;

    const payload: AgentRequest = {
      messages: history.value,
      model: import.meta.env.VITE_AGENT_MODEL ?? 'tongyi',
      temperature: Number(import.meta.env.VITE_AGENT_TEMPERATURE ?? 0.6),
      use_tools: useTools.value,
    };

    try {
      await runStreamRequest(payload);
    } catch (err: any) {
      console.error('[Agent request failed]', err);
      error.value = err?.message || 'Agent 接口请求失败，请稍后重试。';
      history.value.pop();
      throw err;
    } finally {
      loading.value = false;
      currentAssistantIndex.value = null;
    }
  }

  function reset() {
    history.value = [];
    error.value = null;
    currentAssistantIndex.value = null;
    eventLogs.value = [];
    for (const tool of Object.keys(toolUsage.value)) {
      toolUsage.value[tool] = { used: false, count: 0 };
    }
  }

  function getStreamEndpoint() {
    const base = (import.meta.env.VITE_AGENT_API_BASE || '').replace(/\/$/, '');
    const proxyBase = import.meta.env.VITE_AGENT_PROXY_TARGET ? '/api-proxy' : '';
    const resolvedBase = base || proxyBase || '';
    return `${resolvedBase}/v1/chat/agent/stream`;
  }

  async function runStreamRequest(payload: AgentRequest) {
    const endpoint = getStreamEndpoint();
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Agent 接口异常: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const segments = buffer.split('\n');
      buffer = segments.pop() ?? '';
      for (const segment of segments) {
        if (!segment.trim()) continue;
        handleStreamEvent(segment);
      }
    }
  }

  function handleStreamEvent(payload: string) {
    try {
      const event = JSON.parse(payload) as {
        event: string;
        message?: AgentMessage;
        result?: { messages?: AgentMessage[] };
        error?: string;
        tool_name?: string;
        type?: string;
        delta?: string;
        content?: string;
      };

      // 处理流式 delta 更新（逐字输出）
      if (event.event === 'delta' || (event.delta !== undefined)) {
        if (currentAssistantIndex.value === null) {
          const newMessage: AgentMessage = { role: 'assistant', content: event.delta || '' };
          history.value.push(newMessage);
          currentAssistantIndex.value = history.value.length - 1;
        } else {
          const current = history.value[currentAssistantIndex.value];
          current.content += event.delta || '';
        }
        return;
      }

      // 处理完整消息更新
      if (event.event === 'update' && event.message) {
        // 助手消息
        if (event.type === 'assistant' || event.message.role === 'assistant') {
          if (currentAssistantIndex.value === null) {
            history.value.push({ ...event.message });
            currentAssistantIndex.value = history.value.length - 1;
          } else {
            const current = history.value[currentAssistantIndex.value];
            current.content = event.message.content;
          }
          pushLog('assistant', '模型返回新内容');
        }

        // 工具调用消息
        if (event.type === 'tool') {
          const toolName = event.tool_name || 'unknown_tool';
          registerToolUse(toolName, event.message.content);

          // 添加工具结果消息到历史记录
          history.value.push({ role: 'tool', content: event.message.content });
          // 重置当前助手索引，因为下一条消息会是新的助手消息
          currentAssistantIndex.value = null;
        }
        return;
      }

      // 处理完成事件
      if (event.event === 'complete' || event.event === 'done') {
        if (event.result?.messages?.length) {
          history.value = event.result.messages;
        }
        pushLog('complete', '对话完成');
        currentAssistantIndex.value = null;
        return;
      }

      // 处理错误事件 (支持 error 或 message 字段)
      if (event.event === 'error') {
        error.value = event.error || (event as any).message || 'Agent 执行失败';
        pushLog('error', error.value);
        currentAssistantIndex.value = null;
      }
    } catch (err) {
      console.error('解析流事件失败', err, payload);
    }
  }

  function registerToolUse(toolName: string, detail: string, messageIndex?: number) {
    const idx = messageIndex !== undefined ? messageIndex : history.value.length - 1;
    if (!toolUsage.value[toolName]) {
      toolUsage.value[toolName] = { used: true, count: 1, lastMessage: detail, messageIndex: idx };
    } else {
      const tool = toolUsage.value[toolName];
      tool.used = true;
      tool.count += 1;
      tool.lastMessage = detail;
      tool.messageIndex = idx;
    }
    pushLog('tool', `${toolName} 被调用`, idx);
  }

  function pushLog(type: string, content: string, messageIndex?: number) {
    const timestamp = new Date().toLocaleTimeString();
    const idx = messageIndex !== undefined ? messageIndex : history.value.length - 1;
    eventLogs.value.unshift({ type, content, timestamp, messageIndex: idx });
    if (eventLogs.value.length > 50) {
      eventLogs.value.pop();
    }
  }

  return {
    history,
    loading,
    error,
    ask,
    reset,
    useTools,
    toolUsage,
    eventLogs,
  };
}
