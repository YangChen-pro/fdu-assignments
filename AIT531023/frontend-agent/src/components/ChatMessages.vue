<template>
  <div class="chat-messages" ref="container">
    <div
      v-for="(msg, idx) in messages"
      :key="idx"
      :ref="el => setMessageRef(el, idx)"
      :data-message-index="idx"
      :class="['bubble', msg.role, { streaming: isLastAssistant(idx) }]"
    >
      <header>
        {{ msg.role === 'user' ? '用户' : msg.role === 'assistant' ? 'Agent' : 'Tool' }}
        <span v-if="isLastAssistant(idx)" class="streaming-indicator">
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
        </span>
      </header>

      <!-- 用户消息 -->
      <div v-if="msg.role === 'user'" class="bubble__content" v-html="renderMarkdown(msg.content)"></div>

      <!-- 工具消息 -->
      <div v-else-if="msg.role === 'tool'" class="bubble__content tool-result">
        <div class="tool-result-label">🔧 工具返回结果</div>
        <div class="tool-result-wrapper">
          <div
            :class="['tool-result-content', { collapsed: !expandedToolResults[idx] && isToolResultLong(msg.content) }]"
            v-html="renderMarkdown(msg.content)"
          ></div>
          <button
            v-if="isToolResultLong(msg.content)"
            class="tool-result-toggle"
            @click="toggleToolResult(idx)"
          >
            <span class="toggle-icon">{{ expandedToolResults[idx] ? '▲' : '▼' }}</span>
            {{ expandedToolResults[idx] ? '收起' : '展开全部' }}
          </button>
        </div>
      </div>

      <!-- 助手消息（解析后） -->
      <div v-else-if="msg.role === 'assistant'" class="bubble__content assistant-content">
        <template v-if="parsedMessages[idx]">
          <!-- 工具调用 -->
          <div v-for="(toolCall, tcIdx) in parsedMessages[idx].toolCalls" :key="tcIdx" class="tool-call-block">
            <div class="tool-call-header">
              <span class="tool-icon">🛠️</span>
              <span class="tool-name">{{ toolCall.name }}</span>
            </div>
            <pre class="tool-call-json"><code>{{ formatToolCall(toolCall) }}</code></pre>
          </div>

          <!-- 思考过程（可折叠） -->
          <div v-if="parsedMessages[idx].thinking" class="thinking-block">
            <button
              class="thinking-toggle"
              @click="toggleThinking(idx)"
              :class="{ expanded: expandedThinking[idx] }"
            >
              <span class="toggle-icon">{{ expandedThinking[idx] ? '▼' : '▶' }}</span>
              思考过程
              <span class="thinking-badge">AI 推理中</span>
            </button>
            <div v-show="expandedThinking[idx]" class="thinking-content" v-html="renderMarkdown(parsedMessages[idx].thinking!)"></div>
          </div>

          <!-- 最终答案 -->
          <div v-if="parsedMessages[idx].answer" class="answer-block">
            <div class="answer-header">
              <span class="answer-icon">✨</span>
              <span class="answer-label">最终答案</span>
            </div>
            <div class="answer-content" v-html="renderMarkdown(parsedMessages[idx].answer!)"></div>
          </div>

          <!-- 如果没有结构化内容，显示原始内容或作为最终回复 -->
          <div v-if="!parsedMessages[idx].toolCalls.length && !parsedMessages[idx].thinking && !parsedMessages[idx].answer && msg.content.trim()">
            <!-- 如果是对话中最后一条助手消息且对话已完成，显示为答案块 -->
            <div v-if="!isStreaming && isLastAssistantMessage(idx)" class="answer-block fallback-answer">
              <div class="answer-header">
                <span class="answer-icon">💬</span>
                <span class="answer-label">Agent 回复</span>
              </div>
              <div class="answer-content" v-html="renderMarkdown(msg.content)"></div>
            </div>
            <!-- 否则显示为普通内容 -->
            <div v-else v-html="renderMarkdown(msg.content)"></div>
          </div>
        </template>
      </div>

      <span v-if="isLastAssistant(idx)" class="typing-cursor">|</span>
    </div>
    <div v-if="messages.length === 0" class="empty-hint">
      请输入你的问题，Agent 将通过搜索、访问网页等工具一步步完成调研。
    </div>
  </div>
</template>

<script setup lang="ts">
import { onUpdated, ref, computed, watch, nextTick } from 'vue';
import type { AgentMessage } from '@/api/client';
import { renderMarkdown } from '@/utils/markdown';
import { parseAssistantMessage, formatToolCall } from '@/utils/messageParser';

const props = defineProps<{
  messages: AgentMessage[];
  isStreaming?: boolean;
}>();

const container = ref<HTMLElement | null>(null);
const messageRefs = ref<Map<number, HTMLElement>>(new Map());
const expandedThinking = ref<Record<number, boolean>>({});
const expandedToolResults = ref<Record<number, boolean>>({});

// 解析所有助手消息
const parsedMessages = computed(() => {
  const result: Record<number, ReturnType<typeof parseAssistantMessage>> = {};
  props.messages.forEach((msg, idx) => {
    if (msg.role === 'assistant') {
      result[idx] = parseAssistantMessage(msg.content);
    }
  });
  return result;
});

// 设置消息引用
const setMessageRef = (el: any, idx: number) => {
  if (el) {
    messageRefs.value.set(idx, el as HTMLElement);
  }
};

// 切换思考过程展开/折叠
const toggleThinking = (idx: number) => {
  expandedThinking.value[idx] = !expandedThinking.value[idx];
};

// 切换工具结果展开/折叠
const toggleToolResult = (idx: number) => {
  expandedToolResults.value[idx] = !expandedToolResults.value[idx];
};

// 判断工具结果是否过长（超过 300 字符）
const isToolResultLong = (content: string) => {
  return content.length > 300;
};

// 检查是否是最后一条 assistant 消息（正在流式输出）
const isLastAssistant = (idx: number) => {
  if (!props.isStreaming) return false;
  const msg = props.messages[idx];
  if (msg.role !== 'assistant') return false;

  // 检查是否是最后一条 assistant 消息
  for (let i = idx + 1; i < props.messages.length; i++) {
    if (props.messages[i].role === 'assistant') return false;
  }
  return true;
};

// 检查是否是对话中最后一条助手消息（用于显示最终答案）
const isLastAssistantMessage = (idx: number) => {
  const msg = props.messages[idx];
  if (msg.role !== 'assistant') return false;

  // 检查是否是最后一条 assistant 消息
  for (let i = idx + 1; i < props.messages.length; i++) {
    if (props.messages[i].role === 'assistant') return false;
  }
  return true;
};

// 滚动到指定消息
const scrollToMessage = (idx: number) => {
  const element = messageRefs.value.get(idx);
  if (element && container.value) {
    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // 添加高亮效果
    element.classList.add('highlight');
    setTimeout(() => {
      element.classList.remove('highlight');
    }, 2000);
  }
};

// 检查是否有最终答案
const hasAnswer = computed(() => {
  return Object.values(parsedMessages.value).some(parsed => parsed.answer);
});

// 自动滚动逻辑
onUpdated(() => {
  if (container.value) {
    // 如果正在流式输出，始终滚动到底部
    if (props.isStreaming) {
      container.value.scrollTop = container.value.scrollHeight;
    }
  }
});

// 对话完成后滚动到底部（显示答案）
watch(() => props.isStreaming, (newVal, oldVal) => {
  if (oldVal && !newVal && hasAnswer.value) {
    // 从流式变为非流式，且有答案，滚动到底部
    nextTick(() => {
      if (container.value) {
        container.value.scrollTo({
          top: container.value.scrollHeight,
          behavior: 'smooth'
        });
      }
    });
  }
});

// 暴露方法给父组件
defineExpose({
  scrollToMessage,
});
</script>

<style scoped>
.chat-messages {
  padding: 30px 25px;
  min-height: 100%;
}

.bubble {
  margin-bottom: 24px;
  animation: messageSlideIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes messageSlideIn {
  from {
    opacity: 0;
    transform: translateY(15px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.bubble header {
  font-weight: 700;
  font-size: 11px;
  margin-bottom: 10px;
  opacity: 0.7;
  text-transform: uppercase;
  letter-spacing: 1px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.bubble header::before {
  content: '●';
  font-size: 8px;
}

/* 用户消息 */
.bubble.user {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.bubble.user header {
  color: rgba(255, 255, 255, 0.6);
}

.bubble.user .bubble__content {
  max-width: 75%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 16px 20px;
  border-radius: 20px 20px 4px 20px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  word-wrap: break-word;
  position: relative;
}

.bubble.user .bubble__content::after {
  content: '';
  position: absolute;
  bottom: 0;
  right: -8px;
  width: 0;
  height: 0;
  border-left: 8px solid #764ba2;
  border-bottom: 8px solid transparent;
}

/* AI 消息 */
.bubble.assistant {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.bubble.assistant header {
  color: #667eea;
}

.bubble.assistant .bubble__content {
  max-width: 85%;
  background: linear-gradient(145deg, rgba(22, 33, 62, 0.95) 0%, rgba(30, 41, 80, 0.9) 100%);
  backdrop-filter: blur(10px);
  padding: 18px 22px;
  border-radius: 4px 20px 20px 20px;
  border-left: 4px solid #667eea;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  position: relative;
}

.bubble.assistant .bubble__content::before {
  content: '';
  position: absolute;
  top: 0;
  left: -4px;
  width: 4px;
  height: 40px;
  background: linear-gradient(180deg, #667eea, #764ba2);
  border-radius: 4px 0 0 0;
}

/* 内容样式 */
.bubble__content :deep(p) {
  margin: 10px 0;
  line-height: 1.7;
  color: inherit;
}

.bubble__content :deep(p:first-child) {
  margin-top: 0;
}

.bubble__content :deep(p:last-child) {
  margin-bottom: 0;
}

.bubble__content :deep(strong) {
  font-weight: 700;
  color: inherit;
}

.bubble__content :deep(em) {
  font-style: italic;
  color: inherit;
}

/* 代码块 */
.bubble__content :deep(pre) {
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 16px;
  border-radius: 10px;
  overflow-x: auto;
  margin: 16px 0;
  font-size: 13px;
  font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
  line-height: 1.5;
  box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
}

.bubble__content :deep(code) {
  background: rgba(102, 126, 234, 0.15);
  color: #a8dadc;
  padding: 3px 8px;
  border-radius: 5px;
  font-size: 0.9em;
  font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
  border: 1px solid rgba(102, 126, 234, 0.2);
}

.bubble__content :deep(pre code) {
  background: none;
  padding: 0;
  border: none;
  color: #e2e8f0;
}

/* 列表 */
.bubble__content :deep(ul),
.bubble__content :deep(ol) {
  margin: 12px 0;
  padding-left: 24px;
}

.bubble__content :deep(li) {
  margin: 6px 0;
  line-height: 1.6;
}

/* 表格 */
.bubble__content :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 16px 0;
  font-size: 13px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.bubble__content :deep(th),
.bubble__content :deep(td) {
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 10px 14px;
  text-align: left;
}

.bubble__content :deep(th) {
  background: rgba(102, 126, 234, 0.2);
  font-weight: 700;
  color: #b8c5ff;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.5px;
}

.bubble__content :deep(tr:nth-child(even)) {
  background: rgba(255, 255, 255, 0.02);
}

.bubble__content :deep(tr:hover) {
  background: rgba(102, 126, 234, 0.08);
}

/* 链接 */
.bubble__content :deep(a) {
  color: #667eea;
  text-decoration: none;
  border-bottom: 1px dashed rgba(102, 126, 234, 0.4);
  transition: all 0.2s;
  font-weight: 500;
}

.bubble__content :deep(a:hover) {
  color: #8b9bff;
  border-bottom-style: solid;
  border-bottom-color: #8b9bff;
}

/* 引用 */
.bubble__content :deep(blockquote) {
  border-left: 3px solid #667eea;
  padding-left: 16px;
  margin: 16px 0;
  color: rgba(255, 255, 255, 0.7);
  font-style: italic;
}

/* 空状态 */
.empty-hint {
  text-align: center;
  color: rgba(255, 255, 255, 0.3);
  font-size: 15px;
  padding: 80px 30px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.empty-hint::before {
  content: '💬';
  font-size: 48px;
  opacity: 0.5;
}

/* 流式输出指示器 */
.streaming-indicator {
  display: inline-flex;
  gap: 4px;
  margin-left: 8px;
  align-items: center;
}

.streaming-indicator .dot {
  width: 4px;
  height: 4px;
  background: currentColor;
  border-radius: 50%;
  animation: dotPulse 1.4s ease-in-out infinite;
}

.streaming-indicator .dot:nth-child(1) {
  animation-delay: 0s;
}

.streaming-indicator .dot:nth-child(2) {
  animation-delay: 0.2s;
}

.streaming-indicator .dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes dotPulse {
  0%, 100% {
    opacity: 0.3;
    transform: scale(0.8);
  }
  50% {
    opacity: 1;
    transform: scale(1.2);
  }
}

/* 打字光标 */
.typing-cursor {
  display: inline-block;
  margin-left: 4px;
  color: #667eea;
  font-weight: 400;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  0%, 50% {
    opacity: 1;
  }
  51%, 100% {
    opacity: 0;
  }
}

/* ==================== 工具调用块 ==================== */
.tool-call-block {
  margin: 16px 0;
  background: rgba(102, 126, 234, 0.08);
  border-left: 4px solid #667eea;
  border-radius: 8px;
  overflow: hidden;
}

.tool-call-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: rgba(102, 126, 234, 0.12);
  border-bottom: 1px solid rgba(102, 126, 234, 0.2);
}

.tool-icon {
  font-size: 16px;
}

.tool-name {
  font-weight: 700;
  color: #667eea;
  text-transform: uppercase;
  font-size: 13px;
  letter-spacing: 0.5px;
}

.tool-call-json {
  margin: 0;
  padding: 14px;
  background: rgba(0, 0, 0, 0.3);
  overflow-x: auto;
}

.tool-call-json code {
  font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.6;
  color: #a8dadc;
}

/* ==================== 思考过程块 ==================== */
.thinking-block {
  margin: 16px 0;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(255, 193, 7, 0.05);
  border: 1px solid rgba(255, 193, 7, 0.2);
}

.thinking-toggle {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: rgba(255, 193, 7, 0.08);
  border: none;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: #ffc107;
  transition: all 0.3s;
  text-align: left;
}

.thinking-toggle:hover {
  background: rgba(255, 193, 7, 0.15);
}

.toggle-icon {
  font-size: 10px;
  transition: transform 0.3s;
}

.thinking-toggle.expanded .toggle-icon {
  transform: rotate(0deg);
}

.thinking-badge {
  margin-left: auto;
  font-size: 10px;
  background: rgba(255, 193, 7, 0.2);
  padding: 3px 8px;
  border-radius: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.thinking-content {
  padding: 14px;
  color: rgba(255, 255, 255, 0.8);
  font-size: 13px;
  line-height: 1.6;
  animation: slideDown 0.3s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    max-height: 0;
  }
  to {
    opacity: 1;
    max-height: 1000px;
  }
}

/* ==================== 最终答案块 ==================== */
.answer-block {
  margin: 20px 0;
  padding: 20px;
  background: linear-gradient(135deg, rgba(40, 167, 69, 0.15) 0%, rgba(32, 201, 151, 0.1) 100%);
  border-left: 4px solid #28a745;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(40, 167, 69, 0.2);
  animation: answerFadeIn 0.5s ease-out;
}

@keyframes answerFadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.answer-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 2px solid rgba(40, 167, 69, 0.3);
}

.answer-icon {
  font-size: 20px;
}

.answer-label {
  font-size: 15px;
  font-weight: 700;
  color: #28a745;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.answer-content {
  color: #eee;
  font-size: 14px;
  line-height: 1.8;
}

.answer-content :deep(p) {
  margin: 10px 0;
}

.answer-content :deep(strong) {
  color: #28a745;
}

/* 降级的答案块（没有 <answer> 标签的最后回复） */
.fallback-answer {
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%);
  border-left-color: #667eea;
  box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
}

.fallback-answer .answer-label {
  color: #667eea;
}

.fallback-answer .answer-header {
  border-bottom-color: rgba(102, 126, 234, 0.3);
}

/* ==================== 工具结果样式 ==================== */
.tool-result {
  background: rgba(23, 162, 184, 0.05);
  border-left: 4px solid #17a2b8;
  border-radius: 8px;
  padding: 0;
  overflow: hidden;
}

.tool-result-label {
  padding: 10px 14px;
  background: rgba(23, 162, 184, 0.12);
  font-weight: 600;
  color: #17a2b8;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 1px solid rgba(23, 162, 184, 0.2);
}

.tool-result-wrapper {
  position: relative;
}

.tool-result-content {
  padding: 14px;
  color: rgba(255, 255, 255, 0.85);
  font-size: 13px;
  line-height: 1.6;
  transition: all 0.3s ease;
}

.tool-result-content.collapsed {
  max-height: 200px;
  overflow: hidden;
  position: relative;
}

.tool-result-content.collapsed::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 80px;
  background: linear-gradient(to bottom, transparent, rgba(22, 33, 62, 0.95));
  pointer-events: none;
}

.tool-result-toggle {
  width: 100%;
  padding: 10px 14px;
  background: rgba(23, 162, 184, 0.08);
  border: none;
  border-top: 1px solid rgba(23, 162, 184, 0.2);
  color: #17a2b8;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  transition: all 0.3s;
}

.tool-result-toggle:hover {
  background: rgba(23, 162, 184, 0.15);
}

.tool-result-toggle .toggle-icon {
  font-size: 10px;
  transition: transform 0.3s;
}

/* ==================== 高亮效果 ==================== */
.bubble.highlight {
  animation: highlightPulse 2s ease-out;
}

@keyframes highlightPulse {
  0% {
    box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);
    transform: scale(1);
  }
  50% {
    box-shadow: 0 0 30px 10px rgba(102, 126, 234, 0.3);
    transform: scale(1.02);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
    transform: scale(1);
  }
}
</style>
