<template>
  <div class="agent-console">
    <!-- 顶部导航栏 -->
    <header class="topbar">
      <div class="brand">
        <div class="brand-icon">🤖</div>
        <div class="brand-text">
          <h1>DeepResearch Agent</h1>
          <span class="subtitle">多轮深度调研控制台</span>
        </div>
      </div>

      <div class="topbar-center">
        <div class="stat-badge">
          <span class="stat-label">对话轮次</span>
          <span class="stat-value">{{ history.length }}</span>
        </div>
        <div class="stat-badge">
          <span class="stat-label">工具调用</span>
          <span class="stat-value">{{ toolInvokeCount }}</span>
        </div>
        <div class="stat-badge" :class="{ active: loading }">
          <span class="stat-label">状态</span>
          <span class="stat-value">{{ loading ? '思考中' : '就绪' }}</span>
        </div>
      </div>

      <div class="topbar-actions">
        <label class="switch">
          <input type="checkbox" v-model="useTools" />
          <span class="slider"></span>
          <span class="switch-label">启用工具</span>
        </label>
        <button class="btn-reset" :disabled="loading" @click="reset">
          <span class="btn-icon">🗑️</span>
          清空对话
        </button>
      </div>
    </header>

    <!-- 主内容区域 -->
    <div class="main-content">
      <!-- 左侧工具面板 -->
      <aside class="sidebar sidebar-left">
        <div class="sidebar-header">
          <h2>🛠️ 可用工具</h2>
          <span class="tools-count">{{ Object.keys(toolUsage).length }}</span>
        </div>
        <div class="sidebar-body">
          <ToolPanel :tools="toolUsage" @jump-to-message="handleJumpToMessage" />
        </div>
      </aside>

      <!-- 中间对话区域 -->
      <main class="chat-container">
        <div class="messages-wrapper">
          <ChatMessages ref="chatMessagesRef" :messages="history" :is-streaming="loading" />
        </div>
        <div class="input-container">
          <ChatInput :disabled="loading" @send="handleSend" />
          <p v-if="error" class="error-message">{{ error }}</p>
        </div>
      </main>

      <!-- 右侧活动面板 -->
      <aside class="sidebar sidebar-right">
        <div class="sidebar-header">
          <h2>📋 活动日志</h2>
          <span class="logs-count">{{ eventLogs.length }}</span>
        </div>
        <div class="sidebar-body">
          <ActivityPanel
            :logs="eventLogs"
            :history-count="history.length"
            :tool-count="toolInvokeCount"
            @jump-to-message="handleJumpToMessage"
          />
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup lang="ts">
import ChatInput from '@/components/ChatInput.vue';
import ChatMessages from '@/components/ChatMessages.vue';
import ToolPanel from '@/components/ToolPanel.vue';
import ActivityPanel from '@/components/ActivityPanel.vue';
import { useAgentChat } from '@/composables/useAgentChat';
import { computed, ref } from 'vue';

const { history, loading, error, ask, reset, useTools, toolUsage, eventLogs } = useAgentChat();
const chatMessagesRef = ref<InstanceType<typeof ChatMessages> | null>(null);

async function handleSend(message: string) {
  try {
    await ask(message);
  } catch {
    /* 错误信息已在 composable 中处理 */
  }
}

function handleJumpToMessage(messageIndex: number) {
  if (chatMessagesRef.value && messageIndex >= 0 && messageIndex < history.value.length) {
    chatMessagesRef.value.scrollToMessage(messageIndex);
  }
}

const toolInvokeCount = computed(() =>
  Object.values(toolUsage.value).reduce((sum, tool) => sum + (tool.count || 0), 0)
);
</script>

<style scoped>
/* ==================== 整体布局 ==================== */
.agent-console {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
  overflow: hidden;
}

/* ==================== 顶部导航栏 ==================== */
.topbar {
  height: 70px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 30px;
  gap: 30px;
  position: relative;
  z-index: 10;
}

.topbar::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
}

.brand {
  display: flex;
  align-items: center;
  gap: 15px;
  min-width: 280px;
}

.brand-icon {
  font-size: 32px;
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

.brand-text h1 {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: white;
  letter-spacing: -0.5px;
}

.brand-text .subtitle {
  display: block;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.8);
  margin-top: 2px;
}

.topbar-center {
  display: flex;
  gap: 15px;
  flex: 1;
  justify-content: center;
}

.stat-badge {
  background: rgba(255, 255, 255, 0.12);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 12px;
  padding: 8px 18px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  min-width: 100px;
  transition: all 0.3s ease;
}

.stat-badge:hover {
  background: rgba(255, 255, 255, 0.18);
  transform: translateY(-2px);
}

.stat-badge.active {
  background: rgba(255, 193, 7, 0.25);
  border-color: rgba(255, 193, 7, 0.4);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}

.stat-label {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.75);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 600;
}

.stat-value {
  font-size: 18px;
  font-weight: 700;
  color: white;
}

.topbar-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

/* 开关按钮 */
.switch {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  user-select: none;
}

.switch input {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: relative;
  width: 48px;
  height: 26px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 26px;
  transition: all 0.3s;
  border: 2px solid rgba(255, 255, 255, 0.3);
}

.slider::before {
  content: '';
  position: absolute;
  height: 18px;
  width: 18px;
  left: 2px;
  top: 2px;
  background: white;
  border-radius: 50%;
  transition: all 0.3s;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.switch input:checked + .slider {
  background: rgba(40, 167, 69, 0.4);
  border-color: rgba(40, 167, 69, 0.6);
}

.switch input:checked + .slider::before {
  transform: translateX(22px);
  background: #28a745;
}

.switch-label {
  font-size: 13px;
  font-weight: 500;
  color: white;
}

.btn-reset {
  background: rgba(220, 53, 69, 0.2);
  border: 1px solid rgba(220, 53, 69, 0.4);
  color: white;
  padding: 8px 18px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.3s;
}

.btn-reset:hover:not(:disabled) {
  background: rgba(220, 53, 69, 0.3);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(220, 53, 69, 0.3);
}

.btn-reset:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-icon {
  font-size: 14px;
}

/* ==================== 主内容区域 ==================== */
.main-content {
  flex: 1;
  display: grid;
  grid-template-columns: 300px 1fr 350px;
  gap: 1px;
  background: #0a0a15;
  overflow: hidden;
}

/* ==================== 侧边栏 ==================== */
.sidebar {
  background: #16213e;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-header {
  padding: 20px 20px 15px 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(0, 0, 0, 0.2);
}

.sidebar-header h2 {
  font-size: 14px;
  font-weight: 700;
  color: #667eea;
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.tools-count,
.logs-count {
  background: rgba(102, 126, 234, 0.2);
  color: #667eea;
  font-size: 11px;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 12px;
  border: 1px solid rgba(102, 126, 234, 0.3);
}

.sidebar-body {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

/* ==================== 对话容器 ==================== */
.chat-container {
  background: #0f0f1e;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
}

.chat-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 200px;
  background: radial-gradient(ellipse at top, rgba(102, 126, 234, 0.15), transparent 70%);
  pointer-events: none;
  z-index: 0;
}

.messages-wrapper {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  position: relative;
  z-index: 1;
}

.input-container {
  padding: 20px 25px;
  background: rgba(22, 33, 62, 0.8);
  backdrop-filter: blur(10px);
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.2);
}

.error-message {
  margin-top: 12px;
  padding: 12px 16px;
  background: rgba(220, 53, 69, 0.1);
  border-left: 3px solid #dc3545;
  border-radius: 8px;
  color: #ff6b6b;
  font-size: 13px;
  line-height: 1.5;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* ==================== 响应式设计 ==================== */
@media (max-width: 1400px) {
  .main-content {
    grid-template-columns: 280px 1fr 320px;
  }
}

@media (max-width: 1200px) {
  .topbar {
    height: auto;
    padding: 15px 20px;
    flex-wrap: wrap;
  }

  .brand {
    min-width: auto;
  }

  .topbar-center {
    order: 3;
    width: 100%;
    margin-top: 10px;
  }

  .main-content {
    grid-template-columns: 1fr;
    grid-template-rows: 200px 1fr 200px;
  }

  .sidebar-left,
  .sidebar-right {
    max-height: 200px;
  }
}

@media (max-width: 768px) {
  .brand-text .subtitle {
    display: none;
  }

  .stat-badge {
    min-width: 80px;
    padding: 6px 12px;
  }

  .stat-label {
    font-size: 10px;
  }

  .stat-value {
    font-size: 16px;
  }
}
</style>
