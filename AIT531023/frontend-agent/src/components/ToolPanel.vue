<template>
  <section class="tools-panel">
    <h3>工具面板</h3>
    <p class="panel-subtitle">点击工具卡片跳转到对应的对话位置</p>
    <div class="tool-list">
      <article
        v-for="(info, name) in tools"
        :key="name"
        :class="['tool-card', { used: info.used, clickable: info.used }]"
        @click="handleToolClick(info)"
      >
        <header>
          <span class="status-dot" :class="{ used: info.used }"></span>
          <span class="name">{{ formatName(name) }}</span>
          <span class="count" v-if="info.count">×{{ info.count }}</span>
        </header>
        <p class="desc">
          {{ info.used ? truncate(info.lastMessage || '已完成一次调用') : '等待触发' }}
        </p>
        <span v-if="info.used" class="jump-hint">点击跳转</span>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
import type { PropType } from 'vue';

const props = defineProps({
  tools: {
    type: Object as PropType<Record<string, { used: boolean; count: number; lastMessage?: string; messageIndex?: number }>>,
    required: true,
  },
});

const emit = defineEmits<{
  jumpToMessage: [messageIndex: number];
}>();

function formatName(name: string) {
  switch (name) {
    case 'google_scholar':
      return 'Google Scholar';
    case 'PythonInterpreter':
      return 'Python Interpreter';
    default:
      return name.charAt(0).toUpperCase() + name.slice(1);
  }
}

function truncate(text: string) {
  if (text.length <= 80) return text;
  return text.slice(0, 80) + '...';
}

function handleToolClick(info: { used: boolean; count: number; lastMessage?: string; messageIndex?: number }) {
  if (info.used && info.messageIndex !== undefined) {
    emit('jumpToMessage', info.messageIndex);
  }
}
</script>

<style scoped>
.tools-panel {
  padding: 0;
}

.panel-subtitle {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
  padding: 0 20px 15px 20px;
  line-height: 1.5;
}

.tool-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 0 15px 15px 15px;
}

.tool-card {
  background: rgba(26, 26, 46, 0.5);
  backdrop-filter: blur(10px);
  padding: 14px 16px;
  border-radius: 12px;
  border-left: 3px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.tool-card.clickable {
  cursor: pointer;
}

.tool-card:not(.clickable) {
  cursor: default;
}

.tool-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}

.tool-card:hover::before {
  opacity: 1;
}

.tool-card:hover {
  background: rgba(32, 38, 62, 0.8);
  border-left-color: #667eea;
  transform: translateX(4px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.tool-card.used {
  border-left-color: #28a745;
  background: rgba(30, 42, 58, 0.6);
}

.tool-card.used::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 3px;
  height: 100%;
  background: linear-gradient(180deg, #28a745, #20c997);
  box-shadow: 0 0 10px rgba(40, 167, 69, 0.5);
}

.tool-card header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 14px;
  margin-bottom: 6px;
  position: relative;
  z-index: 1;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  border: 2px solid rgba(255, 255, 255, 0.3);
  transition: all 0.3s;
}

.status-dot.used {
  background: #28a745;
  border-color: #28a745;
  box-shadow: 0 0 12px rgba(40, 167, 69, 0.8), 0 0 0 3px rgba(40, 167, 69, 0.2);
  animation: dotPulse 2s ease-in-out infinite;
}

@keyframes dotPulse {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.8;
  }
}

.name {
  color: #eee;
  flex: 1;
}

.count {
  font-size: 11px;
  font-weight: 700;
  color: #667eea;
  background: rgba(102, 126, 234, 0.2);
  padding: 4px 10px;
  border-radius: 10px;
  border: 1px solid rgba(102, 126, 234, 0.3);
}

.desc {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
  line-height: 1.5;
  position: relative;
  z-index: 1;
  margin-bottom: 4px;
}

.jump-hint {
  font-size: 10px;
  color: #667eea;
  font-weight: 600;
  opacity: 0;
  transition: opacity 0.3s;
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.jump-hint::before {
  content: '→';
  font-size: 12px;
}

.tool-card.clickable:hover .jump-hint {
  opacity: 1;
}

/* 工具卡片加载动画 */
.tool-card {
  animation: toolSlideIn 0.4s ease-out backwards;
}

.tool-card:nth-child(1) { animation-delay: 0.05s; }
.tool-card:nth-child(2) { animation-delay: 0.1s; }
.tool-card:nth-child(3) { animation-delay: 0.15s; }
.tool-card:nth-child(4) { animation-delay: 0.2s; }
.tool-card:nth-child(5) { animation-delay: 0.25s; }

@keyframes toolSlideIn {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}
</style>
