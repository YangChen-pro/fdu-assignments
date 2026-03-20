<template>
  <section class="activity-panel">
    <header>
      <div>
        <p class="label">对话统计</p>
        <h3>{{ historyCount }} 条消息</h3>
      </div>
      <div class="stat">
        <p class="label">工具调用</p>
        <h3>{{ toolCount }}</h3>
      </div>
    </header>

    <div class="log-list">
      <article
        v-for="log in logs"
        :key="log.timestamp + log.content"
        :class="['log-item', log.type, { clickable: log.messageIndex !== undefined }]"
        @click="handleLogClick(log)"
      >
        <div class="meta">
          <span class="time">{{ log.timestamp }}</span>
          <span class="tag">{{ formatType(log.type) }}</span>
        </div>
        <p>{{ log.content }}</p>
        <span v-if="log.messageIndex !== undefined" class="jump-hint">点击跳转</span>
      </article>
      <p v-if="logs.length === 0" class="empty">暂无事件，开始提问吧。</p>
    </div>
  </section>
</template>

<script setup lang="ts">
import type { PropType } from 'vue';

defineProps({
  logs: {
    type: Array as PropType<Array<{ type: string; content: string; timestamp: string; messageIndex?: number }>>,
    required: true,
  },
  historyCount: {
    type: Number,
    required: true,
  },
  toolCount: {
    type: Number,
    required: true,
  },
});

const emit = defineEmits<{
  jumpToMessage: [messageIndex: number];
}>();

function formatType(type: string) {
  switch (type) {
    case 'assistant':
      return '回复';
    case 'tool':
      return '工具';
    case 'complete':
      return '完成';
    case 'error':
      return '错误';
    default:
      return type;
  }
}

function handleLogClick(log: { type: string; content: string; timestamp: string; messageIndex?: number }) {
  if (log.messageIndex !== undefined) {
    emit('jumpToMessage', log.messageIndex);
  }
}
</script>

<style scoped>
.activity-panel {
  padding: 0;
}

header {
  display: flex;
  gap: 15px;
  align-items: flex-start;
  padding: 0 20px 15px 20px;
}

header > div {
  flex: 1;
}

.label {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 700;
}

header h3 {
  font-size: 24px;
  font-weight: 700;
  margin: 0;
  color: #eee;
}

.stat {
  text-align: right;
}

.log-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 0 15px 15px 15px;
}

.log-item {
  background: rgba(26, 26, 46, 0.5);
  backdrop-filter: blur(10px);
  padding: 12px 14px;
  border-radius: 10px;
  font-size: 12px;
  border-left: 3px solid rgba(255, 255, 255, 0.1);
  animation: logSlideIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
  position: relative;
  overflow: hidden;
  transition: all 0.3s;
}

.log-item.clickable {
  cursor: pointer;
}

.log-item:not(.clickable) {
  cursor: default;
}

@keyframes logSlideIn {
  from {
    opacity: 0;
    transform: translateX(20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.log-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.03), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}

.log-item:hover::before {
  opacity: 1;
}

.log-item:hover {
  background: rgba(32, 38, 62, 0.7);
  transform: translateX(-4px);
}

.log-item.info {
  border-left-color: #17a2b8;
}

.log-item.tool {
  border-left-color: #17a2b8;
}

.log-item.tool::after {
  content: '🔧';
  position: absolute;
  top: 12px;
  right: 12px;
  font-size: 14px;
  opacity: 0.4;
}

.log-item.complete {
  border-left-color: #28a745;
}

.log-item.complete::after {
  content: '✓';
  position: absolute;
  top: 12px;
  right: 12px;
  font-size: 16px;
  color: #28a745;
  opacity: 0.6;
}

.log-item.error {
  border-left-color: #dc3545;
}

.log-item.error::after {
  content: '⚠️';
  position: absolute;
  top: 12px;
  right: 12px;
  font-size: 14px;
  opacity: 0.6;
}

.log-item.assistant {
  border-left-color: #667eea;
}

.log-item.assistant::after {
  content: '💬';
  position: absolute;
  top: 12px;
  right: 12px;
  font-size: 14px;
  opacity: 0.4;
}

.meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  gap: 8px;
}

.time {
  color: rgba(255, 255, 255, 0.4);
  font-size: 10px;
  font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
}

.tag {
  font-size: 10px;
  font-weight: 700;
  color: rgba(255, 255, 255, 0.6);
  background: rgba(255, 255, 255, 0.1);
  padding: 3px 8px;
  border-radius: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.log-item.tool .tag {
  background: rgba(23, 162, 184, 0.2);
  color: #17a2b8;
}

.log-item.complete .tag {
  background: rgba(40, 167, 69, 0.2);
  color: #28a745;
}

.log-item.error .tag {
  background: rgba(220, 53, 69, 0.2);
  color: #dc3545;
}

.log-item.assistant .tag {
  background: rgba(102, 126, 234, 0.2);
  color: #667eea;
}

.log-item p {
  color: rgba(255, 255, 255, 0.75);
  line-height: 1.5;
  margin: 0;
  padding-right: 24px;
  margin-bottom: 4px;
}

.log-item .jump-hint {
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

.log-item .jump-hint::before {
  content: '→';
  font-size: 12px;
}

.log-item.clickable:hover .jump-hint {
  opacity: 1;
}

.empty {
  text-align: center;
  color: rgba(255, 255, 255, 0.3);
  margin-top: 60px;
  font-size: 13px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.empty::before {
  content: '📋';
  font-size: 40px;
  opacity: 0.5;
}
</style>
