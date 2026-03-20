<template>
  <form class="chat-input" @submit.prevent="handleSubmit">
    <textarea
      v-model="draft"
      placeholder="请输入你的问题，例如：帮我调研电动汽车市场趋势……"
      :disabled="disabled"
      rows="3"
    />
    <button type="submit" :disabled="disabled || !draft.trim()">发送</button>
  </form>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const props = defineProps<{ disabled?: boolean }>();
const emit = defineEmits<{ (e: 'send', value: string): void }>();

const draft = ref('');

function handleSubmit() {
  if (!draft.value.trim() || props.disabled) return;
  emit('send', draft.value.trim());
  draft.value = '';
}
</script>

<style scoped>
.chat-input {
  display: flex;
  gap: 12px;
  align-items: flex-end;
  position: relative;
}

textarea {
  flex: 1;
  background: rgba(26, 26, 46, 0.6);
  backdrop-filter: blur(10px);
  border: 2px solid rgba(102, 126, 234, 0.2);
  color: #eee;
  padding: 14px 20px;
  border-radius: 16px;
  font-size: 14px;
  font-family: inherit;
  outline: none;
  resize: none;
  line-height: 1.5;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

textarea::placeholder {
  color: rgba(255, 255, 255, 0.3);
}

textarea:focus {
  border-color: #667eea;
  background: rgba(26, 26, 46, 0.9);
  box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2), 0 0 0 4px rgba(102, 126, 234, 0.1);
  transform: translateY(-1px);
}

textarea:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

button {
  padding: 14px 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 14px;
  font-weight: 700;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  position: relative;
  overflow: hidden;
}

button::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s;
}

button:hover:not(:disabled)::before {
  left: 100%;
}

button:hover:not(:disabled) {
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
}

button:active:not(:disabled) {
  transform: translateY(-1px);
}

button:disabled {
  background: linear-gradient(135deg, #555 0%, #666 100%);
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
  opacity: 0.6;
}

/* 发送动画 */
@keyframes sendPulse {
  0% {
    box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
  }
}
</style>
