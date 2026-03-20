import { createRouter, createWebHistory } from 'vue-router';
import AgentConsole from '@/views/AgentConsole.vue';

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'console',
      component: AgentConsole,
    },
  ],
});
