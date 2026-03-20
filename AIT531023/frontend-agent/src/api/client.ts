import axios from 'axios';

export interface AgentMessage {
  role: 'user' | 'assistant' | 'tool';
  content: string;
}

export interface AgentRequest {
  messages: AgentMessage[];
  model?: string;
  temperature?: number;
  use_tools?: boolean;
}

export interface AgentResponse {
  question: string;
  prediction?: string;
  messages?: AgentMessage[];
  termination?: string;
  [key: string]: unknown;
}

const proxyTarget = import.meta.env.VITE_AGENT_PROXY_TARGET;
const directBase = import.meta.env.VITE_AGENT_API_BASE;

const baseURL = proxyTarget ? '/api-proxy' : directBase || 'http://127.0.0.1:8000';

export const agentClient = axios.create({
  baseURL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

agentClient.interceptors.request.use((config) => {
  const token = import.meta.env.VITE_AGENT_API_KEY;
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export async function sendAgentRequest(payload: AgentRequest) {
  const response = await agentClient.post<AgentResponse>('/v1/chat/agent', payload);
  return response.data;
}

export interface ModelListResponse {
  data: Array<{
    id: string;
    object: string;
    owned_by?: string;
  }>;
}

export async function fetchModels() {
  const response = await agentClient.get<ModelListResponse>('/v1/models');
  return response.data.data?.map((item) => item.id) ?? [];
}
