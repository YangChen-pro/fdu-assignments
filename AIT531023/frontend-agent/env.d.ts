/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_AGENT_API_BASE?: string;
  readonly VITE_AGENT_API_KEY?: string;
  readonly VITE_AGENT_MODEL?: string;
  readonly VITE_AGENT_TEMPERATURE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
