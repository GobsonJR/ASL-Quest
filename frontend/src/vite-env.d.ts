/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Public base URL of the FastAPI backend (production). Defaults to the dev proxy "/api". */
  readonly VITE_API_BASE_URL?: string;
  /** "true" skips login and runs with a local demo profile (frontend-only deployments). */
  readonly VITE_DEMO_MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
