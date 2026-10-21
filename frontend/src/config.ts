/** Base URL for the ASL Quest FastAPI backend.
 *
 * - Local development: unset -> "/api", which Vite proxies to http://127.0.0.1:8000
 *   (see vite.config.ts), so nothing changes for `npm run dev`.
 * - Production (Vercel): set VITE_API_BASE_URL to the publicly reachable backend,
 *   e.g. https://asl-quest-api.example.com  (no trailing slash needed). This value is
 *   public by design -- never put secrets in a VITE_ variable.
 */
const configured = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();

export const API_BASE: string = (configured || "/api").replace(/\/+$/, "");

/** True when a production build has no backend URL, so "/api" would hit Vercel (404), not FastAPI. */
export const BACKEND_NOT_CONFIGURED: boolean = !import.meta.env.DEV && !configured;

/** Demo mode (set VITE_DEMO_MODE=true at build time, e.g. on the Vercel project only): skips the
 * login screen and runs the UI with a local-only demo profile. Nothing is faked -- progress is the
 * app's real default state kept in this browser, and every backend-dependent feature still reports
 * that the service is unavailable. Unset (the default) leaves normal login behaviour untouched. */
export const DEMO_MODE: boolean = import.meta.env.VITE_DEMO_MODE === "true";
