import type { HealthResponse } from "../api";

export type RecognitionReadiness = {
  /** True only when the backend answered AND at least one recognizer can classify a sign. */
  ready: boolean;
  /** Why Practice is not available (null when it is ready). Shown on the Practice pages and the status line. */
  message: string | null;
};

/** Words for "the health request failed": a developer is told how to start the API, a visitor is not sent to localhost. */
export function unreachableMessage(dev: boolean = import.meta.env.DEV): string {
  return dev
    ? "Backend unavailable. Start FastAPI on port 8000."
    : "Can't reach the recognition service right now. It may be waking up, so try again in a minute.";
}

/**
 * Whether Practice can run, from the backend's /health answer.
 *
 * Practice recognizes letters from hand landmarks the browser detects, so the landmark model reported by the backend
 * (`experimental_landmark.loaded`) is enough on its own; the ResNet (`model_loaded`) is only the fallback. Ready is never
 * claimed while the answer is still pending, when the API could not be reached, or when neither recognizer is loaded.
 */
export function recognitionReadiness(health: HealthResponse | null, healthError: string | null): RecognitionReadiness {
  if (health) {
    if (health.model_loaded || health.experimental_landmark?.loaded) return { ready: true, message: null };
    return {
      ready: false,
      message: "The recognition service is running, but no sign-recognition model is loaded on it, so Practice can't check your signs yet.",
    };
  }
  if (healthError) return { ready: false, message: healthError };
  return { ready: false, message: "Checking the recognition service…" };
}
