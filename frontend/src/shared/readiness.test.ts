import { describe, expect, it } from "vitest";
import type { HealthResponse } from "../api";
import { recognitionReadiness, unreachableMessage } from "./readiness";

// What the live Render /health returns when the ResNet weights are absent (they are git-ignored) but the landmark model is shipped.
const health = (over: Partial<HealthResponse>): HealthResponse => ({
  status: "degraded", model_loaded: false, model_path: null, classes: [], error: "No trained model found.", device: "cpu",
  cuda_available: false, gpu_name: null, cuda_version: null, gpu_memory_gb: null, hand_detection_available: false, hand_detection_error: null,
  ...over,
});
const landmark = (loaded: boolean) => ({ loaded, model_file: loaded ? "landmark_L2_s0.joblib" : null, model: null, trained_on: null, error: null });

describe("recognitionReadiness", () => {
  it("is ready when the ResNet is missing but the landmark model is loaded", () => {
    expect(recognitionReadiness(health({ model_loaded: false, experimental_landmark: landmark(true) }), null)).toEqual({ ready: true, message: null });
  });

  it("is ready when only the ResNet is loaded (older backends have no landmark field)", () => {
    expect(recognitionReadiness(health({ model_loaded: true }), null).ready).toBe(true);
  });

  it("is NOT ready when the backend answers but neither recognizer is loaded", () => {
    const result = recognitionReadiness(health({ model_loaded: false, experimental_landmark: landmark(false) }), null);
    expect(result.ready).toBe(false);
    expect(result.message).toMatch(/no sign-recognition model is loaded/);
    expect(result.message).not.toMatch(/8000|localhost/); // never sends a visitor to a local server
  });

  it("is NOT ready when /health has no landmark field and no ResNet", () => {
    expect(recognitionReadiness(health({ model_loaded: false }), null).ready).toBe(false);
  });

  it("is NOT ready, and keeps the reason, when the API could not be reached", () => {
    expect(recognitionReadiness(null, unreachableMessage(false))).toEqual({ ready: false, message: unreachableMessage(false) });
  });

  it("is NOT ready while the health request is still pending, and says it is checking", () => {
    expect(recognitionReadiness(null, null)).toEqual({ ready: false, message: "Checking the recognition service…" });
  });

  it("does not leak server paths from the health error into the message", () => {
    const result = recognitionReadiness(health({ error: "Expected /opt/render/project/src/models/best_model.pth" }), null);
    expect(result.message).not.toMatch(/opt\/render|best_model/);
  });
});

describe("unreachableMessage", () => {
  it("mentions port 8000 only in development", () => {
    expect(unreachableMessage(true)).toMatch(/port 8000/);
    expect(unreachableMessage(false)).not.toMatch(/8000|FastAPI/);
  });
});
