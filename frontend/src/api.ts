import { API_BASE as API } from "./config";
export type TopPrediction = {
  label: string;
  confidence: number;
};

export type PredictResponse = {
  status: string;
  prediction: string | null;
  confidence: number;
  top_predictions: TopPrediction[];
  low_confidence: boolean;
  threshold: number;
  filename?: string;
  hand_detected?: boolean | null;
  hand_box?: [number, number, number, number] | null;
  hand_detection_available?: boolean;
  message?: string;
  /** Original frame size the server received [width, height]. */
  frame_size?: [number, number];
  /** Present only on answers from the experimental landmark recognizer; ResNet answers carry no such field. */
  recognizer?: "landmarks";
  /** File name of the experimental landmark model that answered. */
  landmark_model?: string;
  /** Image-free server diagnostics; only present when requested with debug: true (development). */
  diagnostics?: {
    input_size?: [number, number];
    tensor_shape?: number[];
    tensor_dtype?: string;
    device?: string;
    class_index?: number;
    inference_ms?: number;
    hand_detection_confidence?: number;
    /** True when the server runs with ASL_QUEST_DEV_DIAGNOSTICS=1. */
    capture_enabled?: boolean;
    /** Set when this request's original frame / crop / model input were written to disk (development only). */
    request_id?: string;
    saved_to?: string;
  };
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  model_path: string | null;
  classes: string[];
  error: string | null;
  device: string;
  cuda_available: boolean;
  gpu_name: string | null;
  cuda_version: string | null;
  gpu_memory_gb: number | null;
  hand_detection_available: boolean;
  hand_detection_error: string | null;
  /** The optional experimental landmark recognizer (set up with ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL). */
  experimental_landmark?: ExperimentalLandmarkInfo;
};

export type ExperimentalLandmarkInfo = {
  loaded: boolean;
  model_file: string | null;
  model: string | null;
  trained_on: { rooms?: string[] | null; n_samples?: number | null } | null;
  error: string | null;
};

/** Which engine answers a prediction. "resnet" is the production default and sends no extra parameter. */
export type RecognizerChoice = "resnet" | "landmarks";

export async function healthCheck(): Promise<HealthResponse> {
  const response = await fetch(`${API}/health`);
  if (!response.ok) {
    throw new Error("Backend is unavailable.");
  }
  return response.json();
}

export type LandmarkPrediction = PredictResponse & {
  actual_hand?: "left" | "right" | null;
  timings_ms?: { classify: number; server_total: number };
};

/** Classify 21 landmarks the browser already detected (no image upload, no server-side detection). */
export async function predictLandmarks(
  body: { landmarks: number[][]; handedness: "Left" | "Right" | null; image_mirrored: boolean },
  signal?: AbortSignal
): Promise<LandmarkPrediction> {
  const response = await fetch(`${API}/predict_landmarks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    let detail = "Prediction failed.";
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function predictImage(
  file: Blob,
  filename = "frame.jpg",
  options: {
    requireHand?: boolean;
    debug?: boolean;
    /** Development only: ask the server to save this request's inputs (needs ASL_QUEST_DEV_DIAGNOSTICS=1). */
    saveDiagnostics?: boolean;
    /** Client context recorded with a saved capture. */
    target?: string;
    previewMirrored?: boolean;
    videoSize?: string;
    /** Per-request engine; omitted (or "resnet") keeps the production image classifier. */
    recognizer?: RecognizerChoice;
  } = {}
): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file, filename);
  const params = new URLSearchParams();
  if (options.requireHand) params.set("require_hand", "true");
  if (options.debug) params.set("debug", "true");
  if (options.saveDiagnostics) params.set("save_diagnostics", "true");
  if (options.target) params.set("target", options.target);
  if (options.previewMirrored !== undefined) params.set("preview_mirrored", String(options.previewMirrored));
  if (options.videoSize) params.set("video_size", options.videoSize);
  if (options.recognizer === "landmarks") params.set("recognizer", "landmarks");
  const query = params.toString();
  const response = await fetch(`${API}/predict${query ? `?${query}` : ""}`, { method: "POST", body });
  if (!response.ok) {
    let detail = "Prediction failed.";
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return response.json();
}
