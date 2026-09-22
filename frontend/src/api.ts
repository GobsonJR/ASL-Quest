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
};

const API = "/api";

export async function healthCheck(): Promise<HealthResponse> {
  const response = await fetch(`${API}/health`);
  if (!response.ok) {
    throw new Error("Backend is unavailable.");
  }
  return response.json();
}

export async function predictImage(
  file: Blob,
  filename = "frame.jpg",
  options: { requireHand?: boolean } = {}
): Promise<PredictResponse> {
  const body = new FormData();
  body.append("file", file, filename);
  const params = new URLSearchParams();
  if (options.requireHand) params.set("require_hand", "true");
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
