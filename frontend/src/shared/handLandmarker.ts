import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// Where the browser gets MediaPipe's WASM runtime and the hand_landmarker model bundle. Neither is built into the app:
// models/*.task is git-ignored (and its redistribution terms are not stated by Google), so by default the browser fetches
// both from their official hosts. Self-host them and point these variables at your copy to avoid the third-party requests.
const WASM_BASE =
  (import.meta.env.VITE_MEDIAPIPE_WASM_BASE as string | undefined)?.trim() ||
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm";
const MODEL_URL =
  (import.meta.env.VITE_HAND_LANDMARKER_MODEL_URL as string | undefined)?.trim() ||
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

/** One persistent VIDEO-mode landmarker (tracks the hand between frames). The caller owns it and must call close(). */
export async function createVideoHandLandmarker(numHands = 1): Promise<HandLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO" as const,
    numHands,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    return HandLandmarker.createFromOptions(fileset, options("CPU")); // no usable GPU delegate on this machine
  }
}
