import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { LandmarkPractice } from "./LandmarkPractice";
import { predictLandmarks, type LandmarkPrediction } from "../api";
import { createVideoHandLandmarker } from "../shared/handLandmarker";

// MediaPipe, the network and the camera are all mocked; the real detection/tracking/XP-gating logic in LandmarkPractice runs.
vi.mock("@mediapipe/tasks-vision", () => ({ HandLandmarker: { HAND_CONNECTIONS: [{ start: 0, end: 1 }] } }));
vi.mock("../api", () => ({ predictLandmarks: vi.fn() }));
vi.mock("../shared/handLandmarker", () => ({ createVideoHandLandmarker: vi.fn() }));

const HAND = Array.from({ length: 21 }, (_, i) => ({ x: 0.3 + i * 0.01, y: 0.3 + i * 0.01, z: 0 }));

function prediction(overrides: Partial<LandmarkPrediction> = {}): LandmarkPrediction {
  return { status: "ok", prediction: "A", confidence: 0.95, top_predictions: [], low_confidence: false, threshold: 0.7, ...overrides };
}

// Resolve on a real macrotask, like a network request, so the animation-frame loop keeps getting turns.
const after = <T,>(value: T) => () => new Promise<T>((resolve) => setTimeout(() => resolve(value), 3));

let tracksStop: ReturnType<typeof vi.fn>;
let landmarkerClose: ReturnType<typeof vi.fn>;
let detect: ReturnType<typeof vi.fn>;

beforeEach(() => {
  tracksStop = vi.fn();
  landmarkerClose = vi.fn();
  detect = vi.fn(() => ({ landmarks: [HAND], handedness: [[{ categoryName: "Right" }]] }));
  vi.mocked(createVideoHandLandmarker).mockResolvedValue({ detectForVideo: detect, close: landmarkerClose } as never);
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  let t = 0;
  Object.defineProperty(HTMLMediaElement.prototype, "currentTime", { configurable: true, get: () => (t += 0.03) });
  Object.defineProperty(HTMLMediaElement.prototype, "readyState", { configurable: true, get: () => 4 });
  Object.defineProperty(HTMLVideoElement.prototype, "videoWidth", { configurable: true, get: () => 640 });
  Object.defineProperty(HTMLVideoElement.prototype, "videoHeight", { configurable: true, get: () => 480 });
  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
    clearRect: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(), arc: vi.fn(), fill: vi.fn(),
  }) as unknown as HTMLCanvasElement["getContext"];
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: tracksStop }] }) },
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

const baseProps = { targetLetter: "A", active: true, ready: true, backendError: null, requestIntervalMs: 0 };

describe("LandmarkPractice", () => {
  it("fires onCorrect exactly once after a stable confident target letter, sending landmarks (not images)", async () => {
    vi.mocked(predictLandmarks).mockImplementation(after(prediction()));
    const onCorrect = vi.fn();
    render(<LandmarkPractice {...baseProps} onCorrect={onCorrect} onIncorrect={vi.fn()} />);

    await waitFor(() => expect(onCorrect).toHaveBeenCalledTimes(1), { timeout: 3000 });
    await new Promise((r) => setTimeout(r, 80)); // frames that keep arriving must not award again
    expect(onCorrect).toHaveBeenCalledTimes(1);
    expect(onCorrect.mock.calls[0][1]).toBeCloseTo(0.95);

    const body = vi.mocked(predictLandmarks).mock.calls[0][0];
    expect(body.landmarks).toHaveLength(21);
    expect(body.handedness).toBe("Right");
    expect(body.image_mirrored).toBe(false);
  });

  it("resumes detection for the next letter of a word while staying active", async () => {
    vi.mocked(predictLandmarks).mockImplementation(after(prediction({ prediction: "A" })));
    const onCorrect = vi.fn();
    const { rerender } = render(<LandmarkPractice {...baseProps} targetLetter="A" onCorrect={onCorrect} onIncorrect={vi.fn()} />);
    await waitFor(() => expect(onCorrect).toHaveBeenCalledTimes(1), { timeout: 3000 });

    vi.mocked(predictLandmarks).mockImplementation(after(prediction({ prediction: "B" })));
    rerender(<LandmarkPractice {...baseProps} targetLetter="B" onCorrect={onCorrect} onIncorrect={vi.fn()} />);
    await waitFor(() => expect(onCorrect).toHaveBeenCalledTimes(2), { timeout: 3000 });
  });

  it("never has two prediction requests in flight at once", async () => {
    let concurrent = 0;
    let maxConcurrent = 0;
    vi.mocked(predictLandmarks).mockImplementation(async () => {
      concurrent += 1;
      maxConcurrent = Math.max(maxConcurrent, concurrent);
      await new Promise((r) => setTimeout(r, 15));
      concurrent -= 1;
      return prediction({ prediction: "B" });
    });
    render(<LandmarkPractice {...baseProps} onCorrect={vi.fn()} onIncorrect={vi.fn()} />);
    await waitFor(() => expect(vi.mocked(predictLandmarks).mock.calls.length).toBeGreaterThan(3), { timeout: 3000 });
    expect(maxConcurrent).toBe(1);
  });

  it("does not award on low-confidence readings or on a different stable letter", async () => {
    vi.mocked(predictLandmarks).mockImplementation(after(prediction({ low_confidence: true, confidence: 0.4 })));
    const onCorrect = vi.fn();
    const { unmount } = render(<LandmarkPractice {...baseProps} onCorrect={onCorrect} onIncorrect={vi.fn()} />);
    await waitFor(() => expect(vi.mocked(predictLandmarks).mock.calls.length).toBeGreaterThan(4), { timeout: 3000 });
    expect(onCorrect).not.toHaveBeenCalled();
    unmount();

    vi.mocked(predictLandmarks).mockReset().mockImplementation(after(prediction({ prediction: "S" })));
    const onIncorrect = vi.fn();
    render(<LandmarkPractice {...baseProps} onCorrect={onCorrect} onIncorrect={onIncorrect} />);
    await waitFor(() => expect(onIncorrect).toHaveBeenCalledWith("S"), { timeout: 3000 });
    expect(onIncorrect).toHaveBeenCalledTimes(1);
    expect(onCorrect).not.toHaveBeenCalled();
  });

  it("shows 'No hand' and sends nothing when no hand is detected", async () => {
    detect.mockImplementation(() => ({ landmarks: [], handedness: [] }));
    render(<LandmarkPractice {...baseProps} onCorrect={vi.fn()} onIncorrect={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("No hand")).toBeInTheDocument(), { timeout: 3000 });
    expect(predictLandmarks).not.toHaveBeenCalled();
  });

  it("releases the camera, MediaPipe and the pending request on unmount", async () => {
    let signal: AbortSignal | undefined;
    vi.mocked(predictLandmarks).mockImplementation((_body, s) => {
      signal = s;
      return new Promise(() => {}); // never answers
    });
    const { unmount } = render(<LandmarkPractice {...baseProps} onCorrect={vi.fn()} onIncorrect={vi.fn()} />);
    await waitFor(() => expect(signal).toBeDefined(), { timeout: 3000 });
    unmount();
    expect(signal?.aborted).toBe(true);
    expect(tracksStop).toHaveBeenCalled();
    expect(landmarkerClose).toHaveBeenCalled();
  });

  it("explains a load failure instead of showing a dead camera", async () => {
    vi.mocked(createVideoHandLandmarker).mockRejectedValue(new Error("model download failed"));
    render(<LandmarkPractice {...baseProps} onCorrect={vi.fn()} onIncorrect={vi.fn()} />);
    await waitFor(() => expect(screen.getByText(/Could not load the hand tracker/)).toBeInTheDocument());
    expect(screen.getByText(/Switch back to ResNet/)).toBeInTheDocument();
  });
});
