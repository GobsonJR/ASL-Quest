import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { CameraPractice } from "./CameraPractice";
import { predictImage, type PredictResponse } from "../api";

// The real network call is always mocked here -- these tests exercise the
// actual prediction-loop locking logic in CameraPractice, not a stand-in.
vi.mock("../api", () => ({ predictImage: vi.fn() }));

function response(overrides: Partial<PredictResponse> = {}): PredictResponse {
  return {
    status: "ok",
    prediction: "A",
    confidence: 0.95,
    top_predictions: [],
    low_confidence: false,
    threshold: 0.5,
    hand_detected: true,
    ...overrides,
  };
}

function fakeStream(): MediaStream {
  return { getTracks: () => [{ stop: vi.fn() }] } as unknown as MediaStream;
}

// The real prediction loop has no artificial delay of its own -- it's paced
// only by how long each `await predictImage(...)` takes. A mock that resolves
// on the same microtask starves the event loop (nothing else, including
// `waitFor`'s polling, ever gets a turn), so every mocked response here goes
// through a real macrotask tick, exactly like a real network request would.
function respondAfterTick(value: PredictResponse) {
  return () => new Promise<PredictResponse>((resolve) => setTimeout(() => resolve(value), 4));
}

async function settleVideo() {
  const video = screen.getByLabelText(/webcam preview/i) as HTMLVideoElement;
  Object.defineProperty(video, "readyState", { value: 4, configurable: true });
  Object.defineProperty(video, "videoWidth", { value: 640, configurable: true });
  Object.defineProperty(video, "videoHeight", { value: 480, configurable: true });
}

beforeEach(() => {
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  // jsdom has no real canvas 2D backend -- stub just enough of the surface
  // CameraPractice touches (drawImage for the capture canvas, strokeRect for
  // the hand-box overlay) so the real prediction loop can run end to end.
  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
    drawImage: vi.fn(),
    clearRect: vi.fn(),
    strokeRect: vi.fn(),
    strokeStyle: "",
    lineWidth: 0,
  }) as unknown as HTMLCanvasElement["getContext"];
  HTMLCanvasElement.prototype.toBlob = vi.fn(function toBlob(callback: BlobCallback) {
    callback(new Blob(["frame"], { type: "image/jpeg" }));
  }) as unknown as HTMLCanvasElement["toBlob"];
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue(fakeStream()) },
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("CameraPractice correct-detection locking", () => {
  it("fires onCorrect exactly once and stops predicting once the target letter is confirmed", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(response({ prediction: "A" })));
    const onCorrect = vi.fn();
    const onIncorrect = vi.fn();

    const { unmount } = render(
      <CameraPractice
        targetLetter="A"
        active
        ready
        backendError={null}
        onCorrect={onCorrect}
        onIncorrect={onIncorrect}
      />
    );
    await settleVideo();

    await waitFor(() => expect(onCorrect).toHaveBeenCalledTimes(1));
    const callsAfterFirst = vi.mocked(predictImage).mock.calls.length;

    // Give the loop several more macrotask turns -- if the lock didn't hold,
    // more onCorrect calls (and therefore more XP awards upstream) would show
    // up here.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(onCorrect).toHaveBeenCalledTimes(1);
    expect(onIncorrect).not.toHaveBeenCalled();
    // The while loop must have actually stopped, not merely stopped calling
    // onCorrect while still spinning.
    expect(vi.mocked(predictImage).mock.calls.length).toBeLessThanOrEqual(callsAfterFirst + 1);

    const [elapsedMs, confidence] = onCorrect.mock.calls[0];
    expect(typeof elapsedMs).toBe("number");
    expect(confidence).toBe(0.95);

    unmount();
  });
});

describe("CameraPractice wrong-detection feedback", () => {
  it("reports the stable wrong letter once (debounced) with a learner-friendly status, not raw prediction text", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(response({ prediction: "S" })));
    const onCorrect = vi.fn();
    const onIncorrect = vi.fn();
    const statuses: string[] = [];

    const { unmount } = render(
      <CameraPractice
        targetLetter="A"
        active
        ready
        backendError={null}
        onCorrect={onCorrect}
        onIncorrect={onIncorrect}
        onStatusChange={(status) => statuses.push(status)}
      />
    );
    await settleVideo();

    await waitFor(() => expect(onIncorrect).toHaveBeenCalledTimes(1));
    expect(onIncorrect).toHaveBeenCalledWith("S");
    expect(onCorrect).not.toHaveBeenCalled();

    // Keep "showing S" going for a bit longer: onIncorrect must not spam on
    // every frame once the wrong letter has already been reported once.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(onIncorrect).toHaveBeenCalledTimes(1);

    const friendlyStatus = statuses.find((s) => s.includes("S"));
    expect(friendlyStatus).toBeDefined();
    expect(friendlyStatus?.toLowerCase()).not.toContain("predicted:");

    unmount();
  });
});
