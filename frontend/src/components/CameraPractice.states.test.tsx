import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { CameraPractice } from "./CameraPractice";
import { predictImage, type PredictResponse } from "../api";

// The network call is mocked; everything else (state machine, smoothing, throttling, lifecycle)
// is the real CameraPractice code. Frames come from a stubbed canvas, so these tests prove the
// component's logic -- not live webcam recognition quality.
vi.mock("../api", () => ({ predictImage: vi.fn() }));

function response(overrides: Partial<PredictResponse> = {}): PredictResponse {
  return {
    status: "ok",
    prediction: "A",
    confidence: 0.95,
    top_predictions: [],
    low_confidence: false,
    threshold: 0.7,
    hand_detected: true,
    ...overrides,
  };
}

const noHand = () =>
  response({ status: "no_hand_detected", prediction: null, confidence: 0, hand_detected: false, hand_box: null });

function fakeStream(stop = vi.fn()): MediaStream {
  return { getTracks: () => [{ stop }] } as unknown as MediaStream;
}

// A mock that resolved on the same microtask would starve the event loop, so every mocked answer
// goes through a real macrotask tick, like a real network request.
function respondAfterTick(value: PredictResponse) {
  return () => new Promise<PredictResponse>((resolve) => setTimeout(() => resolve(value), 4));
}

async function settleVideo() {
  const video = screen.getByLabelText(/webcam preview/i) as HTMLVideoElement;
  Object.defineProperty(video, "readyState", { value: 4, configurable: true });
  Object.defineProperty(video, "videoWidth", { value: 640, configurable: true });
  Object.defineProperty(video, "videoHeight", { value: 480, configurable: true });
}

function renderCamera(props: Partial<ComponentProps<typeof CameraPractice>> = {}) {
  const onCorrect = vi.fn();
  const onIncorrect = vi.fn();
  const utils = render(
    <CameraPractice
      targetLetter="C"
      active
      ready
      backendError={null}
      onCorrect={onCorrect}
      onIncorrect={onIncorrect}
      frameIntervalMs={5}
      {...props}
    />
  );
  return { ...utils, onCorrect, onIncorrect };
}

beforeEach(() => {
  vi.mocked(predictImage).mockReset();
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
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

describe("CameraPractice: hand detection vs letter reading", () => {
  it("shows 'Hand detected' with an uncertain reading for a low-confidence letter, never 'No hand'", async () => {
    vi.mocked(predictImage).mockImplementation(
      respondAfterTick(response({ prediction: "S", confidence: 0.31, low_confidence: true }))
    );
    const { onCorrect, onIncorrect, unmount } = renderCamera();
    await settleVideo();

    const panel = await screen.findByTestId("recognition-status");
    await waitFor(() => expect(panel).toHaveTextContent("Hand detected"));
    expect(panel).toHaveTextContent("Unsure: S 31%");
    expect(panel).not.toHaveTextContent("No hand");
    expect(screen.getAllByText(/not sure which letter/i).length).toBeGreaterThan(0);
    expect(onCorrect).not.toHaveBeenCalled();
    expect(onIncorrect).not.toHaveBeenCalled();
    unmount();
  });

  it("shows 'No hand' and nothing to read when the server found no hand", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(noHand()));
    const { unmount } = renderCamera();
    await settleVideo();

    const panel = await screen.findByTestId("recognition-status");
    await waitFor(() => expect(panel).toHaveTextContent("No hand"));
    expect(panel).toHaveTextContent("Reading: —");
    expect(screen.getAllByText(/no hand detected/i).length).toBeGreaterThan(0);
    unmount();
  });

  it("reports an unavailable hand detector instead of pretending there is no hand", async () => {
    vi.mocked(predictImage).mockImplementation(
      respondAfterTick(
        response({ status: "hand_detector_unavailable", prediction: null, hand_detected: false, message: "Hand detector offline" })
      )
    );
    const { unmount } = renderCamera();
    await settleVideo();
    await waitFor(() => expect(screen.getAllByText("Hand detector offline").length).toBeGreaterThan(0));
    expect(screen.getByTestId("recognition-status")).toHaveTextContent("Hand detector off");
    unmount();
  });

  it("shows troubleshooting advice only after no hand has been seen for a while", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(noHand()));
    const { unmount } = renderCamera();
    await settleVideo();
    expect(screen.queryByRole("note")).toBeNull();
    expect(await screen.findByRole("note", {}, { timeout: 3000 })).toHaveTextContent(/lighting/i);
    unmount();
  });

  it("confirms the target letter only after several agreeing confident frames, even with a dropped frame between", async () => {
    const answers = [response({ prediction: "C" }), noHand(), response({ prediction: "C" }), response({ prediction: "C" }), response({ prediction: "C" })];
    let i = 0;
    vi.mocked(predictImage).mockImplementation(() => respondAfterTick(answers[Math.min(i++, answers.length - 1)])());
    const { onCorrect, unmount } = renderCamera();
    await settleVideo();
    await waitFor(() => expect(onCorrect).toHaveBeenCalledTimes(1));
    unmount();
  });
});

describe("CameraPractice: backend failures", () => {
  it("surfaces a backend error, keeps retrying, and recovers when the backend returns", async () => {
    let calls = 0;
    vi.mocked(predictImage).mockImplementation(() => {
      calls += 1;
      if (calls <= 2) {
        return new Promise<PredictResponse>((_, reject) => setTimeout(() => reject(new Error("Model is not loaded.")), 4));
      }
      return respondAfterTick(response({ prediction: "S", confidence: 0.2, low_confidence: true }))();
    });
    const { unmount } = renderCamera();
    await settleVideo();

    expect(await screen.findByRole("alert")).toHaveTextContent("Model is not loaded.");
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull(), { timeout: 3000 });
    expect(screen.getByTestId("recognition-status")).toHaveTextContent("Hand detected");
    unmount();
  });
});

describe("CameraPractice: frame handling", () => {
  it("does not send frames until the video is actually ready", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(response()));
    const { unmount } = renderCamera(); // settleVideo() deliberately not called: readyState stays 0
    await new Promise((resolve) => setTimeout(resolve, 120));
    expect(predictImage).not.toHaveBeenCalled();
    unmount();
  });

  it("throttles requests instead of running as fast as responses return", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(noHand()));
    const { unmount } = renderCamera({ frameIntervalMs: 100 });
    await settleVideo();
    await new Promise((resolve) => setTimeout(resolve, 450));
    const calls = vi.mocked(predictImage).mock.calls.length;
    expect(calls).toBeGreaterThan(0);
    expect(calls).toBeLessThanOrEqual(6); // ~450ms / 100ms; unthrottled would be ~100
    unmount();
  });

  it("never has two requests in flight at once", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    vi.mocked(predictImage).mockImplementation(() => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise<PredictResponse>((resolve) =>
        setTimeout(() => {
          inFlight -= 1;
          resolve(noHand());
        }, 15)
      );
    });
    const { unmount } = renderCamera();
    await settleVideo();
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(maxInFlight).toBe(1);
    unmount();
  });

  it("discards a late response that belongs to a previous attempt", async () => {
    let resolveFirst!: (value: PredictResponse) => void;
    let calls = 0;
    vi.mocked(predictImage).mockImplementation(() => {
      calls += 1;
      if (calls === 1) return new Promise<PredictResponse>((resolve) => (resolveFirst = resolve));
      return respondAfterTick(noHand())();
    });
    const props = {
      targetLetter: "C",
      active: true,
      ready: true,
      backendError: null,
      onCorrect: vi.fn(),
      onIncorrect: vi.fn(),
      frameIntervalMs: 5,
    };
    const { rerender, unmount } = render(<CameraPractice {...props} attemptKey={0} />);
    await settleVideo();
    await waitFor(() => expect(calls).toBe(1));

    rerender(<CameraPractice {...props} attemptKey={1} />); // "Practice again" while a request is in flight
    await act(async () => {
      resolveFirst(response({ prediction: "Z", confidence: 0.99 }));
    });
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(screen.getByTestId("recognition-status")).not.toHaveTextContent("Z 99%");
    unmount();
  });
});

describe("CameraPractice: camera lifecycle", () => {
  it("stops every track and the prediction loop on unmount, and works again after remounting", async () => {
    const stop = vi.fn();
    const getUserMedia = vi.fn().mockImplementation(async () => fakeStream(stop));
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
    vi.mocked(predictImage).mockImplementation(respondAfterTick(noHand()));

    const first = renderCamera();
    await settleVideo();
    await waitFor(() => expect(predictImage).toHaveBeenCalled());
    first.unmount();
    expect(stop).toHaveBeenCalledTimes(1);

    const callsAtUnmount = vi.mocked(predictImage).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(vi.mocked(predictImage).mock.calls.length).toBeLessThanOrEqual(callsAtUnmount + 1); // loop is gone

    const second = renderCamera();
    await settleVideo();
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(vi.mocked(predictImage).mock.calls.length).toBeGreaterThan(callsAtUnmount + 1));
    second.unmount();
    expect(stop).toHaveBeenCalledTimes(2);
  });

  it("reports a denied camera permission without starting recognition", async () => {
    const denied = new DOMException("denied", "NotAllowedError");
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(denied) },
    });
    const { unmount } = renderCamera();
    expect(await screen.findByText(/camera access was denied/i)).toBeInTheDocument();
    expect(predictImage).not.toHaveBeenCalled();
    unmount();
  });
});

describe("CameraPractice: development frame capture", () => {
  it("saves the exact frame that was sent, named with the target letter", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(noHand()));
    const createObjectURL = vi.fn((_blob: Blob) => "blob:frame");
    const revokeObjectURL = vi.fn();
    Object.assign(URL, { createObjectURL, revokeObjectURL });
    const clicks: string[] = [];
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      clicks.push(this.download);
    });

    const { unmount } = renderCamera({ targetLetter: "C" });
    await settleVideo();
    await waitFor(() => expect(predictImage).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /save last frame/i }));
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    const sent = vi.mocked(predictImage).mock.calls[0][0] as Blob;
    expect(createObjectURL.mock.calls[0][0]).toBeInstanceOf(Blob);
    expect(sent).toBeInstanceOf(Blob);
    expect(clicks[0]).toMatch(/^frame_C_\d+\.jpg$/);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:frame");
    clickSpy.mockRestore();
    unmount();
  });
});

describe("CameraPractice: client context sent with each development request", () => {
  it("tells the server the target letter, video size and that the preview is not mirrored", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(noHand()));
    const { unmount } = renderCamera({ targetLetter: "C" });
    await settleVideo();
    await waitFor(() => expect(predictImage).toHaveBeenCalled());
    const options = vi.mocked(predictImage).mock.calls[0][2];
    expect(options).toMatchObject({
      requireHand: true,
      saveDiagnostics: true,
      target: "C",
      videoSize: "640x480",
      previewMirrored: false,
    });
    unmount();
  });

  it("reports a CSS-mirrored preview as mirrored", async () => {
    vi.mocked(predictImage).mockImplementation(respondAfterTick(noHand()));
    const real = window.getComputedStyle.bind(window);
    vi.spyOn(window, "getComputedStyle").mockImplementation((el: Element) => {
      const style = real(el);
      return el instanceof HTMLVideoElement
        ? ({ ...style, transform: "matrix(-1, 0, 0, 1, 0, 0)" } as unknown as CSSStyleDeclaration)
        : style;
    });
    const { unmount } = renderCamera();
    await settleVideo();
    await waitFor(() => expect(predictImage).toHaveBeenCalled());
    expect(vi.mocked(predictImage).mock.calls[0][2]).toMatchObject({ previewMirrored: true });
    unmount();
  });
});
