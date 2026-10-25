import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { useRef } from "react";
import { useHandOverlay } from "./useHandOverlay";
import { createVideoHandLandmarker } from "./handLandmarker";

vi.mock("@mediapipe/tasks-vision", () => ({ HandLandmarker: { HAND_CONNECTIONS: [{ start: 0, end: 1 }] } }));
vi.mock("./handLandmarker", () => ({ createVideoHandLandmarker: vi.fn() }));

const HAND = Array.from({ length: 21 }, (_, i) => ({ x: 0.2 + i * 0.01, y: 0.3, z: 0 }));
let seen: { status: string; hands: number };

function Probe({ enabled }: { enabled: boolean }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  seen = useHandOverlay(video, canvas, enabled);
  return (
    <>
      <video ref={video} />
      <canvas ref={canvas} />
    </>
  );
}

beforeEach(() => {
  let t = 0;
  Object.defineProperty(HTMLMediaElement.prototype, "currentTime", { configurable: true, get: () => (t += 0.03) });
  Object.defineProperty(HTMLMediaElement.prototype, "readyState", { configurable: true, get: () => 4 });
  Object.defineProperty(HTMLVideoElement.prototype, "videoWidth", { configurable: true, get: () => 640 });
  Object.defineProperty(HTMLVideoElement.prototype, "videoHeight", { configurable: true, get: () => 480 });
  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
    clearRect: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(), arc: vi.fn(), fill: vi.fn(),
  }) as unknown as HTMLCanvasElement["getContext"];
});

afterEach(() => vi.restoreAllMocks());

describe("useHandOverlay", () => {
  it("is off until enabled, then reports how many hands are detected", async () => {
    const close = vi.fn();
    vi.mocked(createVideoHandLandmarker).mockResolvedValue({
      detectForVideo: () => ({ landmarks: [HAND, HAND] }),
      close,
    } as never);
    const { rerender, unmount } = render(<Probe enabled={false} />);
    expect(seen.status).toBe("off");
    expect(createVideoHandLandmarker).not.toHaveBeenCalled();

    rerender(<Probe enabled />);
    await waitFor(() => expect(seen.hands).toBe(2), { timeout: 3000 });
    expect(seen.status).toBe("ready");
    expect(createVideoHandLandmarker).toHaveBeenCalledWith(2);

    unmount();
    expect(close).toHaveBeenCalled();
  });

  it("degrades to 'unavailable' instead of throwing when MediaPipe cannot load", async () => {
    vi.mocked(createVideoHandLandmarker).mockRejectedValue(new Error("offline"));
    render(<Probe enabled />);
    await waitFor(() => expect(seen.status).toBe("unavailable"));
    expect(seen.hands).toBe(0);
  });
});
