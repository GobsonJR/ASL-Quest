import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { NativeVideoCapture } from "./NativeVideoCapture";

// jsdom has no real MediaRecorder -- this fake reproduces just enough of its
// surface (start/stop, ondataavailable/onstop/onerror callbacks, the static
// isTypeSupported guard NativeVideoCapture checks up front) for the
// component's real lifecycle wiring to run end to end.
class FakeMediaRecorder {
  static isTypeSupported = vi.fn().mockReturnValue(true);
  state: "inactive" | "recording" = "inactive";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;

  constructor(
    public stream: MediaStream,
    public options: { mimeType: string }
  ) {
    instances.push(this);
  }

  start() {
    this.state = "recording";
  }

  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["frame"], { type: "video/webm" }) });
    this.onstop?.();
  }
}

let instances: FakeMediaRecorder[] = [];

function fakeStream(): MediaStream {
  return { getTracks: () => [{ stop: vi.fn() }] } as unknown as MediaStream;
}

beforeEach(() => {
  instances = [];
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue(fakeStream()) },
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("NativeVideoCapture: capture flow", () => {
  it("produces a non-empty clip via onClipReady once recording is stopped", async () => {
    const onClipReady = vi.fn();
    render(<NativeVideoCapture onClipReady={onClipReady} />);

    fireEvent.click(screen.getByRole("button", { name: /start recording/i }));
    await waitFor(() => expect(screen.getByRole("button", { name: /stop now/i })).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /stop now/i }));

    await waitFor(() => expect(onClipReady).toHaveBeenCalledTimes(1));
    const [clip, mimeType] = onClipReady.mock.calls[0];
    expect(clip).toBeInstanceOf(Blob);
    expect(clip.size).toBeGreaterThan(0);
    expect(typeof mimeType).toBe("string");
  });
});

describe("NativeVideoCapture: camera error specificity", () => {
  it("shows a permission-denied message for NotAllowedError", async () => {
    vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValueOnce(
      new DOMException("denied", "NotAllowedError")
    );
    render(<NativeVideoCapture onClipReady={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /start recording/i }));

    expect(await screen.findByText(/denied/i)).toBeInTheDocument();
  });

  it("shows a no-camera-found message for NotFoundError", async () => {
    vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValueOnce(
      new DOMException("none", "NotFoundError")
    );
    render(<NativeVideoCapture onClipReady={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /start recording/i }));

    expect(await screen.findByText(/no camera/i)).toBeInTheDocument();
  });
});

describe("NativeVideoCapture: recorder failure mid-recording", () => {
  it("resets to idle and shows an error instead of hanging in 'Recording...'", async () => {
    render(<NativeVideoCapture onClipReady={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /start recording/i }));
    await waitFor(() => expect(instances.length).toBe(1));
    await waitFor(() => expect(screen.getByRole("button", { name: /stop now/i })).toBeInTheDocument());

    instances[0].onerror?.({ error: new DOMException("in use", "NotReadableError") });

    await waitFor(() => expect(screen.getByRole("button", { name: /start recording/i })).toBeInTheDocument());
    expect(screen.getByText(/in use/i)).toBeInTheDocument();
  });
});
