import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { NativeSignPracticePage } from "./NativeSignPracticePage";
import type { NativePracticeResult, NativeSign, NativeSignProgressEntry } from "./api";

const navigate = vi.fn();
const startNativeSignPractice = vi.fn();
const applyNativeSignResult = vi.fn();
const push = vi.fn();

const { getNativeSigns, getNativeSignProgress, practiceNativeSign } = vi.hoisted(() => ({
  getNativeSigns: vi.fn(),
  getNativeSignProgress: vi.fn(),
  practiceNativeSign: vi.fn(),
}));

vi.mock("../game/GameContext", () => ({
  useGame: () => ({
    practiceNativeSignId: 1,
    navigate,
    startNativeSignPractice,
    applyNativeSignResult,
  }),
}));

vi.mock("../components/Toast", () => ({
  useToast: () => ({ push }),
}));

vi.mock("./api", () => ({
  getNativeSigns,
  getNativeSignProgress,
  practiceNativeSign,
}));

// The real capture/MediaRecorder lifecycle is covered by
// NativeVideoCapture.test.tsx -- here we only need a stand-in that lets the
// test drive onClipReady and observe how NativeSignPracticePage reacts.
vi.mock("./NativeVideoCapture", () => ({
  NativeVideoCapture: (props: { onClipReady: (clip: Blob, mimeType: string) => void }) => (
    <button onClick={() => props.onClipReady(new Blob(["clip"], { type: "video/webm" }), "video/webm")}>
      fire-clip-ready
    </button>
  ),
}));

function sign(overrides: Partial<NativeSign> = {}): NativeSign {
  return {
    id: 1,
    gloss: "HELLO",
    display_name: "Hello",
    meaning: "A greeting.",
    category: "Greetings",
    difficulty: "Easy",
    description: "The sign for hello.",
    example_text: "Hello there.",
    dataset_available: true,
    model_available: true,
    active: true,
    reference: { available: false, video_url: null, thumbnail_url: null, source_type: null, license_note: null },
    ...overrides,
  };
}

function progress(overrides: Partial<NativeSignProgressEntry> = {}): NativeSignProgressEntry {
  return {
    sign_id: 1,
    gloss: "HELLO",
    display_name: "Hello",
    attempts: 2,
    correct: 1,
    mastery: 50,
    last_practiced: null,
    ...overrides,
  };
}

function result(overrides: Partial<NativePracticeResult> = {}): NativePracticeResult {
  return {
    prediction: "HELLO",
    confidence: 0.92,
    top_k: [{ gloss: "HELLO", confidence: 0.92 }],
    model_type: "i3d",
    model_version: "1",
    latency_ms: 120,
    native_sign_id: 1,
    correct: true,
    expected_sign: { id: 1, gloss: "HELLO", display_name: "Hello" },
    session_id: 1,
    response_time: 3000,
    xp_earned: 20,
    new_achievements: [],
    ...overrides,
  };
}

beforeEach(() => {
  navigate.mockReset();
  startNativeSignPractice.mockReset();
  applyNativeSignResult.mockReset();
  push.mockReset();
  getNativeSigns.mockReset().mockResolvedValue({ items: [sign()] });
  getNativeSignProgress.mockReset().mockResolvedValue(progress());
  practiceNativeSign.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("NativeSignPracticePage: capture to result flow", () => {
  it("shows the prediction and awards XP once a clip is captured", async () => {
    practiceNativeSign.mockResolvedValue(result());
    render(<NativeSignPracticePage />);

    const clipButton = await screen.findByText("fire-clip-ready");
    fireEvent.click(clipButton);

    expect(await screen.findByText("Correct!")).toBeInTheDocument();
    expect(screen.getAllByText("HELLO").length).toBeGreaterThan(0);
    expect(applyNativeSignResult).toHaveBeenCalledWith(true, 20, []);
    expect(push).toHaveBeenCalledWith("+20 XP", "success");
  });
});

describe("NativeSignPracticePage: prediction failure", () => {
  it("surfaces the backend's error message and returns to the capture phase, not stuck processing", async () => {
    practiceNativeSign.mockRejectedValue(new Error("Could not read or process the uploaded video."));
    render(<NativeSignPracticePage />);

    const clipButton = await screen.findByText("fire-clip-ready");
    fireEvent.click(clipButton);

    await waitFor(() =>
      expect(screen.getByText("Could not read or process the uploaded video.")).toBeInTheDocument()
    );
    expect(push).toHaveBeenCalledWith("Could not read or process the uploaded video.", "error");
    // Back in the capture phase, not hung on "Analyzing your sign...".
    expect(screen.getByText("fire-clip-ready")).toBeInTheDocument();
    expect(applyNativeSignResult).not.toHaveBeenCalled();
  });
});
