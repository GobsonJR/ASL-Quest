import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, act } from "@testing-library/react";
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
  NativeVideoCapture: (props: { disabled?: boolean; onClipReady: (clip: Blob, mimeType: string) => void }) => (
    <button
      data-testid="mock-capture"
      data-disabled={String(Boolean(props.disabled))}
      onClick={() => props.onClipReady(new Blob(["clip"], { type: "video/webm" }), "video/webm")}
    >
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

// Vimeo's Player SDK is replaced by a stand-in whose ready() each test settles itself (the real one needs a real iframe).
const { vimeoPlayers, FakeVimeoPlayer } = vi.hoisted(() => {
  const vimeoPlayers: Array<InstanceType<typeof FakeVimeoPlayer>> = [];
  class FakeVimeoPlayer {
    resolveReady!: () => void;
    rejectReady!: (error: Error) => void;
    private readyPromise = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    constructor(public element: HTMLIFrameElement) {
      vimeoPlayers.push(this);
    }
    on() {}
    off() {}
    ready() {
      return this.readyPromise;
    }
    destroy() {
      return Promise.resolve();
    }
  }
  return { vimeoPlayers, FakeVimeoPlayer };
});
vi.mock("@vimeo/player", () => ({ default: FakeVimeoPlayer }));

beforeEach(() => {
  vimeoPlayers.length = 0;
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
    // The ranked list is no longer shown, but confidence, expected/predicted sign and response time are.
    expect(screen.queryByText(/top predictions/i)).not.toBeInTheDocument();
    expect(screen.getByText("92.0% confidence")).toBeInTheDocument();
    expect(screen.getByText("Expected")).toBeInTheDocument();
    expect(screen.getByText("Predicted")).toBeInTheDocument();
    expect(screen.getByText("3000ms")).toBeInTheDocument();
  });

  it("shows 'Not quite' with no success message, XP or XP toast for an incorrect attempt (HELLO expected, PLEASE predicted)", async () => {
    practiceNativeSign.mockResolvedValue(
      result({
        prediction: "PLEASE",
        confidence: 0.199,
        correct: false,
        xp_earned: 0,
        top_k: [
          { gloss: "PLEASE", confidence: 0.199 },
          { gloss: "HELLO", confidence: 0.11 },
          { gloss: "EAT1", confidence: 0.1 },
        ],
      })
    );
    render(<NativeSignPracticePage />);

    fireEvent.click(await screen.findByText("fire-clip-ready"));

    expect(await screen.findByText("Not quite")).toBeInTheDocument();
    expect(screen.queryByText("Correct!")).not.toBeInTheDocument();
    expect(screen.queryByText("Great job!")).not.toBeInTheDocument();
    expect(screen.queryByText(/\+\d+ XP/)).not.toBeInTheDocument();
    expect(screen.getByText(/no correctness xp/i)).toBeInTheDocument();
    expect(push).not.toHaveBeenCalledWith(expect.stringContaining("XP"), "success");
    expect(applyNativeSignResult).toHaveBeenCalledWith(false, 0, []);
    // Expected and predicted stay visible and differ; the ranked list is gone, so HELLO at 11% is not advertised.
    expect(screen.getByText("19.9% confidence")).toBeInTheDocument();
    expect(screen.queryByText(/top predictions/i)).not.toBeInTheDocument();
    expect(screen.queryByText("11%")).not.toBeInTheDocument();
  });

  it("treats a missing correctness flag as incorrect", async () => {
    practiceNativeSign.mockResolvedValue(result({ correct: null, xp_earned: 0 }));
    render(<NativeSignPracticePage />);
    fireEvent.click(await screen.findByText("fire-clip-ready"));
    expect(await screen.findByText("Not quite")).toBeInTheDocument();
    expect(screen.queryByText("Correct!")).not.toBeInTheDocument();
    expect(applyNativeSignResult).toHaveBeenCalledWith(false, 0, []);
  });
});

describe("NativeSignPracticePage: reference panel", () => {
  it("shows an available external reference with a tip and an Open source link, not an inline video tag", async () => {
    getNativeSigns.mockResolvedValue({
      items: [
        sign({
          reference: {
            available: true,
            video_url: "https://learnhowtosign.com/dictionary/hello/",
            thumbnail_url: null,
            source_type: "external_url",
            license_note: "Reference: \"Hello\" by Learn How to Sign (learnhowtosign.com).",
          },
        }),
      ],
    });
    render(<NativeSignPracticePage />);

    const link = await screen.findByRole("link", { name: /open source/i });
    expect(link).toHaveAttribute("href", "https://learnhowtosign.com/dictionary/hello/");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
    expect(screen.getByText(/watch carefully/i)).toBeInTheDocument();
    expect(screen.getByText(/learn how to sign/i)).toBeInTheDocument();
    // Never a bare <video src=".../hello/"> -- that URL is an HTML page, not a media file.
    expect(document.querySelector("video")).not.toBeInTheDocument();
  });

  it("gates recording behind watching the reference, and unlocks it after 'I'm ready'", async () => {
    render(<NativeSignPracticePage />);

    const capture = await screen.findByTestId("mock-capture");
    expect(capture).toHaveAttribute("data-disabled", "true");

    fireEvent.click(screen.getByRole("button", { name: /i.m ready/i }));

    expect(capture).toHaveAttribute("data-disabled", "false");
  });

  it("shows the positive demo-on-its-way placeholder with the sign's meaning when no reference is available", async () => {
    render(<NativeSignPracticePage />);

    expect(await screen.findByText(/demo video on its way/i)).toBeInTheDocument();
    expect(screen.getByText(/use the meaning and description above/i)).toBeInTheDocument();
    expect(screen.queryByText(/coming soon|no verified/i)).not.toBeInTheDocument();
  });

  it("EAT1 (no reference) still shows the demo-on-its-way placeholder", async () => {
    getNativeSigns.mockResolvedValue({
      items: [
        sign({
          gloss: "EAT1",
          display_name: "Eat",
          reference: { available: false, video_url: null, thumbnail_url: null, source_type: null, license_note: null },
        }),
      ],
    });
    render(<NativeSignPracticePage />);

    expect(await screen.findByText(/demo video on its way/i)).toBeInTheDocument();
    expect(screen.getAllByText("Eat").length).toBeGreaterThan(0);
  });
});

describe("NativeSignPracticePage: embedded Vimeo reference", () => {
  function vimeoSign(overrides: Partial<NativeSign> = {}) {
    return sign({
      gloss: "BOOK",
      display_name: "Book",
      reference: {
        available: true,
        video_url: "https://vimeo.com/897045923/ca27cc3775",
        thumbnail_url: null,
        source_type: "vimeo",
        license_note: 'Reference: "Book" by Learn How to Sign (learnhowtosign.com) -- played here via Vimeo.',
      },
      ...overrides,
    });
  }

  it("renders an inline iframe embed for a vimeo source, preserving the unlisted privacy hash", async () => {
    getNativeSigns.mockResolvedValue({ items: [vimeoSign()] });
    render(<NativeSignPracticePage />);

    const iframe = await screen.findByTitle(/reference demonstration of book/i);
    expect(iframe.tagName).toBe("IFRAME");
    expect(iframe).toHaveAttribute("src", "https://player.vimeo.com/video/897045923?h=ca27cc3775");
    // Never the bare/hash-stripped form -- an unlisted Vimeo video 404s without its hash.
    expect(iframe.getAttribute("src")).not.toBe("https://player.vimeo.com/video/897045923");
  });

  it("plays the locally stored video (no Vimeo, no internet needed) when the server has one", async () => {
    const withLocal = vimeoSign();
    withLocal.reference = { ...withLocal.reference, local_video_url: "/native/reference-video/BOOK" };
    getNativeSigns.mockResolvedValue({ items: [withLocal] });
    const { container } = render(<NativeSignPracticePage />);

    const video = await waitFor(() => {
      const el = container.querySelector("video[src]");
      expect(el).not.toBeNull();
      return el as HTMLVideoElement;
    });
    expect(video.getAttribute("src")).toMatch(/\/native\/reference-video\/BOOK$/);
    expect(video).toHaveAttribute("controls");
    expect(container.querySelector("iframe")).toBeNull(); // the online embed is not even created
    expect(vimeoPlayers).toHaveLength(0); // so Vimeo's SDK is never started
    expect(screen.queryByText(/couldn.t load|loading reference/i)).not.toBeInTheDocument();
  });

  it("falls back to the Vimeo embed if the local video cannot be played", async () => {
    const withLocal = vimeoSign();
    withLocal.reference = { ...withLocal.reference, local_video_url: "/native/reference-video/BOOK" };
    getNativeSigns.mockResolvedValue({ items: [withLocal] });
    const { container } = render(<NativeSignPracticePage />);

    const video = await waitFor(() => {
      const el = container.querySelector("video[src]");
      expect(el).not.toBeNull();
      return el as HTMLVideoElement;
    });
    act(() => {
      video.dispatchEvent(new Event("error"));
    });
    expect(await screen.findByTitle(/reference demonstration of book/i)).toHaveAttribute(
      "src",
      "https://player.vimeo.com/video/897045923?h=ca27cc3775"
    );
  });

  it("shows the local video even for a sign that has no online reference at all", async () => {
    const localOnly = sign({
      gloss: "EAT1",
      display_name: "Eat",
      reference: {
        available: true,
        video_url: null,
        local_video_url: "/native/reference-video/EAT1",
        thumbnail_url: null,
        source_type: null,
        license_note: null,
      },
    });
    getNativeSigns.mockResolvedValue({ items: [localOnly] });
    const { container } = render(<NativeSignPracticePage />);
    await waitFor(() => expect(container.querySelector("video[src]")).not.toBeNull());
    expect(screen.queryByText(/demo video on its way/i)).not.toBeInTheDocument();
  });

  it("shows the fallback immediately when Vimeo reports the video unavailable, even though the iframe itself loaded", async () => {
    // An iframe's "load" event fires for Vimeo's "this video does not exist" page too; the player SDK's ready() rejection is
    // the real signal.
    getNativeSigns.mockResolvedValue({ items: [vimeoSign()] });
    render(<NativeSignPracticePage />);

    const iframe = await screen.findByTitle(/reference demonstration of book/i);
    act(() => {
      iframe.dispatchEvent(new Event("load"));
    });
    expect(screen.queryByText(/couldn.t load/i)).not.toBeInTheDocument();
    expect(screen.getByText(/loading reference/i)).toBeInTheDocument(); // "load" alone proves nothing
    await waitFor(() => expect(vimeoPlayers).toHaveLength(1));

    await act(async () => {
      vimeoPlayers[0].rejectReady(new Error("NotFoundError: This video does not exist."));
    });
    expect(screen.getByText(/couldn.t load/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /open source instead/i })).toHaveAttribute("href", "https://vimeo.com/897045923/ca27cc3775");
    expect(screen.getByRole("button", { name: /i.m ready/i })).toBeInTheDocument();
  });

  it("shows a graceful fallback if the embed never becomes ready, without breaking the rest of the page", async () => {
    vi.useFakeTimers();
    try {
      getNativeSigns.mockResolvedValue({ items: [vimeoSign()] });
      render(<NativeSignPracticePage />);

      await vi.waitFor(() => expect(screen.getByTitle(/reference demonstration of book/i)).toBeInTheDocument());

      await act(async () => {
        await vi.advanceTimersByTimeAsync(14000);
      });
      expect(screen.queryByText(/couldn.t load/i)).not.toBeInTheDocument(); // slow is not broken
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1500);
      });

      expect(screen.getByText(/couldn.t load/i)).toBeInTheDocument();
      const fallbackLink = screen.getByRole("link", { name: /open source instead/i });
      expect(fallbackLink).toHaveAttribute("href", "https://vimeo.com/897045923/ca27cc3775");
      expect(fallbackLink).toHaveAttribute("rel", expect.stringContaining("noopener"));
      // The rest of native practice (the gating CTA) still works -- an embed failure
      // never breaks the practice flow.
      expect(screen.getByRole("button", { name: /i.m ready/i })).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not show the failure fallback once the player reports ready in time", async () => {
    vi.useFakeTimers();
    try {
      getNativeSigns.mockResolvedValue({ items: [vimeoSign()] });
      render(<NativeSignPracticePage />);

      await vi.waitFor(() => screen.getByTitle(/reference demonstration of book/i));
      await vi.waitFor(() => expect(vimeoPlayers).toHaveLength(1));
      await act(async () => {
        vimeoPlayers[0].resolveReady();
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(30000);
      });

      expect(screen.queryByText(/couldn.t load/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/loading reference/i)).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
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
