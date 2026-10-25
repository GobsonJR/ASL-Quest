import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";
import { useRef } from "react";
import { EMBED_READY_TIMEOUT_MS, useVimeoReady, type EmbedState } from "./useVimeoReady";

// A controllable stand-in for Vimeo's Player SDK: each test decides when (and whether) ready() settles.
const { players, FakePlayer } = vi.hoisted(() => {
  const players: Array<InstanceType<typeof FakePlayer>> = [];
  class FakePlayer {
    element: HTMLIFrameElement;
    resolveReady!: () => void;
    rejectReady!: (error: Error) => void;
    handlers = new Map<string, Set<() => void>>();
    off = vi.fn((event: string, cb: () => void) => void this.handlers.get(event)?.delete(cb));
    destroy = vi.fn().mockResolvedValue(undefined);
    private readyPromise = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    constructor(element: HTMLIFrameElement) {
      this.element = element;
      players.push(this);
    }
    on(event: string, cb: () => void) {
      if (!this.handlers.has(event)) this.handlers.set(event, new Set());
      this.handlers.get(event)!.add(cb);
    }
    ready() {
      return this.readyPromise;
    }
    emit(event: string) {
      this.handlers.get(event)?.forEach((cb) => cb());
    }
  }
  return { players, FakePlayer };
});
vi.mock("@vimeo/player", () => ({ default: FakePlayer }));

let seen: EmbedState;
function Probe({ url }: { url: string | null }) {
  const ref = useRef<HTMLIFrameElement>(null);
  seen = useVimeoReady(ref, url);
  return url ? <iframe key={url} ref={ref} src={url} title="ref" /> : null;
}

const A = "https://player.vimeo.com/video/1?h=aaaa";
const B = "https://player.vimeo.com/video/2?h=bbbb";

beforeEach(() => {
  players.length = 0;
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

const flush = () => act(async () => void (await vi.advanceTimersByTimeAsync(0)));

describe("useVimeoReady", () => {
  it("reports loaded once the player is ready, and never shows the failure afterwards", async () => {
    render(<Probe url={A} />);
    expect(seen).toBe("loading");
    players[0].resolveReady();
    await flush();
    expect(seen).toBe("loaded");
    await act(async () => void (await vi.advanceTimersByTimeAsync(EMBED_READY_TIMEOUT_MS * 2)));
    expect(seen).toBe("loaded");
  });

  it("does not call a slow player broken, falls back only after the full timeout, and recovers if it turns up late", async () => {
    render(<Probe url={A} />);
    await act(async () => void (await vi.advanceTimersByTimeAsync(EMBED_READY_TIMEOUT_MS - 1000)));
    expect(seen).toBe("loading"); // 14 s of slowness is not a failure
    await act(async () => void (await vi.advanceTimersByTimeAsync(1500)));
    expect(seen).toBe("error"); // never became ready: offer the fallback
    players[0].resolveReady(); // ... but a late player clears it again
    await flush();
    expect(seen).toBe("loaded");
  });

  it("reports an error as soon as the player rejects ready() (video missing / private)", async () => {
    render(<Probe url={A} />);
    players[0].rejectReady(new Error("NotFoundError: This video does not exist."));
    await flush();
    expect(seen).toBe("error");
  });

  it("reports an error when the player emits an error event", async () => {
    render(<Probe url={A} />);
    act(() => players[0].emit("error"));
    expect(seen).toBe("error");
  });

  it("reports an error if the SDK cannot attach to the iframe", async () => {
    const original = players.length;
    vi.spyOn(FakePlayer.prototype, "ready").mockImplementation(() => {
      throw new Error("The player element passed isn't a Vimeo embed.");
    });
    render(<Probe url={A} />);
    expect(players.length).toBe(original + 1);
    expect(seen).toBe("error");
    vi.restoreAllMocks();
  });

  it("ignores a previous sign's player when the sign changes", async () => {
    const { rerender } = render(<Probe url={A} />);
    players[0].resolveReady();
    await flush();
    expect(seen).toBe("loaded");

    rerender(<Probe url={B} />);
    expect(seen).toBe("loading"); // A's "loaded" must not carry over to B
    expect(players).toHaveLength(2);

    // A's player reports something late: B must be unaffected.
    players[0].rejectReady(new Error("late"));
    act(() => players[0].emit("error"));
    await flush();
    expect(seen).toBe("loading");

    players[1].resolveReady();
    await flush();
    expect(seen).toBe("loaded");
  });

  it("releases listeners and timers on unmount, and a late answer after unmount does nothing", async () => {
    const clear = vi.spyOn(window, "clearTimeout");
    const { unmount } = render(<Probe url={A} />);
    const player = players[0];
    expect(player.handlers.get("error")?.size).toBe(1);
    unmount();
    expect(player.off).toHaveBeenCalledWith("error", expect.any(Function));
    expect(player.handlers.get("error")?.size).toBe(0);
    expect(clear).toHaveBeenCalled();
    expect(player.destroy).toHaveBeenCalled(); // the iframe was already detached, so it is safe to destroy
    expect(vi.getTimerCount()).toBe(0);
    player.resolveReady(); // late answer: no state update, no error
    await flush();
  });
});
