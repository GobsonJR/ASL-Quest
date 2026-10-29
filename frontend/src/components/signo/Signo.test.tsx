import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MASTERED_CORRECT } from "../../game/constants";
import { createInitialState } from "../../game/storage";
import type { GameState } from "../../game/types";
import { SignoCharacter } from "./SignoCharacter";
import { SignoCompanion } from "./SignoCompanion";
import { SignoProvider, useSigno } from "./SignoContext";
import type { SignoEvent } from "./types";

let mockPage = "learn";
let mockReady = true;
let mockState: GameState = createInitialState();

vi.mock("../../game/GameContext", () => ({
  useGame: () => ({ page: mockPage, ready: mockReady, state: mockState }),
}));

const askSigno = vi.fn();
vi.mock("../../services/signoService", () => ({
  askSigno: (...args: unknown[]) => askSigno(...args),
}));

/** Test-only buttons that report real-looking events through the same emit() screens use. */
function Emitter({ event }: { event: SignoEvent }) {
  const { emit } = useSigno();
  return <button onClick={() => emit(event)}>emit-{event.type}</button>;
}

function renderSigno(events: SignoEvent[] = []) {
  const tree = () => (
    <SignoProvider>
      <SignoCompanion />
      {events.map((e) => (
        <Emitter key={e.type} event={e} />
      ))}
    </SignoProvider>
  );
  const utils = render(tree());
  return { ...utils, rerenderSigno: () => utils.rerender(tree()) };
}

const launcherMood = () => {
  const svgs = document.querySelectorAll("svg.signo");
  return svgs[svgs.length - 1].getAttribute("data-mood");
};
const liveText = () => screen.getByTestId("signo-companion").querySelector("p.sr-only")?.textContent ?? "";
const openLauncher = () => fireEvent.click(screen.getByRole("button", { name: /open signo/i }));

beforeEach(() => {
  mockPage = "learn";
  mockReady = true;
  mockState = createInitialState();
  askSigno.mockReset();
  window.localStorage.clear();
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("SignoCharacter", () => {
  it("renders an accessible default (idle) mascot and only draws accessories it is given", () => {
    const { container, rerender } = render(<SignoCharacter label="SIGNO mascot" />);
    expect(screen.getByRole("img", { name: "SIGNO mascot" })).toHaveAttribute("data-mood", "idle");
    expect(container.querySelector("path[fill='#ffb020']")).toBeNull();
    rerender(<SignoCharacter label="SIGNO mascot" accessories={["crown"]} effects={false} />);
    expect(container.querySelector("path[fill='#ffb020']")).not.toBeNull();
    expect(container.querySelector("svg")).toHaveAttribute("data-effects", "off");
  });

  it("is hidden from assistive tech when decorative", () => {
    const { container } = render(<SignoCharacter mood="happy" />);
    expect(container.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
  });
});

describe("SignoCompanion: launcher and panel", () => {
  it("starts idle until the app is ready, then shows a page welcome", () => {
    mockReady = false;
    renderSigno();
    expect(launcherMood()).toBe("idle");
    expect(screen.getByRole("button", { name: /open signo/i })).toHaveAttribute("aria-expanded", "false");
  });

  it("opens the panel, focuses the input, closes on Escape and returns focus to the launcher", () => {
    renderSigno();
    openLauncher();
    const dialog = screen.getByRole("dialog", { name: "SIGNO" });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByLabelText("Ask SIGNO a question")).toHaveFocus();

    fireEvent.keyDown(screen.getByLabelText("Ask SIGNO a question"), { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("button", { name: /open signo/i })).toHaveFocus();
  });

  it("closes with the Close button", () => {
    renderSigno();
    openLauncher();
    fireEvent.click(screen.getByRole("button", { name: "Close SIGNO" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("is honest about what it cannot do", () => {
    renderSigno();
    openLauncher();
    expect(screen.getByRole("dialog")).toHaveTextContent(/can't see your camera/i);
  });
});

describe("SignoCompanion: contextual content", () => {
  it("welcomes the learner on Learn with their real next letter", () => {
    renderSigno();
    expect(screen.getByTestId("signo-companion")).toHaveTextContent("Up next: letter C");
  });

  it("offers different suggestions per page, using real progress", () => {
    renderSigno();
    openLauncher();
    expect(screen.getByRole("button", { name: /what should i learn next\? i've completed 0 of 26/i })).toBeInTheDocument();
    cleanup();

    mockPage = "achievements";
    renderSigno();
    openLauncher();
    expect(screen.getByRole("button", { name: "Celebrate my progress!" })).toBeInTheDocument();
  });

  it("shows letter-specific prompts after a letter is selected on Alphabet", () => {
    mockPage = "alphabet";
    renderSigno([{ type: "letter_selected", letter: "D" }]);
    fireEvent.click(screen.getByText("emit-letter_selected"));
    openLauncher();
    expect(screen.getByRole("button", { name: /explain how to sign the letter d/i })).toBeInTheDocument();
  });
});

describe("SignoCompanion: reacts to real events", () => {
  it("celebrates a confirmed sign with the real XP and announces it once", () => {
    renderSigno([{ type: "sign_correct", letter: "C", xp: 20 }]);
    fireEvent.click(screen.getByText("emit-sign_correct"));
    expect(launcherMood()).toBe("happy");
    expect(liveText()).toBe("Nice! C confirmed (+20 XP).");
  });

  it("responds gently to an incorrect sign without claiming to know what went wrong", () => {
    renderSigno([{ type: "sign_incorrect", letter: "C", predicted: "S" }]);
    fireEvent.click(screen.getByText("emit-sign_incorrect"));
    expect(launcherMood()).toBe("encouraging");
    const text = screen.getByTestId("signo-companion").textContent ?? "";
    expect(text).toContain("The camera's last stable reading was S");
    expect(text).not.toMatch(/you are showing|your thumb|your fingers/i);
  });

  it("switches to listening when practice starts, without claiming recognition", () => {
    vi.useFakeTimers();
    mockPage = "practice";
    renderSigno([{ type: "practice_started", letter: "C" }]);
    fireEvent.click(screen.getByText("emit-practice_started"));
    // Entering a page makes SIGNO briefly curious; after that it settles into listening.
    expect(launcherMood()).toBe("curious");
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(launcherMood()).toBe("listening");
    expect(screen.getByTestId("signo-companion").textContent).not.toMatch(/recogni[sz]ed|i can see/i);
  });

  it("gives a lesson hint from existing lesson content and offers the hint action", () => {
    mockPage = "practice";
    renderSigno([{ type: "hint_requested", letter: "C" }]);
    fireEvent.click(screen.getByText("emit-hint_requested"));
    expect(launcherMood()).toBe("explaining");
    expect(liveText()).toBe("Hint: Curve your hand like the letter C.");
    openLauncher();
    expect(screen.getByRole("button", { name: "Give me a hint" })).toBeInTheDocument();
    expect(screen.getByText("SIGNO tip")).toBeInTheDocument();
  });

  it("celebrates a completed word with the real XP", () => {
    renderSigno([{ type: "word_complete", word: "HELLO", xp: 75 }]);
    fireEvent.click(screen.getByText("emit-word_complete"));
    expect(launcherMood()).toBe("celebrating");
    expect(liveText()).toBe("Word complete: HELLO (+75 XP)!");
  });

  it("celebrates real progress changes (completed letter, XP) but never on first load", () => {
    mockState = { ...createInitialState(), xp: 500, streak: 3 };
    mockState.letterStats = { ...mockState.letterStats, C: { correct: MASTERED_CORRECT - 1, attempts: MASTERED_CORRECT - 1 } };
    const { rerenderSigno } = renderSigno();
    expect(liveText()).toBe(""); // loading existing progress is not an achievement

    mockState = {
      ...mockState,
      xp: 520,
      letterStats: { ...mockState.letterStats, C: { correct: MASTERED_CORRECT, attempts: MASTERED_CORRECT } },
    };
    act(() => rerenderSigno());
    expect(liveText()).toBe("Letter C complete (+20 XP)!");
    expect(launcherMood()).toBe("celebrating");
  });

  it("reacts to a newly unlocked badge", () => {
    const { rerenderSigno } = renderSigno();
    mockState = { ...mockState, unlockedBadges: ["first_sign"] };
    act(() => rerenderSigno());
    expect(liveText()).toBe("New badge: First Sign!");
  });

  it("does not fire a second celebration when nothing changed", () => {
    const { rerenderSigno } = renderSigno([{ type: "sign_correct", letter: "C", xp: 20 }]);
    fireEvent.click(screen.getByText("emit-sign_correct"));
    const first = liveText();
    act(() => rerenderSigno());
    act(() => rerenderSigno());
    expect(liveText()).toBe(first);
  });
});

describe("SignoCompanion: AI conversation", () => {
  it("sends the question through the existing AURA-backed service and labels the answer", async () => {
    askSigno.mockResolvedValue({ conversationId: 7, text: "Keep your wrist straight.", source: "llm" });
    renderSigno();
    openLauncher();
    fireEvent.change(screen.getByLabelText("Ask SIGNO a question"), { target: { value: "How do I improve my hand positioning?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Keep your wrist straight.")).toBeInTheDocument();
    expect(askSigno).toHaveBeenCalledWith("How do I improve my hand positioning?", null);
    expect(screen.getByText("AURA assistant")).toBeInTheDocument();
  });

  it("shows a thinking state while waiting for the real answer", async () => {
    let resolve!: (v: unknown) => void;
    askSigno.mockReturnValue(new Promise((r) => (resolve = r)));
    renderSigno();
    openLauncher();
    fireEvent.change(screen.getByLabelText("Ask SIGNO a question"), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("SIGNO is thinking…")).toBeInTheDocument();
    expect(launcherMood()).toBe("thinking");
    await act(async () => resolve({ conversationId: 1, text: "Hi!", source: "knowledge_base" }));
    expect(await screen.findByText("ASL Quest knowledge base")).toBeInTheDocument();
  });

  it("reuses the conversation for follow-up questions", async () => {
    askSigno.mockResolvedValue({ conversationId: 7, text: "ok", source: "llm" });
    renderSigno();
    openLauncher();
    const input = screen.getByLabelText("Ask SIGNO a question");
    fireEvent.change(input, { target: { value: "one" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findAllByText("ok");
    fireEvent.change(input, { target: { value: "two" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(askSigno).toHaveBeenCalledTimes(2));
    expect(askSigno).toHaveBeenLastCalledWith("two", 7);
  });

  it("fails gracefully when the backend is unavailable and lets the user retry", async () => {
    askSigno.mockRejectedValueOnce(new Error("I can't reach the ASL Quest server right now."));
    askSigno.mockResolvedValueOnce({ conversationId: 3, text: "Back online.", source: "llm" });
    renderSigno();
    openLauncher();
    fireEvent.change(screen.getByLabelText("Ask SIGNO a question"), { target: { value: "What should I learn next?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("can't reach the ASL Quest server");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Back online.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(askSigno).toHaveBeenCalledTimes(2);
    // The retry re-sends the same question without duplicating the user's message.
    expect(screen.getAllByText("What should I learn next?")).toHaveLength(1);
  });

  it("answers 'Celebrate my progress!' locally from real stats and labels it as a SIGNO tip, not AI", () => {
    mockPage = "achievements";
    mockState = { ...createInitialState(), xp: 120, streak: 2, unlockedBadges: ["first_sign"] };
    renderSigno();
    openLauncher();
    fireEvent.click(screen.getByRole("button", { name: "Celebrate my progress!" }));
    expect(askSigno).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog")).toHaveTextContent("120 XP, 2-day streak, 1 badge earned");
    expect(screen.getByText("SIGNO tip")).toBeInTheDocument();
  });
});

describe("SignoCompanion: preferences, motion and inactivity", () => {
  it("lets the user turn animations off, persisting only the UI preference", () => {
    renderSigno([{ type: "sign_correct", letter: "C", xp: 20 }]);
    openLauncher();
    fireEvent.click(screen.getByRole("button", { name: "Animations" }));
    expect(screen.getByRole("button", { name: "Animations" })).toHaveAttribute("aria-pressed", "false");
    expect(document.querySelector("svg.signo")).toHaveAttribute("data-effects", "off");

    const stored = JSON.parse(window.localStorage.getItem("signo-ui-v1") ?? "{}");
    expect(stored).toEqual({ minimized: false, effects: false });
    expect(window.localStorage.length).toBe(1); // nothing else (no chat, tokens or progress)
  });

  it("defaults animations off when the OS asks for reduced motion", () => {
    const original = window.matchMedia;
    window.matchMedia = ((query: string) => ({ matches: /reduce/.test(query), media: query, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
    try {
      renderSigno();
      expect(document.querySelector("svg.signo")).toHaveAttribute("data-effects", "off");
    } finally {
      window.matchMedia = original;
    }
  });

  it("minimize closes the panel and stops volunteering tips, but still shows a hint the user asked for", () => {
    renderSigno([{ type: "hint_requested", letter: "C" }]);
    openLauncher();
    fireEvent.click(screen.getByRole("button", { name: "Minimize SIGNO" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(JSON.parse(window.localStorage.getItem("signo-ui-v1") ?? "{}").minimized).toBe(true);

    fireEvent.click(screen.getByText("emit-hint_requested"));
    expect(screen.getByTestId("signo-companion")).toHaveTextContent("Hint: Curve your hand like the letter C.");
  });

  it("gets sleepy after inactivity and welcomes the learner back with their real streak", () => {
    vi.useFakeTimers();
    mockState = { ...createInitialState(), streak: 4 };
    renderSigno();
    act(() => {
      vi.advanceTimersByTime(91_000);
    });
    expect(launcherMood()).toBe("sleepy");

    fireEvent.pointerDown(window);
    expect(screen.getByTestId("signo-companion")).toHaveTextContent("Welcome back! You're on a 4-day streak.");
    expect(launcherMood()).not.toBe("sleepy");
  });
});
