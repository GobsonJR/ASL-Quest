import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import { WordPracticePage } from "./WordPracticePage";
import type { WordSessionResult } from "../words/api";

const startWordPractice = vi.fn();
const navigate = vi.fn();
const beginSession = vi.fn();
const applyWordXp = vi.fn();
const markWordChallengeComplete = vi.fn();
const push = vi.fn();

// "../words/api" is imported (and its mock factory evaluated) at module-load
// time, before any of this file's own top-level `const`s have run -- unlike
// the useGame()/useToast() mocks below, whose factories only return closures
// that aren't *called* until render time. vi.hoisted() gives a binding that's
// safe to reference directly inside a vi.mock factory.
const { recordWordSession } = vi.hoisted(() => ({
  recordWordSession: vi.fn(),
}));

vi.mock("../game/GameContext", () => ({
  useGame: () => ({
    practiceWordId: "beginner-cat",
    wordQueue: [],
    challengeMode: "none",
    startWordPractice,
    navigate,
    beginSession,
    applyWordXp,
    markWordChallengeComplete,
  }),
}));

vi.mock("../components/Toast", () => ({
  useToast: () => ({ push }),
}));

vi.mock("../words/api", () => ({
  recordWordSession,
}));

// House style from PracticePage.test.tsx: the real camera/ML loop is covered
// by CameraPractice.test.tsx -- here we only need a stand-in that lets the
// test drive onCorrect/onIncorrect and observe what WordPracticePage does.
vi.mock("../components/CameraPractice", () => ({
  CameraPractice: (props: {
    targetLetter: string;
    onCorrect: (elapsedMs: number, confidence: number) => void;
    onIncorrect: (predicted: string) => void;
  }) => (
    <div data-testid="camera-practice">
      <span>target:{props.targetLetter}</span>
      <button onClick={() => props.onCorrect(1200, 0.9)}>fire-correct</button>
      <button onClick={() => props.onIncorrect("X")}>fire-incorrect</button>
    </div>
  ),
}));

function sessionResult(overrides: Partial<WordSessionResult> = {}): WordSessionResult {
  return {
    id: 1,
    word_id: "beginner-cat",
    word: "CAT",
    completed: true,
    accuracy: 100,
    duration_ms: 3000,
    xp_earned: 110,
    xp_events: [],
    new_achievements: [],
    progress: {} as WordSessionResult["progress"],
    ...overrides,
  };
}

async function spellCat() {
  render(<WordPracticePage ready backendError={null} />);
  expect(screen.getByText("target:C")).toBeInTheDocument();
  fireEvent.click(screen.getByText("fire-correct"));
  await waitFor(() => expect(screen.getByText("target:A")).toBeInTheDocument());
  fireEvent.click(screen.getByText("fire-correct"));
  await waitFor(() => expect(screen.getByText("target:T")).toBeInTheDocument());
  fireEvent.click(screen.getByText("fire-correct"));
}

beforeEach(() => {
  startWordPractice.mockReset();
  navigate.mockReset();
  beginSession.mockReset();
  applyWordXp.mockReset();
  markWordChallengeComplete.mockReset();
  push.mockReset();
  recordWordSession.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("WordPracticePage: spelling flow", () => {
  it("spells C-A-T letter by letter and completes the word", async () => {
    recordWordSession.mockResolvedValue(sessionResult());
    await spellCat();

    await waitFor(() => expect(recordWordSession).toHaveBeenCalledTimes(1));
    const call = recordWordSession.mock.calls[0][0] as {
      word_id: string;
      letters: Array<{ letter: string; correct: boolean }>;
    };
    expect(call.word_id).toBe("beginner-cat");
    expect(call.letters.map((l) => l.letter)).toEqual(["C", "A", "T"]);
    expect(call.letters.every((l) => l.correct)).toBe(true);

    expect(await screen.findByText("CAT completed!")).toBeInTheDocument();
  });

  it("applies XP exactly once for the whole word, not per letter", async () => {
    recordWordSession.mockResolvedValue(sessionResult({ xp_earned: 110 }));
    await spellCat();

    await waitFor(() => expect(applyWordXp).toHaveBeenCalledTimes(1));
    expect(applyWordXp).toHaveBeenCalledWith(110, []);
  });
});

describe("WordPracticePage: sync failure toasts", () => {
  it("shows the offline-save toast for a genuine network failure", async () => {
    recordWordSession.mockRejectedValue(new TypeError("Failed to fetch"));
    await spellCat();

    await waitFor(() =>
      expect(push).toHaveBeenCalledWith(
        "Saved locally — we'll sync word progress when you're back online.",
        "warning"
      )
    );
    expect(applyWordXp).not.toHaveBeenCalled();
  });

  it("shows the backend's specific rejection message, not the generic offline toast", async () => {
    recordWordSession.mockRejectedValue(new Error("Unknown word."));
    await spellCat();

    await waitFor(() => expect(push).toHaveBeenCalledWith("Unknown word.", "error"));
    expect(push).not.toHaveBeenCalledWith(
      "Saved locally — we'll sync word progress when you're back online.",
      "warning"
    );
  });
});
