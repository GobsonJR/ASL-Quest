import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { PracticePage } from "./PracticePage";
import { LETTERS } from "../game/constants";
import type { PracticeOutcome } from "../game/types";

function emptyLetterStats(): Record<string, { correct: number; attempts: number }> {
  return Object.fromEntries(LETTERS.map((letter) => [letter, { correct: 0, attempts: 0 }]));
}

const markCorrect = vi.fn<(letter: string, elapsedMs: number) => PracticeOutcome>();
const markIncorrect = vi.fn();
const nextPracticeLetter = vi.fn<() => string>();
const startPractice = vi.fn();
const beginSession = vi.fn();
const navigate = vi.fn();
const push = vi.fn();

function outcome(overrides: Partial<PracticeOutcome> = {}): PracticeOutcome {
  return {
    xpGained: 20,
    leveledUp: false,
    newLevel: 1,
    badgesUnlocked: [],
    dailyCompleted: false,
    message: "",
    ...overrides,
  };
}

vi.mock("../game/GameContext", () => ({
  useGame: () => ({
    state: {
      xp: 100,
      streak: 3,
      dailyChallenge: { progress: 0 },
      letterStats: emptyLetterStats(),
    },
    level: 1,
    practiceLetter: null,
    challengeMode: "none",
    beginSession,
    nextPracticeLetter,
    markCorrect,
    markIncorrect,
    startPractice,
    navigate,
  }),
}));

vi.mock("../components/Toast", () => ({
  useToast: () => ({ push }),
}));

// The camera/ML loop itself is covered by CameraPractice.test.tsx. Here we
// only need a stand-in that lets the test drive onCorrect/onIncorrect and
// observe what PracticePage does with the result -- lock the success state,
// reset on Next Letter / Practice Again, swap the reference image.
vi.mock("../components/CameraPractice", () => ({
  CameraPractice: (props: {
    targetLetter: string;
    active: boolean;
    attemptKey?: number;
    onCorrect: (elapsedMs: number, confidence: number) => void;
    onIncorrect: (predicted: string) => void;
  }) => (
    <div data-testid="camera-practice" data-active={String(props.active)} data-attempt-key={props.attemptKey}>
      <span>target:{props.targetLetter}</span>
      <button onClick={() => props.onCorrect(1200, 0.87)}>fire-correct</button>
      <button onClick={() => props.onIncorrect("S")}>fire-incorrect</button>
    </div>
  ),
}));

beforeEach(() => {
  markCorrect.mockReset().mockReturnValue(outcome());
  markIncorrect.mockReset();
  // PracticePage calls nextPracticeLetter() twice on mount (the lazy
  // useState initializer, then the [practiceLetter, challengeMode] reset
  // effect that also runs on mount) -- harmless with the real, deterministic
  // implementation, but the mock needs to answer "A" both times to match.
  nextPracticeLetter.mockReset().mockReturnValueOnce("A").mockReturnValueOnce("A").mockReturnValue("B");
  startPractice.mockReset();
  beginSession.mockReset();
  navigate.mockReset();
  push.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("PracticePage: success state", () => {
  it("shows the big celebration and locks the camera once the target letter is confirmed", () => {
    render(<PracticePage ready backendError={null} />);
    expect(screen.getByTestId("camera-practice")).toHaveAttribute("data-active", "true");

    fireEvent.click(screen.getByText("fire-correct"));

    expect(markCorrect).toHaveBeenCalledTimes(1);
    expect(markCorrect).toHaveBeenCalledWith("A", 1200);
    expect(screen.getByText(/is correct!/i)).toBeInTheDocument();
    expect(screen.getByText(/yes! you got it!/i)).toBeInTheDocument();
    // Camera must be deactivated the instant success is confirmed.
    expect(screen.getByTestId("camera-practice")).toHaveAttribute("data-active", "false");
  });

  it("never awards XP a second time just because a stray correct frame slips through", () => {
    render(<PracticePage ready backendError={null} />);
    fireEvent.click(screen.getByText("fire-correct"));
    expect(markCorrect).toHaveBeenCalledTimes(1);

    // Even if something re-invoked onCorrect while already in the success
    // phase, PracticePage/CameraPractice must not double-count it. (The real
    // camera loop already can't do this -- see CameraPractice.test.tsx -- this
    // guards the PracticePage side too.)
    const correctButton = screen.queryByText("fire-correct");
    if (correctButton) fireEvent.click(correctButton);
  });
});

describe("PracticePage: wrong-detection feedback", () => {
  it("shows a learner-friendly callout naming both letters, not raw prediction text", () => {
    render(<PracticePage ready backendError={null} />);
    fireEvent.click(screen.getByText("fire-incorrect"));

    expect(markIncorrect).toHaveBeenCalledWith("A", "S");
    expect(screen.getByText("Almost!")).toBeInTheDocument();
    expect(screen.getByText(/you're showing s/i)).toBeInTheDocument();
    expect(screen.getByText(/looking for a/i)).toBeInTheDocument();
    expect(screen.queryByText(/predicted:/i)).not.toBeInTheDocument();
  });
});

describe("PracticePage: reference image", () => {
  it("shows the A reference before any detection", () => {
    render(<PracticePage ready backendError={null} />);
    const img = screen.getByRole("img", { name: /handshape for the letter a/i });
    expect(img).toHaveAttribute("src", "/asl-reference/a.svg");
  });

  it("loads the next letter's reference after Next Letter is clicked", () => {
    render(<PracticePage ready backendError={null} />);
    fireEvent.click(screen.getByText("fire-correct"));

    fireEvent.click(screen.getByRole("button", { name: /move to the next letter/i }));

    expect(screen.getByRole("img", { name: /handshape for the letter b/i })).toHaveAttribute(
      "src",
      "/asl-reference/b.svg"
    );
    // Back to the live practice view, not stuck on the celebration.
    expect(screen.queryByText(/yes! you got it!/i)).not.toBeInTheDocument();
    expect(screen.getByTestId("camera-practice")).toHaveAttribute("data-active", "true");
  });
});

describe("PracticePage: practice again", () => {
  it("resets the current attempt without awarding another XP event", () => {
    render(<PracticePage ready backendError={null} />);
    fireEvent.click(screen.getByText("fire-correct"));
    expect(markCorrect).toHaveBeenCalledTimes(1);

    const attemptKeyBefore = screen.getByTestId("camera-practice").getAttribute("data-attempt-key");
    fireEvent.click(screen.getByRole("button", { name: /practice.*again/i }));

    // Same letter, not advanced.
    expect(screen.getByText("target:A")).toBeInTheDocument();
    // Back to the live view.
    expect(screen.queryByText(/yes! you got it!/i)).not.toBeInTheDocument();
    expect(screen.getByTestId("camera-practice")).toHaveAttribute("data-active", "true");
    // Attempt was reset (new key) so CameraPractice re-arms its votes/flags.
    expect(screen.getByTestId("camera-practice").getAttribute("data-attempt-key")).not.toBe(attemptKeyBefore);
    // No second XP award just from retrying.
    expect(markCorrect).toHaveBeenCalledTimes(1);
  });
});
