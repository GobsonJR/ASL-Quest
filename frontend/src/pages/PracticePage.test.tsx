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
let mockPracticeLetter: string | null = null;

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
    practiceLetter: mockPracticeLetter,
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
    recognizer?: string;
    onCorrect: (elapsedMs: number, confidence: number) => void;
    onIncorrect: (predicted: string) => void;
  }) => (
    <div data-testid="camera-practice" data-active={String(props.active)} data-attempt-key={props.attemptKey} data-recognizer={props.recognizer}>
      <span>target:{props.targetLetter}</span>
      <button onClick={() => props.onCorrect(1200, 0.87)}>fire-correct</button>
      <button onClick={() => props.onIncorrect("S")}>fire-incorrect</button>
    </div>
  ),
}));

vi.mock("../components/LandmarkPractice", () => ({
  LandmarkPractice: (props: {
    targetLetter: string;
    onCorrect: (elapsedMs: number, confidence: number) => void;
  }) => (
    <div data-testid="landmark-practice">
      <span>target:{props.targetLetter}</span>
      <button onClick={() => props.onCorrect(900, 0.9)}>fire-landmark-correct</button>
    </div>
  ),
}));

beforeEach(() => {
  mockPracticeLetter = null;
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

describe("PracticePage: recognizer selection", () => {
  const LOADED = { loaded: true, model_file: "landmark_L2_s0.joblib", model: "logreg", trained_on: null, error: null };

  it("uses the landmark recognizer by default and offers no recognizer choice when the model is loaded", () => {
    render(<PracticePage ready backendError={null} experimentalLandmark={LOADED} />);
    expect(screen.queryByTestId("camera-practice")).toBeNull();
    expect(screen.getByTestId("landmark-practice")).toBeInTheDocument();
    expect(screen.queryByRole("radiogroup", { name: "Recognizer" })).toBeNull();
    expect(screen.queryByRole("radio")).toBeNull();
  });

  it("falls back to ResNet only when no landmark model is available", () => {
    render(<PracticePage ready backendError={null} />);
    expect(screen.getByTestId("camera-practice")).toHaveAttribute("data-recognizer", "resnet");
    expect(screen.queryByRole("radio")).toBeNull();
  });

  it("ignores a landmark model that failed to load", () => {
    render(<PracticePage ready backendError={null} experimentalLandmark={{ ...LOADED, loaded: false, error: "not found" }} />);
    expect(screen.getByTestId("camera-practice")).toHaveAttribute("data-recognizer", "resnet");
  });

  it("a confirmed landmark sign goes through the same markCorrect/XP path", () => {
    render(<PracticePage ready backendError={null} experimentalLandmark={LOADED} />);
    fireEvent.click(screen.getByRole("button", { name: "fire-landmark-correct" }));
    expect(markCorrect).toHaveBeenCalledTimes(1);
    expect(markCorrect).toHaveBeenCalledWith("A", 900);
  });
});
