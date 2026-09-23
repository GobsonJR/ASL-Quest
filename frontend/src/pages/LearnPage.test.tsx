import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { LearnPage } from "./LearnPage";
import { createInitialState } from "../game/storage";
import { MASTERED_CORRECT, LETTERS } from "../game/constants";
import type { GameState } from "../game/types";

const startPractice = vi.fn();
const navigate = vi.fn();
const push = vi.fn();

let mockState: GameState = createInitialState();

vi.mock("../game/GameContext", () => ({
  useGame: () => ({ state: mockState, startPractice, navigate }),
}));

vi.mock("../components/Toast", () => ({
  useToast: () => ({ push }),
}));

function withMastered(letters: string[]): GameState {
  const state = createInitialState();
  for (const letter of letters) {
    state.letterStats[letter] = { correct: MASTERED_CORRECT, attempts: MASTERED_CORRECT };
  }
  return state;
}

beforeEach(() => {
  startPractice.mockReset();
  navigate.mockReset();
  push.mockReset();
  mockState = createInitialState();
});

afterEach(() => {
  cleanup();
});

describe("LearnPage: Continue Learning card", () => {
  it("shows the A reference image and hint before anything is practiced", () => {
    render(<LearnPage />);
    const img = screen.getByRole("img", { name: /handshape for the letter a/i });
    expect(img).toHaveAttribute("src", "/asl-reference/a.svg");
    expect(screen.getByText("Letter A")).toBeInTheDocument();
  });

  it("Continue launches practice for the current letter", () => {
    mockState = withMastered(["A", "B"]);
    render(<LearnPage />);
    expect(screen.getByText("Letter C")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Continue →" }));
    expect(startPractice).toHaveBeenCalledWith("C");
  });
});

describe("LearnPage: progress summary", () => {
  it("uses real mastered-letter counts and streak, not fabricated numbers", () => {
    mockState = withMastered(["A", "B", "C"]);
    mockState.streak = 5;
    render(<LearnPage />);
    expect(screen.getByText(`3 / ${LETTERS.length} letters completed`)).toBeInTheDocument();
    expect(screen.getByText("🔥 5-day streak")).toBeInTheDocument();
  });

  it("reflects zero progress honestly when nothing has been practiced yet", () => {
    render(<LearnPage />);
    expect(screen.getByText(`0 / ${LETTERS.length} letters completed`)).toBeInTheDocument();
  });
});

describe("LearnPage: learning path integration", () => {
  it("renders the full path and lets a locked letter show its explanation via the toast system, not alert()", () => {
    const alertSpy = vi.spyOn(window, "alert").mockImplementation(() => {
      throw new Error("must not call window.alert()");
    });
    render(<LearnPage />);
    fireEvent.click(screen.getByRole("button", { name: /letter c, locked\./i }));
    expect(push).toHaveBeenCalledWith("Complete B first to unlock C.", "info");
    expect(alertSpy).not.toHaveBeenCalled();
    alertSpy.mockRestore();
  });

  it("a completed letter in the path launches practice for that exact letter", () => {
    mockState = withMastered(["A"]);
    render(<LearnPage />);
    fireEvent.click(screen.getByRole("button", { name: /letter a, completed\./i }));
    expect(startPractice).toHaveBeenCalledWith("A");
  });
});

describe("LearnPage: completion state", () => {
  it("shows the alphabet-complete celebration once all 26 letters are mastered", () => {
    mockState = withMastered(LETTERS);
    render(<LearnPage />);
    expect(screen.getByText(/alphabet complete/i)).toBeInTheDocument();
    expect(screen.getByText(`${LETTERS.length} / ${LETTERS.length} letters completed`)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /practice any letter/i }));
    expect(startPractice).toHaveBeenCalledWith();
  });

  it("surfaces the existing Alphabet Master badge instead of inventing a new one, only when actually unlocked", () => {
    mockState = withMastered(LETTERS);
    mockState.unlockedBadges = ["alphabet_master"];
    render(<LearnPage />);
    expect(screen.getByText(/alphabet master/i)).toBeInTheDocument();
  });
});
