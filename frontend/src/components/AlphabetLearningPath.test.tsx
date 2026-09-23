import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AlphabetLearningPath } from "./AlphabetLearningPath";
import { getLearningPathNodes } from "../game/gamification";
import { createInitialState } from "../game/storage";
import { MASTERED_CORRECT } from "../game/constants";
import type { GameState } from "../game/types";

function withMastered(letters: string[]): GameState {
  const state = createInitialState();
  for (const letter of letters) {
    state.letterStats[letter] = { correct: MASTERED_CORRECT, attempts: MASTERED_CORRECT };
  }
  return state;
}

describe("AlphabetLearningPath", () => {
  it("renders all 26 letter nodes", () => {
    const nodes = getLearningPathNodes(createInitialState());
    render(<AlphabetLearningPath nodes={nodes} onPractice={vi.fn()} onLocked={vi.fn()} />);
    expect(screen.getAllByRole("button")).toHaveLength(26);
  });

  it("clicking a completed letter launches practice for it, not a locked letter", () => {
    const nodes = getLearningPathNodes(withMastered(["A"]));
    const onPractice = vi.fn();
    render(<AlphabetLearningPath nodes={nodes} onPractice={onPractice} onLocked={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /letter a, completed\. practice a\./i }));
    expect(onPractice).toHaveBeenCalledWith("A");
  });

  it("clicking the current letter launches practice for it", () => {
    const nodes = getLearningPathNodes(createInitialState());
    const onPractice = vi.fn();
    render(<AlphabetLearningPath nodes={nodes} onPractice={onPractice} onLocked={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /letter a, current\. continue learning a\./i }));
    expect(onPractice).toHaveBeenCalledWith("A");
  });

  it("clicking a locked letter never starts practice, and explains what unlocks it", () => {
    const nodes = getLearningPathNodes(createInitialState());
    const onPractice = vi.fn();
    const onLocked = vi.fn();
    render(<AlphabetLearningPath nodes={nodes} onPractice={onPractice} onLocked={onLocked} />);

    const lockedC = screen.getByRole("button", { name: /letter c, locked\./i });
    expect(lockedC).toHaveAttribute("aria-label", "Letter C, locked. Complete B first to unlock C.");
    fireEvent.click(lockedC);

    expect(onPractice).not.toHaveBeenCalled();
    expect(onLocked).toHaveBeenCalledWith("C", "B");
  });

  it("locked nodes are plain, fully-interactive buttons (not `disabled` or `aria-disabled`), so both keyboard and pointer users reach the explanation", () => {
    const nodes = getLearningPathNodes(createInitialState());
    render(<AlphabetLearningPath nodes={nodes} onPractice={vi.fn()} onLocked={vi.fn()} />);
    const lockedZ = screen.getByRole("button", { name: /letter z, locked\./i });
    expect(lockedZ).not.toBeDisabled();
    expect(lockedZ).not.toHaveAttribute("aria-disabled");
  });
});
