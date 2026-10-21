import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AlphabetLearningPath } from "./AlphabetLearningPath";
import { getLearningPathNodes } from "../game/gamification";
import { createInitialState } from "../game/storage";
import { BEGINNER_LETTERS, MASTERED_CORRECT } from "../game/constants";
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
    const nodes = getLearningPathNodes(withMastered(["C"]));
    const onPractice = vi.fn();
    render(<AlphabetLearningPath nodes={nodes} onPractice={onPractice} onLocked={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /letter c, completed\. practice c\./i }));
    expect(onPractice).toHaveBeenCalledWith("C");
  });

  it("clicking the current letter launches practice for it", () => {
    const nodes = getLearningPathNodes(createInitialState());
    const onPractice = vi.fn();
    render(<AlphabetLearningPath nodes={nodes} onPractice={onPractice} onLocked={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /letter c, current\. continue learning c\./i }));
    expect(onPractice).toHaveBeenCalledWith("C");
  });

  it("clicking an available (non-current, beginner-unlocked) letter launches practice for it", () => {
    const nodes = getLearningPathNodes(createInitialState());
    const onPractice = vi.fn();
    render(<AlphabetLearningPath nodes={nodes} onPractice={onPractice} onLocked={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: /letter d, available\. practice d\./i }));
    expect(onPractice).toHaveBeenCalledWith("D");
  });

  it("all 5 beginner letters (current + available) are clickable and open practice, none are locked", () => {
    const nodes = getLearningPathNodes(createInitialState());
    const onPractice = vi.fn();
    render(<AlphabetLearningPath nodes={nodes} onPractice={onPractice} onLocked={vi.fn()} />);

    for (const letter of BEGINNER_LETTERS) {
      const button = screen.getByRole("button", { name: new RegExp(`^letter ${letter.toLowerCase()}, (current|available)`, "i") });
      fireEvent.click(button);
    }
    expect(onPractice.mock.calls.map((call) => call[0]).sort()).toEqual([...BEGINNER_LETTERS].sort());
  });

  it("clicking a locked letter never starts practice, and explains what unlocks it", () => {
    const nodes = getLearningPathNodes(createInitialState());
    const onPractice = vi.fn();
    const onLocked = vi.fn();
    render(<AlphabetLearningPath nodes={nodes} onPractice={onPractice} onLocked={onLocked} />);

    const lockedA = screen.getByRole("button", { name: /letter a, locked\./i });
    fireEvent.click(lockedA);

    expect(onPractice).not.toHaveBeenCalled();
    expect(onLocked).toHaveBeenCalledWith("A", expect.stringMatching(/beginner signs/i));
  });

  it("locked nodes are plain, fully-interactive buttons (not `disabled` or `aria-disabled`), so both keyboard and pointer users reach the explanation", () => {
    const nodes = getLearningPathNodes(createInitialState());
    render(<AlphabetLearningPath nodes={nodes} onPractice={vi.fn()} onLocked={vi.fn()} />);
    const lockedZ = screen.getByRole("button", { name: /letter z, locked\./i });
    expect(lockedZ).not.toBeDisabled();
    expect(lockedZ).not.toHaveAttribute("aria-disabled");
  });

  it("shows a 'Beginner signs · Start here' heading while the beginner set is still unlocking", () => {
    const nodes = getLearningPathNodes(createInitialState());
    render(<AlphabetLearningPath nodes={nodes} onPractice={vi.fn()} onLocked={vi.fn()} />);
    expect(screen.getByText(/beginner signs.*start here/i)).toBeInTheDocument();
  });
});
