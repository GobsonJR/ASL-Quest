import { describe, expect, it } from "vitest";
import { getLearningPathNodes } from "./gamification";
import { createInitialState } from "./storage";
import { LETTERS, MASTERED_CORRECT } from "./constants";
import type { GameState } from "./types";

function withMastered(letters: string[]): GameState {
  const state = createInitialState();
  for (const letter of letters) {
    state.letterStats[letter] = { correct: MASTERED_CORRECT, attempts: MASTERED_CORRECT };
  }
  return state;
}

describe("getLearningPathNodes", () => {
  it("renders all 26 letters in order", () => {
    const nodes = getLearningPathNodes(createInitialState());
    expect(nodes.map((n) => n.letter)).toEqual(LETTERS);
    expect(nodes).toHaveLength(26);
  });

  it("starts with A current and the rest locked", () => {
    const nodes = getLearningPathNodes(createInitialState());
    expect(nodes[0]).toMatchObject({ letter: "A", status: "current" });
    for (const node of nodes.slice(1)) {
      expect(node.status).toBe("locked");
    }
  });

  it("marks A completed and advances B to current once A is mastered", () => {
    const nodes = getLearningPathNodes(withMastered(["A"]));
    expect(nodes[0]).toMatchObject({ letter: "A", status: "completed" });
    expect(nodes[1]).toMatchObject({ letter: "B", status: "current" });
    for (const node of nodes.slice(2)) {
      expect(node.status).toBe("locked");
    }
  });

  it("advances correctly across multiple completed letters", () => {
    const nodes = getLearningPathNodes(withMastered(["A", "B", "C"]));
    expect(nodes[0].status).toBe("completed");
    expect(nodes[1].status).toBe("completed");
    expect(nodes[2].status).toBe("completed");
    expect(nodes[3]).toMatchObject({ letter: "D", status: "current" });
    for (const node of nodes.slice(4)) {
      expect(node.status).toBe("locked");
    }
  });

  it("reflects real out-of-order progress instead of hiding it: a mastered letter is always completed", () => {
    // Free practice (pickPracticeLetter) lets a learner practice any letter,
    // not just the path's current one, so mastery can legitimately land out
    // of A-Z order (e.g. C mastered before B). The path must show that
    // truthfully -- never lock a letter the learner has actually mastered --
    // while "current" still tracks the first not-yet-mastered letter.
    const nodes = getLearningPathNodes(withMastered(["A", "C"]));
    expect(nodes[0].status).toBe("completed"); // A
    expect(nodes[1].status).toBe("current"); // B: first non-mastered letter
    expect(nodes[2].status).toBe("completed"); // C: mastered, stays completed even though B isn't yet
    expect(nodes[2].correct).toBe(MASTERED_CORRECT);
    for (const node of nodes.slice(3)) {
      expect(node.status).toBe("locked");
    }
  });

  it("produces an all-completed path once every letter is mastered", () => {
    const nodes = getLearningPathNodes(withMastered(LETTERS));
    expect(nodes.every((n) => n.status === "completed")).toBe(true);
    expect(nodes.some((n) => n.status === "current")).toBe(false);
  });

  it("never mutates the input state", () => {
    const state = createInitialState();
    const snapshot = JSON.stringify(state);
    getLearningPathNodes(state);
    expect(JSON.stringify(state)).toBe(snapshot);
  });
});
