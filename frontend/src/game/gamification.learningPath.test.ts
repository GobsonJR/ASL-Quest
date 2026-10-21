import { describe, expect, it } from "vitest";
import { getLearningPathNodes } from "./gamification";
import { createInitialState } from "./storage";
import { BEGINNER_LETTERS, LETTERS, MASTERED_CORRECT } from "./constants";
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

  it("initial roadmap has exactly 5 beginner-unlocked letters: 1 current + 4 available, and 21 locked", () => {
    const nodes = getLearningPathNodes(createInitialState());

    const current = nodes.filter((n) => n.status === "current");
    const available = nodes.filter((n) => n.status === "available");
    const locked = nodes.filter((n) => n.status === "locked");

    expect(current).toHaveLength(1);
    expect(available).toHaveLength(4);
    expect(locked).toHaveLength(21);

    const unlockedLetters = [...current, ...available].map((n) => n.letter).sort();
    expect(unlockedLetters).toEqual([...BEGINNER_LETTERS].sort());
  });

  it("the first beginner letter (A-Z order) is current, the rest of the beginner set is available", () => {
    const nodes = getLearningPathNodes(createInitialState());
    const firstBeginner = [...BEGINNER_LETTERS].sort()[0];
    const node = nodes.find((n) => n.letter === firstBeginner)!;
    expect(node.status).toBe("current");

    for (const letter of BEGINNER_LETTERS) {
      if (letter === firstBeginner) continue;
      const beginnerNode = nodes.find((n) => n.letter === letter)!;
      expect(beginnerNode.status).toBe("available");
    }
  });

  it("non-beginner letters stay locked until every beginner letter is mastered", () => {
    const nodes = getLearningPathNodes(createInitialState());
    for (const node of nodes) {
      if (!BEGINNER_LETTERS.includes(node.letter)) {
        expect(node.status).toBe("locked");
      }
    }
  });

  it("mastering one beginner letter marks it completed and the rest remain unlocked (never locked)", () => {
    const firstBeginner = [...BEGINNER_LETTERS].sort()[0];
    const nodes = getLearningPathNodes(withMastered([firstBeginner]));
    const completedNode = nodes.find((n) => n.letter === firstBeginner)!;
    expect(completedNode.status).toBe("completed");

    for (const letter of BEGINNER_LETTERS) {
      if (letter === firstBeginner) continue;
      const node = nodes.find((n) => n.letter === letter)!;
      expect(["current", "available"]).toContain(node.status);
    }

    // non-beginner letters are still locked -- mastering one beginner letter
    // doesn't open the rest of the alphabet.
    for (const node of nodes) {
      if (!BEGINNER_LETTERS.includes(node.letter)) {
        expect(node.status).toBe("locked");
      }
    }
  });

  it("mastering letters out of beginner order still updates real progress for each one", () => {
    const outOfOrder = [...BEGINNER_LETTERS].sort().slice(-1); // last beginner letter alphabetically
    const nodes = getLearningPathNodes(withMastered(outOfOrder));
    const node = nodes.find((n) => n.letter === outOfOrder[0])!;
    expect(node.status).toBe("completed");
    expect(node.correct).toBe(MASTERED_CORRECT);
  });

  it("unlocks sequential progression into the rest of the alphabet only once all 5 beginner letters are completed", () => {
    const nodes = getLearningPathNodes(withMastered(BEGINNER_LETTERS));

    for (const letter of BEGINNER_LETTERS) {
      expect(nodes.find((n) => n.letter === letter)!.status).toBe("completed");
    }

    const firstNonBeginner = LETTERS.find((letter) => !BEGINNER_LETTERS.includes(letter))!;
    expect(nodes.find((n) => n.letter === firstNonBeginner)!.status).toBe("current");

    for (const node of nodes) {
      if (!BEGINNER_LETTERS.includes(node.letter) && node.letter !== firstNonBeginner) {
        expect(node.status).toBe("locked");
      }
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
