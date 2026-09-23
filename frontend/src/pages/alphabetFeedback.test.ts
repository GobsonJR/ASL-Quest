import { describe, expect, it } from "vitest";
import {
  buildIncorrectFeedback,
  pickEncouragement,
  referenceImageAlt,
  referenceImageSrc,
} from "./alphabetFeedback";
import { LETTERS } from "../game/constants";

describe("referenceImageSrc / referenceImageAlt", () => {
  it("points at the public asl-reference folder, lowercased", () => {
    expect(referenceImageSrc("A")).toBe("/asl-reference/a.svg");
    expect(referenceImageSrc("z")).toBe("/asl-reference/z.svg");
  });

  it("produces a distinct, descriptive alt for every letter", () => {
    const alts = new Set(LETTERS.map((letter) => referenceImageAlt(letter)));
    expect(alts.size).toBe(LETTERS.length);
    expect(referenceImageAlt("a")).toBe("ASL fingerspelling handshape for the letter A");
  });

  it("changes with the target letter", () => {
    expect(referenceImageSrc("A")).not.toBe(referenceImageSrc("B"));
    expect(referenceImageAlt("A")).not.toBe(referenceImageAlt("B"));
  });
});

describe("buildIncorrectFeedback", () => {
  it("never just echoes a raw 'Predicted: X' -- names both letters in plain language", () => {
    const feedback = buildIncorrectFeedback("a", "s");
    expect(feedback.headline).toBe("Almost!");
    expect(feedback.detail).toContain("showing S");
    expect(feedback.detail).toContain("looking for A");
    expect(feedback.detail.toLowerCase()).not.toContain("predicted:");
    expect(feedback.detail).toContain("Try matching the handshape shown on the right.");
  });

  it("still gives a specific hint when no stable wrong letter has been detected yet", () => {
    const feedback = buildIncorrectFeedback("b", null);
    expect(feedback.detail).toContain("for B");
    expect(feedback.detail).not.toContain("undefined");
    expect(feedback.detail).not.toContain("null");
  });
});

describe("pickEncouragement", () => {
  it("always includes the target letter", () => {
    for (const letter of ["a", "m", "z"]) {
      expect(pickEncouragement(letter, () => 0)).toContain(letter.toUpperCase());
    }
  });

  it("is deterministic for an injected random source (testable, not flaky)", () => {
    const first = pickEncouragement("c", () => 0);
    const last = pickEncouragement("c", () => 0.999);
    expect(first).not.toBe(last);
    expect(pickEncouragement("c", () => 0)).toBe(first);
  });
});
