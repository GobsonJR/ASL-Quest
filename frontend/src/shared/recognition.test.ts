import { describe, expect, it } from "vitest";
import type { PredictResponse } from "../api";
import {
  HAND_MOVED_IOU,
  LOW_CONFIDENCE_RESET_FRAMES,
  MIN_STABLE_VOTES,
  NO_HAND_GRACE_FRAMES,
  StableLetterTracker,
  boxIou,
  TIP_AFTER_FRAMES,
  classifyFrame,
  majority,
  statusText,
  troubleshootingTip,
} from "./recognition";

function res(overrides: Partial<PredictResponse> = {}): PredictResponse {
  return {
    status: "ok",
    prediction: "A",
    confidence: 0.9,
    top_predictions: [],
    low_confidence: false,
    threshold: 0.7,
    hand_detected: true,
    ...overrides,
  };
}

describe("classifyFrame: hand detection and letter confidence are separate", () => {
  it("no_hand_detected is 'no_hand', with nothing to read", () => {
    expect(classifyFrame(res({ status: "no_hand_detected", prediction: null, hand_detected: false }))).toBe("no_hand");
  });

  it("a low-confidence prediction is NOT treated as no hand", () => {
    const state = classifyFrame(res({ prediction: "S", confidence: 0.31, low_confidence: true }));
    expect(state).toBe("low_confidence");
    expect(state).not.toBe("no_hand");
  });

  it("a confident prediction is 'recognized'", () => {
    expect(classifyFrame(res())).toBe("recognized");
  });

  it("an unavailable detector is its own state, not 'no hand'", () => {
    expect(classifyFrame(res({ status: "hand_detector_unavailable", prediction: null, hand_detected: false }))).toBe(
      "detector_unavailable"
    );
  });

  it("an empty prediction never counts as a recognised letter", () => {
    expect(classifyFrame(res({ prediction: null }))).toBe("no_hand");
  });
});

describe("StableLetterTracker", () => {
  it("needs several agreeing confident frames before a letter is stable", () => {
    const t = new StableLetterTracker();
    for (let i = 1; i < MIN_STABLE_VOTES; i++) {
      expect(t.update("recognized", "C")).toMatchObject({ stable: null, phase: "hand_waiting" });
    }
    expect(t.update("recognized", "C")).toMatchObject({ stable: "C", phase: "stable" });
  });

  it("does not let a single missed-hand frame erase a good run (detectors drop frames)", () => {
    const t = new StableLetterTracker();
    for (let i = 0; i < 2; i++) t.update("recognized", "C");
    expect(t.update("no_hand", null).phase).toBe("no_hand"); // one miss < grace
    expect(t.update("recognized", "C")).toMatchObject({ stable: "C", phase: "stable" });
  });

  it("clears the votes once the hand has really gone", () => {
    const t = new StableLetterTracker();
    for (let i = 0; i < 3; i++) t.update("recognized", "C");
    for (let i = 0; i < NO_HAND_GRACE_FRAMES; i++) t.update("no_hand", null);
    expect(t.update("recognized", "C").stable).toBeNull(); // starts over
  });

  it("low-confidence frames neither add votes nor wipe good ones immediately", () => {
    const t = new StableLetterTracker();
    for (let i = 0; i < 2; i++) t.update("recognized", "C");
    expect(t.update("low_confidence", "S")).toMatchObject({ stable: null, phase: "low_confidence" });
    expect(t.update("recognized", "C")).toMatchObject({ stable: "C" }); // the earlier votes survived
  });

  it("a long weak streak does reset", () => {
    const t = new StableLetterTracker();
    for (let i = 0; i < 2; i++) t.update("recognized", "C");
    for (let i = 0; i < LOW_CONFIDENCE_RESET_FRAMES; i++) t.update("low_confidence", "S");
    expect(t.update("recognized", "C").stable).toBeNull();
  });

  it("a flickering reading never becomes stable", () => {
    const t = new StableLetterTracker();
    const labels = ["A", "B", "A", "B", "C", "B", "A"];
    const results = labels.map((l) => t.update("recognized", l));
    expect(results.every((r) => r.stable === null)).toBe(true);
  });

  it("reset() forgets everything (new session / new attempt)", () => {
    const t = new StableLetterTracker();
    for (let i = 0; i < 3; i++) t.update("recognized", "C");
    t.reset();
    expect(t.update("recognized", "C")).toMatchObject({ stable: null, framesInPhase: 1 });
  });

  it("counts frames spent in the current phase for troubleshooting tips", () => {
    const t = new StableLetterTracker();
    let last = t.update("no_hand", null);
    for (let i = 0; i < 4; i++) last = t.update("no_hand", null);
    expect(last.framesInPhase).toBe(5);
    expect(t.update("recognized", "C").framesInPhase).toBe(1);
  });
});

describe("majority", () => {
  it("returns null below the minimum vote count and for ties", () => {
    expect(majority(["A", "A"])).toBeNull();
    expect(majority(["A", "A", "B", "B"])).toBeNull();
    expect(majority(["A", "A", "A", "B"])).toBe("A");
  });
});

describe("learner-facing text", () => {
  it("each phase has its own message and no-hand differs from low confidence", () => {
    expect(statusText("no_hand")).toMatch(/no hand/i);
    expect(statusText("low_confidence")).toMatch(/hand found/i);
    expect(statusText("low_confidence")).not.toBe(statusText("no_hand"));
  });

  it("troubleshooting appears only after the problem persists", () => {
    expect(troubleshootingTip("no_hand", TIP_AFTER_FRAMES - 1)).toBeNull();
    expect(troubleshootingTip("no_hand", TIP_AFTER_FRAMES)).toMatch(/lighting/i);
    expect(troubleshootingTip("low_confidence", TIP_AFTER_FRAMES)).toMatch(/still/i);
    // Measured on real captures: the face/room inside the crop, not the handshape, defeated recognition.
    expect(troubleshootingTip("low_confidence", TIP_AFTER_FRAMES)).toMatch(/plain (wall|background)/i);
    expect(troubleshootingTip("low_confidence", TIP_AFTER_FRAMES)).toMatch(/face/i);
    expect(troubleshootingTip("stable", 99)).toBeNull();
  });
});

describe("hand-moved reset (mirrors src/inference/stability.py)", () => {
  const BOX: [number, number, number, number] = [100, 100, 300, 300];

  it("computes box overlap", () => {
    expect(boxIou(BOX, BOX)).toBe(1);
    expect(boxIou(BOX, [400, 400, 500, 500])).toBe(0);
  });

  it("small jitter keeps the votes", () => {
    const t = new StableLetterTracker();
    t.update("recognized", "C", BOX);
    t.update("recognized", "C", [104, 98, 305, 302]);
    expect(t.update("recognized", "C", [95, 103, 297, 299]).stable).toBe("C");
  });

  it("votes for the old pose do not confirm a letter after the hand jumps elsewhere", () => {
    const t = new StableLetterTracker();
    t.update("recognized", "W", BOX);
    t.update("recognized", "W", BOX);
    const far: [number, number, number, number] = [600, 400, 800, 600];
    expect(boxIou(BOX, far)).toBeLessThan(HAND_MOVED_IOU);
    expect(t.update("recognized", "W", far).stable).toBeNull();
  });

  it("works without a box (older callers)", () => {
    const t = new StableLetterTracker();
    for (let i = 0; i < 2; i++) t.update("recognized", "C");
    expect(t.update("recognized", "C").stable).toBe("C");
  });
});
