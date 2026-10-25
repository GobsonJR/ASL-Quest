import type { PredictResponse } from "../api";

// ---------------------------------------------------------------------------
// Live-recognition state and smoothing. Pure functions/classes (no React, no DOM) so each rule
// is unit-tested. The key idea is that "is there a hand?" and "which letter is it?" are separate
// questions: a weak letter guess never means "no hand", and a missed detection never erases a
// good run of letter votes on its own.
// ---------------------------------------------------------------------------

/** How many recent *confident* readings are considered for the majority vote. */
export const VOTE_WINDOW = 7;
/** A letter is only "stable" once at least this many recent readings agree (and they are a majority). */
export const MIN_STABLE_VOTES = 3;
/** Consecutive no-hand frames tolerated before the vote window is cleared. MediaPipe regularly
 * drops a frame or two; clearing on the first miss made steady signs impossible to confirm. */
export const NO_HAND_GRACE_FRAMES = 2;
/** Consecutive low-confidence frames tolerated before the vote window is cleared. */
export const LOW_CONFIDENCE_RESET_FRAMES = 6;
/** A hand box that overlaps its previous position by less than this (intersection over union) means the hand moved or
 * changed size substantially, so votes for the old pose must not count towards the new one. Mirrors
 * src/inference/stability.py, which measures stability offline. */
export const HAND_MOVED_IOU = 0.35;
/** Upper bound on request rate (about 8 frames/s): keeps the camera responsive and the server calm. */
export const MIN_FRAME_INTERVAL_MS = 120;
/** After this many frames in the same not-yet-recognised state, show troubleshooting advice. */
export const TIP_AFTER_FRAMES = 12;

export type FrameState =
  /** The hand detector is not running on the server, so nothing could be classified. */
  | "detector_unavailable"
  /** No hand was found in the frame. */
  | "no_hand"
  /** A hand was found, but the model's best guess is below the confidence threshold. */
  | "low_confidence"
  /** A hand was found and the model is confident in a letter. */
  | "recognized";

/** What the camera UI is doing, from the learner's point of view. */
export type LivePhase = "starting" | FrameState | "hand_waiting" | "stable" | "error";

export function classifyFrame(response: PredictResponse): FrameState {
  if (response.status === "hand_detector_unavailable") return "detector_unavailable";
  if (response.status === "no_hand_detected" || response.hand_detected === false) return "no_hand";
  if (!response.prediction) return "no_hand"; // nothing was classified, so there is nothing to show
  return response.low_confidence ? "low_confidence" : "recognized";
}

export type TrackerResult = {
  /** The majority letter among recent confident frames, or null if there is no stable letter yet. */
  stable: string | null;
  phase: Exclude<LivePhase, "starting" | "error">;
  /** How many consecutive frames the tracker has seen in the current phase (for troubleshooting tips). */
  framesInPhase: number;
};

export type HandBox = [number, number, number, number];

export function boxIou(a: HandBox, b: HandBox): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

export class StableLetterTracker {
  private votes: string[] = [];
  private missStreak = 0;
  private weakStreak = 0;
  private lastPhase: TrackerResult["phase"] | null = null;
  private phaseFrames = 0;
  private lastBox: HandBox | null = null;

  reset(): void {
    this.votes = [];
    this.missStreak = 0;
    this.weakStreak = 0;
    this.lastPhase = null;
    this.phaseFrames = 0;
    this.lastBox = null;
  }

  update(state: FrameState, label: string | null, box: HandBox | null = null): TrackerResult {
    let phase: TrackerResult["phase"];
    let stable: string | null = null;

    if (state === "detector_unavailable") {
      this.votes = [];
      phase = "detector_unavailable";
    } else if (state === "no_hand") {
      this.missStreak += 1;
      this.weakStreak = 0;
      if (this.missStreak >= NO_HAND_GRACE_FRAMES) {
        this.votes = [];
        this.lastBox = null;
      }
      phase = "no_hand";
    } else if (state === "low_confidence") {
      // A weak reading neither adds a vote nor throws away good ones straight away.
      this.missStreak = 0;
      this.noteBox(box);
      this.weakStreak += 1;
      if (this.weakStreak >= LOW_CONFIDENCE_RESET_FRAMES) this.votes = [];
      phase = "low_confidence";
    } else {
      this.missStreak = 0;
      this.noteBox(box);
      this.weakStreak = 0;
      if (label) this.votes = [...this.votes, label].slice(-VOTE_WINDOW);
      stable = majority(this.votes);
      phase = stable ? "stable" : "hand_waiting";
    }

    this.phaseFrames = phase === this.lastPhase ? this.phaseFrames + 1 : 1;
    this.lastPhase = phase;
    return { stable, phase, framesInPhase: this.phaseFrames };
  }

  /** If the hand jumped to a clearly different place/size, the old votes belong to another pose: drop them. */
  private noteBox(box: HandBox | null): void {
    if (!box) return;
    if (this.lastBox && boxIou(this.lastBox, box) < HAND_MOVED_IOU) {
      this.votes = [];
      this.weakStreak = 0;
    }
    this.lastBox = box;
  }
}

export function majority(labels: string[]): string | null {
  if (labels.length < MIN_STABLE_VOTES) return null;
  const counts = new Map<string, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  const [label, count] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  return count >= Math.max(MIN_STABLE_VOTES, Math.floor(labels.length / 2) + 1) ? label : null;
}

/** Learner-facing one-liner for the status chip. */
export function statusText(phase: LivePhase): string {
  switch (phase) {
    case "starting":
      return "Starting camera…";
    case "detector_unavailable":
      return "Hand detector unavailable";
    case "no_hand":
      return "No hand detected: show your hand";
    case "hand_waiting":
      return "Hand found: hold the sign steady";
    case "low_confidence":
      return "Hand found, but not sure which letter";
    case "recognized":
    case "stable":
      return "Reading your sign…";
    case "error":
      return "Can't reach the recognition service";
  }
}

/** Troubleshooting advice shown only after the problem persists (never on a single dropped frame). */
export function troubleshootingTip(phase: LivePhase, framesInPhase: number): string | null {
  if (framesInPhase < TIP_AFTER_FRAMES) return null;
  switch (phase) {
    case "no_hand":
      return "Move your hand into view, about an arm's length from the camera, with the palm side facing it. Good front lighting and a plain background help.";
    case "low_confidence":
      return "Hold your hand still with all fingers visible, centered in the green box. Your hand in the middle of the box, a little away from your face and in front of a plain background, works best. Match the reference sign on the right.";
    case "detector_unavailable":
      return "The server's hand detector isn't running (see /health on the backend). Check the hand_landmarker.task model file.";
    case "error":
      return "Make sure the backend is running on port 8000, then try again.";
    default:
      return null;
  }
}
