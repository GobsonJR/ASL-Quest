// Pure helpers for the A-Z practice page's learner-facing copy and reference
// image lookup. Kept free of React/DOM so they're trivial to unit test, and
// so CameraPractice/PracticePage/AlphabetReferencePanel all derive the same
// wording and file paths from one place.

/** Public-domain ASL fingerspelling reference images live in
 * public/asl-reference/<letter>.svg — see that folder's ATTRIBUTION.md. */
export function referenceImageSrc(letter: string): string {
  return `/asl-reference/${letter.trim().toLowerCase()}.svg`;
}

export function referenceImageAlt(letter: string): string {
  return `ASL fingerspelling handshape for the letter ${letter.trim().toUpperCase()}`;
}

export const REFERENCE_LICENSE_NOTE =
  "Public-domain ASL fingerspelling reference (wpclipart.com, via Wikimedia Commons).";

// A learner who just missed the sign should never see only "Predicted: S" --
// this turns the (already temporally-smoothed/debounced) wrong stable letter
// into an encouraging, specific hint. `predicted` is null when the model
// hasn't settled on any stable letter yet (e.g. no hand steady in frame).
export function buildIncorrectFeedback(
  target: string,
  predicted: string | null
): { headline: string; detail: string } {
  const targetLetter = target.trim().toUpperCase();
  if (!predicted) {
    return {
      headline: "Almost!",
      detail: `Try matching the handshape shown on the right for ${targetLetter}.`,
    };
  }
  const predictedLetter = predicted.trim().toUpperCase();
  return {
    headline: "Almost!",
    detail:
      `You're showing ${predictedLetter}, but we're looking for ${targetLetter}. ` +
      "Try matching the handshape shown on the right.",
  };
}

const ENCOURAGEMENTS = [
  "Amazing! That's {letter}!",
  "Perfect! You nailed {letter}!",
  "Great job! That's {letter}!",
  "Beautiful! That's exactly {letter}.",
  "Nailed it — {letter} looked great.",
] as const;

/** `random` is injectable (defaults to Math.random) so tests can pick a
 * deterministic message instead of asserting against a moving target. */
export function pickEncouragement(letter: string, random: () => number = Math.random): string {
  const index = Math.min(ENCOURAGEMENTS.length - 1, Math.floor(random() * ENCOURAGEMENTS.length));
  return ENCOURAGEMENTS[index].replace("{letter}", letter.trim().toUpperCase());
}
