import {
  Card,
  PageHeader,
  PageLayout,
  PrimaryButton,
  ProgressBar,
  SectionHeader,
  SectionLabel,
} from "../components/AppShell";
import { AlphabetLearningPath } from "../components/AlphabetLearningPath";
import { LetterMasteryBar } from "../components/LetterMasteryBar";
import { useToast } from "../components/Toast";
import { LETTERS } from "../game/constants";
import { useGame } from "../game/GameContext";
import { getLearningPathNodes, lettersMasteredCount } from "../game/gamification";
import { referenceImageAlt, referenceImageSrc } from "./alphabetFeedback";

const SIGN_HINTS: Record<string, string> = {
  A: "Make a fist with thumb alongside.",
  B: "Four fingers up, thumb tucked in.",
  C: "Curve your hand like the letter C.",
  D: "Index up, other fingers curled to thumb.",
  E: "All fingertips touch thumb.",
  F: "Index and thumb form a circle, other fingers up.",
  G: "Index and thumb point sideways.",
  H: "Index and middle finger point sideways.",
  I: "Pinky up, other fingers in a fist.",
  J: "Pinky up — hold a static J shape (not motion).",
  K: "Index and middle in a V, thumb between them.",
  L: "Thumb and index form an L.",
  M: "Three fingers over thumb in fist.",
  N: "Two fingers over thumb in fist.",
  O: "Fingers and thumb form a circle.",
  P: "Like K but pointing down.",
  Q: "Like G but pointing down.",
  R: "Index and middle crossed.",
  S: "Closed fist with thumb over fingers.",
  T: "Thumb tucked between index and middle.",
  U: "Index and middle up together.",
  V: "Index and middle spread in a V.",
  W: "Three fingers up in a W shape.",
  X: "Index finger bent like a hook.",
  Y: "Thumb and pinky extended.",
  Z: "Index points — hold a static Z shape (not motion).",
};

export function LearnPage() {
  const { state, startPractice, navigate } = useGame();
  const { push } = useToast();

  const nodes = getLearningPathNodes(state);
  const currentNode = nodes.find((node) => node.status === "current");
  const masteredCount = lettersMasteredCount(state);
  const allCompleted = masteredCount >= LETTERS.length;
  const progressPercent = (masteredCount / LETTERS.length) * 100;

  function handleLocked(letter: string, previousLetter: string) {
    push(`Complete ${previousLetter} first to unlock ${letter}.`, "info");
  }

  return (
    <PageLayout>
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Learn section">
        <button
          type="button"
          role="tab"
          aria-selected={true}
          className="rounded-[var(--radius-control)] bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-[var(--color-ink)]"
        >
          Alphabet
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={false}
          className="rounded-[var(--radius-control)] border border-[var(--color-line)] px-4 py-2 text-sm font-medium text-[var(--color-mist)]"
          onClick={() => navigate("words")}
        >
          Words
        </button>
      </div>

      <PageHeader
        eyebrow="Learn"
        title="Master the A–Z alphabet"
        description="Work through the alphabet in order, one handshape at a time. Static signs only — dynamic J and Z motion is not supported."
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <div className="space-y-6 xl:sticky xl:top-28 xl:self-start">
          {allCompleted ? (
            <Card className="space-y-4 text-center">
              <p className="text-3xl" aria-hidden="true">
                🎉
              </p>
              <h2 className="font-display text-2xl md:text-3xl">Alphabet complete!</h2>
              <p className="text-sm text-[var(--color-mist)]">
                You've completed all {LETTERS.length} ASL letters. Keep any of them sharp any time.
              </p>
              {state.unlockedBadges.includes("alphabet_master") && (
                <p className="text-sm font-medium text-[var(--color-warm)]">🏅 Alphabet Master badge unlocked</p>
              )}
              <PrimaryButton onClick={() => startPractice()}>Practice Any Letter</PrimaryButton>
            </Card>
          ) : (
            currentNode && (
              <Card className="space-y-4">
                <SectionLabel>👋 Continue learning</SectionLabel>
                <div className="flex items-start gap-4">
                  <div className="w-24 shrink-0 overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-line)] bg-white p-2">
                    <img
                      src={referenceImageSrc(currentNode.letter)}
                      alt={referenceImageAlt(currentNode.letter)}
                      className="h-auto w-full"
                      width={80}
                      height={105}
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-display text-2xl">Letter {currentNode.letter}</h3>
                    <p className="mt-1 text-sm leading-relaxed text-[var(--color-mist)]">
                      {SIGN_HINTS[currentNode.letter] ?? `Learn the ${currentNode.letter} handshape.`}
                    </p>
                  </div>
                </div>

                <LetterMasteryBar correct={currentNode.correct} />

                <PrimaryButton className="w-full" onClick={() => startPractice(currentNode.letter)}>
                  Continue →
                </PrimaryButton>
              </Card>
            )
          )}

          <Card className="space-y-3">
            <SectionLabel>Alphabet progress</SectionLabel>
            <p className="text-sm text-[var(--color-mist)]">
              {masteredCount} / {LETTERS.length} letters completed
            </p>
            <ProgressBar percent={progressPercent} showPercent={false} />
            <p className="text-sm text-[var(--color-mist)]">🔥 {state.streak}-day streak</p>
          </Card>
        </div>

        <section>
          <SectionHeader
            title="Learning path"
            description="Complete a letter to unlock the next. Completed letters stay open for a revisit any time."
          />
          <AlphabetLearningPath
            nodes={nodes}
            onPractice={(letter) => startPractice(letter)}
            onLocked={handleLocked}
          />
        </section>
      </div>
    </PageLayout>
  );
}
