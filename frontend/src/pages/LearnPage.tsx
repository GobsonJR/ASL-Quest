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
import { SIGN_HINTS, referenceImageAlt, referenceImageSrc } from "./alphabetFeedback";
import { Glyph } from "../components/Icon";

export function LearnPage() {
  const { state, startPractice, navigate } = useGame();
  const { push } = useToast();

  const nodes = getLearningPathNodes(state);
  const currentNode = nodes.find((node) => node.status === "current");
  const masteredCount = lettersMasteredCount(state);
  const allCompleted = masteredCount >= LETTERS.length;
  const progressPercent = (masteredCount / LETTERS.length) * 100;

  function handleLocked(_letter: string, reason: string) {
    push(reason, "info");
  }

  return (
    <PageLayout>
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Learn section">
        <button
          type="button"
          role="tab"
          aria-selected={true}
          className="rounded-[var(--radius-pill)] bg-[var(--color-primary)] px-5 py-2 text-sm font-semibold text-white"
        >
          Alphabet
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={false}
          className="rounded-[var(--radius-pill)] border-2 border-[var(--color-border)] px-5 py-2 text-sm font-semibold text-[var(--color-ink-soft)] transition hover:border-[var(--color-primary-soft)]"
          onClick={() => navigate("words")}
        >
          Words
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={false}
          className="rounded-[var(--radius-pill)] border-2 border-[var(--color-border)] px-5 py-2 text-sm font-semibold text-[var(--color-ink-soft)] transition hover:border-[var(--color-primary-soft)]"
          onClick={() => navigate("native")}
        >
          Native Signs
        </button>
      </div>

      <PageHeader
        eyebrow="Your ASL Journey"
        title="Master the A–Z alphabet"
        description="Learn signs step by step and build your confidence — start with beginner-friendly handshapes, then work through the rest of the alphabet."
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <div className="space-y-6 xl:sticky xl:top-28 xl:self-start">
          {allCompleted ? (
            <Card className="space-y-4 text-center">
              <p className="text-3xl" aria-hidden="true">
                <Glyph value="🎉" />
              </p>
              <h2 className="font-display text-2xl md:text-3xl">Alphabet complete!</h2>
              <p className="text-sm text-[var(--color-ink-soft)]">
                You've completed all {LETTERS.length} ASL letters. Keep any of them sharp any time.
              </p>
              {state.unlockedBadges.includes("alphabet_master") && (
                <p className="text-sm font-semibold text-[var(--color-xp)]">Alphabet Master badge unlocked</p>
              )}
              <PrimaryButton onClick={() => startPractice()}>Practice Any Letter</PrimaryButton>
            </Card>
          ) : (
            currentNode && (
              <Card className="space-y-4">
                <SectionLabel>Continue learning</SectionLabel>
                <div className="flex items-start gap-4">
                  <div className="w-24 shrink-0 overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] p-2">
                    <img
                      src={referenceImageSrc(currentNode.letter)}
                      alt={referenceImageAlt(currentNode.letter)}
                      className="h-auto w-full"
                      width={80}
                      height={105}
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-display text-2xl font-semibold text-[var(--color-ink)]">Letter {currentNode.letter}</h3>
                    <p className="mt-1 text-sm leading-relaxed text-[var(--color-ink-soft)]">
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
            <p className="text-sm text-[var(--color-ink-soft)]">
              {masteredCount} / {LETTERS.length} letters completed
            </p>
            <ProgressBar percent={progressPercent} showPercent={false} />
            <p className="text-sm text-[var(--color-ink-soft)]">{state.streak}-day streak</p>
          </Card>
        </div>

        <section>
          <SectionHeader
            title="Learning path"
            description="Start with easier-to-recognize handshapes, build confidence, then unlock more challenging signs. Completed letters stay open for a revisit any time."
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
