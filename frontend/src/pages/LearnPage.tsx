import { useState } from "react";
import {
  MasteryBadge,
  PageHeader,
  PageLayout,
  Panel,
  PrimaryButton,
  SecondaryButton,
  SectionHeader,
} from "../components/AppShell";
import { LetterMasteryBar } from "../components/LetterMasteryBar";
import { LETTERS } from "../game/constants";
import { useGame } from "../game/GameContext";
import { getLetterStatus, getMasteryPercent } from "../game/gamification";

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
  const [selected, setSelected] = useState<string>("A");

  const stats = state.letterStats[selected];
  const status = getLetterStatus(stats.correct);
  const percent = getMasteryPercent(stats.correct);

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
        description="Track progress for each letter. Static signs only — dynamic J and Z motion is not supported."
      />
          <div className="grid gap-8 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
            <Panel className="space-y-5 xl:sticky xl:top-28 xl:self-start">
              <div className="flex items-start gap-4">
                <div className="grid h-24 w-24 shrink-0 place-items-center rounded-[var(--radius-panel)] border border-[var(--color-line-soft)] bg-[var(--color-ink)] font-display text-5xl text-[var(--color-accent)]">
                  {selected}
                </div>
                <div className="min-w-0 flex-1">
                  <MasteryBadge status={status} />
                  <h3 className="mt-2 font-display text-2xl">Letter {selected}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-[var(--color-mist)]">{SIGN_HINTS[selected]}</p>
                </div>
              </div>

              <LetterMasteryBar correct={stats.correct} />

              <div className="flex gap-6 text-sm">
                <div>
                  <p className="text-[var(--color-muted)]">Correct</p>
                  <p className="mt-0.5 text-lg font-semibold">{stats.correct}</p>
                </div>
                <div>
                  <p className="text-[var(--color-muted)]">Mastery</p>
                  <p className="mt-0.5 text-lg font-semibold">{Math.round(percent)}%</p>
                </div>
              </div>

              <div className="flex flex-wrap gap-3">
                <PrimaryButton onClick={() => startPractice(selected)}>Practice {selected}</PrimaryButton>
                <SecondaryButton
                  onClick={() => {
                    const idx = LETTERS.indexOf(selected);
                    setSelected(LETTERS[(idx + 1) % LETTERS.length]);
                  }}
                  ariaLabel="Next letter"
                >
                  Next letter
                </SecondaryButton>
              </div>
            </Panel>

            <section>
              <SectionHeader title="Alphabet map" description="Tap a letter to see details and practice." />
              <div className="mt-4 grid grid-cols-4 gap-2 sm:grid-cols-6 md:grid-cols-7 xl:grid-cols-6">
                {LETTERS.map((letter) => {
                  const letterStats = state.letterStats[letter];
                  const letterStatus = getLetterStatus(letterStats.correct);
                  const letterPercent = getMasteryPercent(letterStats.correct);
                  const active = selected === letter;
                  const mastered = letterStatus === "MASTERED";

                  return (
                    <button
                      key={letter}
                      type="button"
                      className={`rounded-[var(--radius-control)] border px-2 py-2.5 text-left transition ${
                        active
                          ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10"
                          : mastered
                            ? "border-[var(--color-success)]/20 bg-[var(--color-panel-soft)]/50"
                            : "border-[var(--color-line-soft)] bg-[var(--color-panel-soft)]/40 hover:border-[var(--color-line)]"
                      }`}
                      onClick={() => setSelected(letter)}
                      aria-label={`Letter ${letter}, ${letterStatus}, ${Math.round(letterPercent)} percent mastery`}
                      aria-pressed={active}
                    >
                      <span className="font-display text-xl text-[var(--color-accent)]">{letter}</span>
                      <div className="mt-2 h-1 overflow-hidden rounded-full bg-[var(--color-ink)]">
                        <div
                          className="h-full rounded-full bg-[var(--color-accent)] transition-all"
                          style={{ width: `${letterPercent}%` }}
                        />
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>
          </div>
    </PageLayout>
  );
}
