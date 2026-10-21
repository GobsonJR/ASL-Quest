import { useState } from "react";
import {
  Card,
  PageHeader,
  PageLayout,
  PrimaryButton,
  ProgressBar,
  SectionLabel,
  StatusChip,
} from "../components/AppShell";
import { lockedExplanation } from "../components/AlphabetLearningPath";
import { LetterMasteryBar } from "../components/LetterMasteryBar";
import { LETTERS } from "../game/constants";
import { useSignoEmit } from "../components/signo/SignoContext";
import { useGame } from "../game/GameContext";
import { getLearningPathNodes, lettersMasteredCount } from "../game/gamification";
import type { LearningPathNode } from "../game/types";
import { REFERENCE_LICENSE_NOTE, SIGN_HINTS, referenceImageAlt, referenceImageSrc } from "./alphabetFeedback";
import { Glyph } from "../components/Icon";

type Filter = "all" | "completed" | "unlocked" | "locked";

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: "all", label: "All" },
  { id: "completed", label: "Completed" },
  { id: "unlocked", label: "Unlocked" },
  { id: "locked", label: "Locked" },
];

const STATUS_TEXT: Record<LearningPathNode["status"], string> = {
  completed: "Completed",
  current: "Up next",
  available: "Unlocked",
  locked: "Locked",
};

function matchesFilter(node: LearningPathNode, filter: Filter): boolean {
  if (filter === "all") return true;
  if (filter === "completed") return node.status === "completed";
  if (filter === "locked") return node.status === "locked";
  return node.status === "current" || node.status === "available";
}

function tileClass(node: LearningPathNode, selected: boolean): string {
  const tone =
    node.status === "completed"
      ? "border-[var(--color-primary-dim)] bg-[var(--color-primary-soft)] text-[var(--color-primary-dim)]"
      : node.status === "current"
        ? "border-[var(--color-primary)] bg-[var(--color-surface)] text-[var(--color-primary)]"
        : node.status === "available"
          ? "border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-ink)]"
          : "border-[var(--color-border-soft)] bg-[var(--color-locked-soft)] text-[var(--color-locked)]";
  return `${tone} ${selected ? "ring-4 ring-[var(--color-primary-soft)] -translate-y-0.5" : ""}`;
}

export function AlphabetPage() {
  const { state, startPractice } = useGame();
  const emit = useSignoEmit();
  const nodes = getLearningPathNodes(state);
  const current = nodes.find((node) => node.status === "current");
  const [selected, setSelected] = useState<string>(current?.letter ?? LETTERS[0]);
  const [filter, setFilter] = useState<Filter>("all");

  const node = nodes.find((n) => n.letter === selected) ?? nodes[0];
  const locked = node.status === "locked";
  const mastered = lettersMasteredCount(state);
  const visible = nodes.filter((n) => matchesFilter(n, filter));

  return (
    <PageLayout>
      <PageHeader
        eyebrow="A–Z"
        title="The ASL alphabet"
        description="Pick a letter to see its handshape, then practice it with your camera. Only static handshapes are recognized — J and Z are taught as held shapes, not motion."
      />

      <Card className="space-y-2">
        <p className="text-sm font-semibold text-[var(--color-ink)]">
          {mastered} / {LETTERS.length} letters completed
        </p>
        <ProgressBar percent={(mastered / LETTERS.length) * 100} showPercent={false} />
      </Card>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <section className="min-w-0 space-y-4" aria-label="Letters">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Filter letters">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                type="button"
                aria-pressed={filter === f.id}
                onClick={() => setFilter(f.id)}
                className={`min-h-11 rounded-[var(--radius-pill)] border-2 px-4 text-sm font-bold transition ${
                  filter === f.id
                    ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)] text-[var(--color-primary-dim)]"
                    : "border-[var(--color-border)] text-[var(--color-ink-soft)] hover:border-[var(--color-primary-soft)]"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          {visible.length === 0 ? (
            <Card className="text-center text-sm text-[var(--color-ink-soft)]">
              No letters match this filter yet.
            </Card>
          ) : (
            <ul className="grid grid-cols-4 gap-3 sm:grid-cols-6 lg:grid-cols-5 xl:grid-cols-7" data-testid="alphabet-grid">
              {visible.map((n) => (
                <li key={n.letter}>
                  <button
                    type="button"
                    onClick={() => {
                      setSelected(n.letter);
                      if (n.status !== "locked") emit({ type: "letter_selected", letter: n.letter });
                    }}
                    aria-pressed={selected === n.letter}
                    aria-label={`Letter ${n.letter}, ${STATUS_TEXT[n.status].toLowerCase()}`}
                    className={`flex aspect-square w-full flex-col items-center justify-center rounded-2xl border-2 border-b-4 font-display text-2xl font-black transition ${tileClass(n, selected === n.letter)}`}
                  >
                    {n.status === "locked" ? <span aria-hidden="true" className="text-lg"><Glyph value="🔒" /></span> : n.letter}
                    <span className="sr-only">{n.letter}</span>
                    {n.status === "completed" && (
                      <span aria-hidden="true" className="text-xs leading-none">✓</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <Card className="min-w-0 space-y-4 lg:sticky lg:top-24 lg:self-start" data-testid="alphabet-detail">
          <div className="flex items-center justify-between gap-2">
            <SectionLabel>Letter {node.letter}</SectionLabel>
            <StatusChip tone={node.status === "completed" ? "success" : locked ? "neutral" : "accent"}>
              {STATUS_TEXT[node.status]}
            </StatusChip>
          </div>

          {locked ? (
            <div className="space-y-3 rounded-[var(--radius-panel)] bg-[var(--color-locked-soft)] p-4 text-center">
              <p className="text-3xl" aria-hidden="true"><Glyph value="🔒" /></p>
              <p className="text-sm text-[var(--color-ink-soft)]">{lockedExplanation(nodes)}</p>
            </div>
          ) : (
            <>
              <div className="mx-auto w-full max-w-[200px] overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] p-3">
                <img
                  src={referenceImageSrc(node.letter)}
                  alt={referenceImageAlt(node.letter)}
                  className="h-auto w-full"
                  width={200}
                  height={263}
                />
              </div>
              <p className="text-center text-sm leading-relaxed text-[var(--color-ink-soft)]">
                {SIGN_HINTS[node.letter] ?? `Learn the ${node.letter} handshape.`}
              </p>
              <LetterMasteryBar correct={node.correct} />
              <PrimaryButton className="w-full" onClick={() => startPractice(node.letter)}>
                Practice {node.letter} with camera
              </PrimaryButton>
              <p className="text-center text-xs text-[var(--color-muted)]">
                Needs camera access and the ASL Quest backend. {REFERENCE_LICENSE_NOTE}
              </p>
            </>
          )}
        </Card>
      </div>
    </PageLayout>
  );
}
