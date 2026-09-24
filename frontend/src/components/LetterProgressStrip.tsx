import type { LearningPathNode } from "../game/types";

/** A compact "letters completed" overview for the A-Z practice page --
 * built on the same getLearningPathNodes(state) data AlphabetLearningPath
 * (LearnPage) uses, so it's a second view over one progress system, not a
 * second progress system. Deliberately just a dot row: AlphabetLearningPath
 * already owns the full interactive zigzag path experience on LearnPage. */
export function LetterProgressStrip({ nodes }: { nodes: LearningPathNode[] }) {
  return (
    <div
      className="flex flex-wrap gap-1.5"
      role="group"
      aria-label="Letters completed"
      data-testid="letter-progress-strip"
    >
      {nodes.map((node) => {
        const toneClass =
          node.status === "completed"
            ? "border-[var(--color-success)]/40 bg-[var(--color-success)]/20 text-[var(--color-success)]"
            : node.status === "current"
              ? "animate-pulse-glow border-[var(--color-accent)] bg-[var(--color-accent)]/15 text-[var(--color-accent)]"
              : "border-[var(--color-line-soft)] bg-transparent text-[var(--color-muted)]";
        return (
          <span
            key={node.letter}
            className={`grid h-6 w-6 place-items-center rounded-full border text-[0.65rem] font-semibold ${toneClass}`}
            title={`${node.letter}: ${node.status}`}
            aria-label={`${node.letter} ${node.status}`}
          >
            {node.status === "completed" ? "✓" : node.letter}
          </span>
        );
      })}
    </div>
  );
}
