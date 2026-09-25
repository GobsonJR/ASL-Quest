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
            ? "bg-[var(--color-success)] text-white"
            : node.status === "current"
              ? "animate-pulse-glow bg-[var(--color-primary)] text-white"
              : "bg-[var(--color-locked-soft)] text-[var(--color-locked)]";
        return (
          <span
            key={node.letter}
            className={`grid h-6 w-6 place-items-center rounded-full text-[0.65rem] font-bold ${toneClass}`}
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
