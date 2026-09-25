import type { LearningPathNode } from "../game/types";

const STATUS_LABEL: Record<LearningPathNode["status"], string> = {
  completed: "Completed",
  current: "Continue",
  locked: "Locked",
};

// Alternates node alignment across three lanes for a gentle left/center/right
// wave down the page -- inspired by progressive learning-path UIs, but a
// plain flex column (no absolute positioning, no SVG curve) so it can never
// overflow or misbehave at any width, including 320px.
const LANE_ALIGN = ["self-start", "self-center", "self-end"] as const;

function nodeIcon(node: LearningPathNode): string {
  if (node.status === "completed") return "✓";
  if (node.status === "locked") return "🔒";
  return node.letter;
}

/** Reusable Duolingo-inspired A-Z learning path. Purely presentational and
 * derived-state-in: it renders whatever `nodes` (see getLearningPathNodes)
 * says, and reports intent via onPractice/onLocked -- it never launches
 * practice or shows a toast itself, so it stays easy to reuse/test. */
export function AlphabetLearningPath({
  nodes,
  onPractice,
  onLocked,
}: {
  nodes: LearningPathNode[];
  onPractice: (letter: string) => void;
  onLocked: (letter: string, previousLetter: string) => void;
}) {
  return (
    <ol className="relative flex flex-col items-stretch gap-3 py-2" aria-label="A-Z alphabet learning path">
      {nodes.map((node, index) => {
        const lane = LANE_ALIGN[index % LANE_ALIGN.length];
        const previousLetter = nodes[index - 1]?.letter;
        const label =
          node.status === "completed"
            ? `Letter ${node.letter}, completed. Practice ${node.letter}.`
            : node.status === "current"
              ? `Letter ${node.letter}, current. Continue learning ${node.letter}.`
              : `Letter ${node.letter}, locked. Complete ${previousLetter} first to unlock ${node.letter}.`;

        return (
          <li key={node.letter} className={`flex w-fit max-w-[40%] flex-col items-center gap-1.5 sm:max-w-[30%] ${lane}`}>
            {index > 0 && (
              <span
                aria-hidden="true"
                className={`h-6 w-1 rounded-full ${
                  nodes[index - 1].status === "locked" ? "bg-[var(--color-border-soft)]" : "bg-[var(--color-border)]"
                }`}
              />
            )}
            <button
              type="button"
              onClick={() => (node.status === "locked" ? onLocked(node.letter, previousLetter ?? node.letter) : onPractice(node.letter))}
              aria-label={label}
              // Deliberately NOT aria-disabled: the button still does
              // something meaningful on click/Enter (surfaces the unlock
              // explanation), which is exactly what aria-disabled tells
              // assistive tech NOT to expect. A native `disabled` button
              // would be worse still -- unreachable by keyboard, so the
              // explanation could never be discovered at all.
              className={`relative grid place-items-center rounded-full font-display text-2xl font-bold shadow-[0_4px_0_0_rgba(0,0,0,0.08)] transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-primary)] ${
                node.status === "current"
                  ? "h-20 w-20 animate-pulse-glow bg-[var(--color-primary)] text-white"
                  : node.status === "completed"
                    ? "h-16 w-16 bg-[var(--color-success)] text-white"
                    : "h-16 w-16 bg-[var(--color-locked-soft)] text-[var(--color-locked)] shadow-none"
              }`}
            >
              {nodeIcon(node)}
              {node.status === "current" && (
                <span
                  aria-hidden="true"
                  className="absolute -bottom-1 -right-1 grid h-6 w-6 place-items-center rounded-full border-2 border-[var(--color-surface)] bg-[var(--color-xp)] text-xs text-white"
                >
                  ▶
                </span>
              )}
            </button>
            <span
              className={`text-[0.65rem] font-bold uppercase tracking-[0.08em] ${
                node.status === "current"
                  ? "text-[var(--color-primary)]"
                  : node.status === "completed"
                    ? "text-[var(--color-success)]"
                    : "text-[var(--color-muted)]"
              }`}
            >
              {STATUS_LABEL[node.status]}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
