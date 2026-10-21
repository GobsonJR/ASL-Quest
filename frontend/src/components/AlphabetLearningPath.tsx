import type { LearningPathNode } from "../game/types";
import type { ReactNode } from "react";
import { Icon } from "./Icon";
import { BEGINNER_LETTERS } from "../game/constants";

const STATUS_LABEL: Record<LearningPathNode["status"], string> = {
  completed: "Completed",
  current: "Continue",
  available: "Available",
  locked: "Locked",
};

// Alternates node alignment across three lanes for a gentle left/center/right
// wave down the page -- inspired by progressive learning-path UIs, but a
// plain flex column (no absolute positioning, no SVG curve) so it can never
// overflow or misbehave at any width, including 320px.
const LANE_ALIGN = ["self-start", "self-center", "self-end"] as const;

function nodeIcon(node: LearningPathNode): ReactNode {
  if (node.status === "completed") return <Icon name="check" className="h-8 w-8" />;
  if (node.status === "locked") return <Icon name="lock" className="h-7 w-7" />;
  return node.letter;
}

export function lockedExplanation(nodes: LearningPathNode[]): string {
  const stillBeginner = nodes.some(
    (n) => BEGINNER_LETTERS.includes(n.letter) && n.status !== "completed"
  );
  if (stillBeginner) {
    return `Practice the ${BEGINNER_LETTERS.length} beginner signs (${BEGINNER_LETTERS.join(", ")}) first to unlock more letters.`;
  }
  const current = nodes.find((n) => n.status === "current");
  return current
    ? `Complete ${current.letter} first to unlock more letters.`
    : `Keep practicing to unlock more letters.`;
}

function NodeButton({
  node,
  onPractice,
  onLocked,
  lockedReason,
}: {
  node: LearningPathNode;
  onPractice: (letter: string) => void;
  onLocked: (letter: string, reason: string) => void;
  lockedReason: string;
}) {
  const label =
    node.status === "completed"
      ? `Letter ${node.letter}, completed. Practice ${node.letter}.`
      : node.status === "current"
        ? `Letter ${node.letter}, current. Continue learning ${node.letter}.`
        : node.status === "available"
          ? `Letter ${node.letter}, available. Practice ${node.letter}.`
          : `Letter ${node.letter}, locked. ${lockedReason}`;

  return (
    <button
      type="button"
      onClick={() => (node.status === "locked" ? onLocked(node.letter, lockedReason) : onPractice(node.letter))}
      aria-label={label}
      // Deliberately NOT aria-disabled: the button still does something
      // meaningful on click/Enter (surfaces the unlock explanation), which
      // is exactly what aria-disabled tells assistive tech NOT to expect. A
      // native `disabled` button would be worse still -- unreachable by
      // keyboard, so the explanation could never be discovered at all.
      className={`relative grid place-items-center rounded-full font-display text-2xl font-bold shadow-[0_4px_0_0_rgba(0,0,0,0.08)] transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-primary)] ${
        node.status === "current"
          ? "h-20 w-20 animate-pulse-glow bg-[var(--color-primary)] text-white"
          : node.status === "completed"
            ? "h-16 w-16 bg-[var(--color-success)] text-white"
            : node.status === "available"
              ? "h-16 w-16 bg-[var(--color-native)] text-white"
              : "h-16 w-16 bg-[var(--color-locked-soft)] text-[var(--color-locked)] shadow-none"
      }`}
    >
      {nodeIcon(node)}
      {node.status === "current" && (
        <span
          aria-hidden="true"
          className="absolute -bottom-1 -right-1 grid h-6 w-6 place-items-center rounded-full border-2 border-[var(--color-surface)] bg-[var(--color-xp)] text-xs text-white"
        ><Icon name="play" className="h-3 w-3" /></span>
      )}
    </button>
  );
}

/** Reusable Duolingo-inspired A-Z learning path. Purely presentational and
 * derived-state-in: it renders whatever `nodes` (see getLearningPathNodes)
 * says, and reports intent via onPractice/onLocked -- it never launches
 * practice or shows a toast itself, so it stays easy to reuse/test.
 *
 * Nodes stay in their canonical A-Z order (same array LetterProgressStrip
 * uses), but are split into two zigzag lanes for display: everything
 * unlocked (current/available/completed) forms the "start here" path,
 * everything locked follows below a divider. During the beginner phase this
 * naturally surfaces exactly BEGINNER_LETTERS as the path, since every other
 * letter is locked -- no separate ordering config needed. */
export function AlphabetLearningPath({
  nodes,
  onPractice,
  onLocked,
}: {
  nodes: LearningPathNode[];
  onPractice: (letter: string) => void;
  onLocked: (letter: string, reason: string) => void;
}) {
  const unlockedNodes = nodes.filter((n) => n.status !== "locked");
  const lockedNodes = nodes.filter((n) => n.status === "locked");
  const isBeginnerPhase = unlockedNodes.some((n) => n.status === "current" || n.status === "available");
  const lockedReason = lockedExplanation(nodes);

  return (
    <div className="space-y-4">
      {isBeginnerPhase && (
        <p className="text-xs font-bold uppercase tracking-[0.08em] text-[var(--color-primary)]">
          Beginner signs · Start here
        </p>
      )}
      <ol className="relative flex flex-col items-stretch gap-3 py-2" aria-label="A-Z alphabet learning path">
        {unlockedNodes.map((node, index) => (
          <li
            key={node.letter}
            className={`flex w-fit max-w-[40%] flex-col items-center gap-1.5 sm:max-w-[30%] ${LANE_ALIGN[index % LANE_ALIGN.length]}`}
          >
            {index > 0 && <span aria-hidden="true" className="h-6 w-1 rounded-full bg-[var(--color-border)]" />}
            <NodeButton node={node} onPractice={onPractice} onLocked={onLocked} lockedReason={lockedReason} />
            <span
              className={`text-[0.65rem] font-bold uppercase tracking-[0.08em] ${
                node.status === "current"
                  ? "text-[var(--color-primary)]"
                  : node.status === "completed"
                    ? "text-[var(--color-success)]"
                    : "text-[var(--color-native)]"
              }`}
            >
              {STATUS_LABEL[node.status]}
            </span>
          </li>
        ))}
      </ol>

      {lockedNodes.length > 0 && (
        <>
          <div className="flex items-center gap-3 text-[0.65rem] font-bold uppercase tracking-[0.08em] text-[var(--color-muted)]">
            <span className="h-px flex-1 bg-[var(--color-border)]" aria-hidden="true" />
            More signs
            <span className="h-px flex-1 bg-[var(--color-border)]" aria-hidden="true" />
          </div>
          <ol className="flex flex-wrap justify-center gap-2" aria-label="Locked letters">
            {lockedNodes.map((node) => (
              <li key={node.letter}>
                <NodeButton node={node} onPractice={onPractice} onLocked={onLocked} lockedReason={lockedReason} />
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}
