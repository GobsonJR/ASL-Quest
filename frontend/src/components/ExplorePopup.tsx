import type { NavPage } from "../game/constants";
import { Dialog } from "./Dialog";
import { Glyph } from "./Icon";

type Path = {
  id: Extract<NavPage, "alphabet" | "words" | "native">;
  icon: string;
  title: string;
  tagline: string;
  blurb: string;
  tint: string;
  accent: string;
};

const PATHS: Path[] = [
  {
    id: "alphabet",
    icon: "🔤",
    title: "Alphabet",
    tagline: "Master A–Z",
    blurb: "Learn all 26 handshapes step by step, with your camera cheering you on.",
    tint: "var(--color-primary-soft)",
    accent: "var(--color-primary-dim)",
  },
  {
    id: "words",
    icon: "🧩",
    title: "Words",
    tagline: "Spell it out",
    blurb: "Put your letters together and fingerspell everyday words.",
    tint: "var(--color-xp-soft)",
    accent: "var(--color-ink)",
  },
  {
    id: "native",
    icon: "🎥",
    title: "Native Signs",
    tagline: "Sign whole words",
    blurb: "Watch the demo, copy the sign, and learn real ASL one word at a time.",
    tint: "var(--color-native-soft)",
    accent: "var(--color-native)",
  },
];

/** A centered "choose your path" popup opened from the sidebar. Reuses the app's Dialog (focus trap, Escape, click outside). */
export function ExplorePopup({
  lettersMastered,
  onSelect,
  onClose,
}: {
  lettersMastered: number;
  onSelect: (page: NavPage) => void;
  onClose: () => void;
}) {
  return (
    <Dialog labelledBy="explore-title" onClose={onClose} className="w-full max-w-3xl">
      <div className="animate-pop overflow-hidden rounded-[2rem] border-2 border-[var(--color-border)] bg-[var(--color-surface)] shadow-2xl">
        <div className="relative bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-native)] px-6 py-7 text-center text-white sm:px-10">
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="absolute right-4 top-4 grid h-9 w-9 place-items-center rounded-full bg-white/20 text-lg font-bold transition hover:bg-white/35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
          >
            <span aria-hidden="true">×</span>
          </button>
          <p className="text-xs font-extrabold uppercase tracking-[0.16em] text-white/85">Your ASL journey</p>
          <h2 id="explore-title" className="mt-1 font-display text-3xl font-bold sm:text-4xl">
            Where would you like to go?
          </h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-white/90">
            Three ways to learn, all at your own pace. Pick one and jump right in.
          </p>
        </div>

        <ul className="grid gap-4 p-5 sm:grid-cols-3 sm:p-7">
          {PATHS.map((path) => (
            <li key={path.id}>
              <button
                type="button"
                onClick={() => onSelect(path.id)}
                className="group flex h-full w-full flex-col items-center rounded-[var(--radius-panel)] border-2 border-[var(--color-border-soft)] bg-[var(--color-surface)] p-5 text-center shadow-[0_4px_0_0_rgba(0,0,0,0.06)] transition hover:-translate-y-1 hover:border-[var(--color-primary)] hover:shadow-[0_10px_24px_-10px_rgba(0,0,0,0.25)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-primary)]"
              >
                <span
                  className="grid h-16 w-16 place-items-center rounded-2xl text-3xl transition group-hover:scale-110"
                  style={{ background: path.tint }}
                  aria-hidden="true"
                >
                  <Glyph value={path.icon} />
                </span>
                <span className="mt-3 text-xs font-extrabold uppercase tracking-[0.12em]" style={{ color: path.accent }}>
                  {path.tagline}
                </span>
                <span className="mt-1 font-display text-xl font-bold text-[var(--color-ink)]">{path.title}</span>
                <span className="mt-2 text-sm leading-relaxed text-[var(--color-ink-soft)]">{path.blurb}</span>
                {path.id === "alphabet" && (
                  <span className="mt-3 rounded-full bg-[var(--color-primary-soft)] px-3 py-1 text-xs font-bold text-[var(--color-primary-dim)]">
                    {lettersMastered} / 26 letters mastered
                  </span>
                )}
                <span className="mt-auto pt-4 text-sm font-bold text-[var(--color-primary)] group-hover:underline">
                  Let&rsquo;s go →
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </Dialog>
  );
}
