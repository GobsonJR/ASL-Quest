import type { ReactNode } from "react";

// Stroke icons (24px grid, currentColor) so the UI never depends on the
// platform's emoji font. `Glyph` lets existing data that stores an emoji
// character (badge/challenge/stat `icon` fields) render as an SVG instead.

const circle = (cx: number, cy: number, r: number) => <circle key={`${cx}-${cy}-${r}`} cx={cx} cy={cy} r={r} />;

const ICONS: Record<string, ReactNode> = {
  book: (
    <>
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </>
  ),
  hand: (
    <>
      <path d="M18 11V6a2 2 0 0 0-4 0" />
      <path d="M14 10V4a2 2 0 0 0-4 0v2" />
      <path d="M10 10.5V6a2 2 0 0 0-4 0v8" />
      <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-6-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
    </>
  ),
  letters: (
    <>
      <path d="M4 7V4h16v3" />
      <path d="M9 20h6" />
      <path d="M12 4v16" />
    </>
  ),
  grid: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  award: (
    <>
      {circle(12, 8, 6)}
      <path d="M15.5 12.9 17 22l-5-3-5 3 1.5-9.1" />
    </>
  ),
  user: (
    <>
      {circle(12, 8, 4)}
      <path d="M4 21a8 8 0 0 1 16 0" />
    </>
  ),
  home: (
    <>
      <path d="m3 11 9-8 9 8" />
      <path d="M5 10v10h14V10" />
    </>
  ),
  video: (
    <>
      <rect x="2" y="6" width="14" height="12" rx="2" />
      <path d="m22 8-6 4 6 4z" />
    </>
  ),
  target: (
    <>
      {circle(12, 12, 10)}
      {circle(12, 12, 6)}
      {circle(12, 12, 2)}
    </>
  ),
  "chart-up": (
    <>
      <path d="m3 17 6-6 4 4 8-8" />
      <path d="M14 7h7v7" />
    </>
  ),
  "chart-bar": (
    <>
      <path d="M18 20V10" />
      <path d="M12 20V4" />
      <path d="M6 20v-6" />
    </>
  ),
  sparkles: (
    <>
      <path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z" />
      <path d="M19 3v4M17 5h4" />
    </>
  ),
  sliders: <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" />,
  wrench: (
    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94z" />
  ),
  lock: (
    <>
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </>
  ),
  warning: (
    <>
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      <path d="M12 9v4M12 17h.01" />
    </>
  ),
  users: (
    <>
      {circle(9, 7, 4)}
      <path d="M2 21v-2a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v2" />
      <path d="M16 3.1a4 4 0 0 1 0 7.8M22 21v-2a4 4 0 0 0-3-3.9" />
    </>
  ),
  zap: <path d="M13 2 3 14h9l-1 8 10-12h-9z" />,
  pencil: <path d="M17 3a2.8 2.8 0 0 1 4 4L7.5 20.5 2 22l1.5-5.5z" />,
  star: <path d="m12 2 3.1 6.3 6.9 1-5 4.8 1.2 6.9-6.2-3.2-6.2 3.2L7 14.1 2 9.3l6.9-1z" />,
  flame: (
    <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.4-.5-2-1-3-1.1-2.3-.3-4.1 1-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.2.4-2.3 1-3 0 1.4 1 2.5 2.5 2.5z" />
  ),
  camera: (
    <>
      <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z" />
      {circle(12, 13, 3)}
    </>
  ),
  check: <path d="M20 6 9 17l-5-5" />,
  x: <path d="M18 6 6 18M6 6l12 12" />,
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  trophy: (
    <>
      <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6M18 9h1.5a2.5 2.5 0 0 0 0-5H18M4 22h16" />
      <path d="M10 14.7V17c0 .6-.5 1-1 1.2C7.8 18.8 7 20.2 7 22M14 14.7V17c0 .6.5 1 1 1.2 1.2.6 2 2 2 3.8" />
      <path d="M18 2H6v7a6 6 0 0 0 12 0z" />
    </>
  ),
  compass: (
    <>
      {circle(12, 12, 10)}
      <path d="m16.2 7.8-2.1 6.3-6.3 2.1 2.1-6.3z" />
    </>
  ),
  wind: (
    <path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2M9.6 4.6A2 2 0 1 1 11 8H2M12.6 19.4A2 2 0 1 0 14 16H2" />
  ),
  globe: (
    <>
      {circle(12, 12, 10)}
      <path d="M2 12h20M12 2a15 15 0 0 1 4 10 15 15 0 0 1-4 10 15 15 0 0 1-4-10 15 15 0 0 1 4-10z" />
    </>
  ),
  crown: <path d="m2 4 3 12h14l3-12-6 7-4-7-4 7zM5 20h14" />,
  lightbulb: (
    <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5M9 18h6M10 22h4" />
  ),
  map: <path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3zM9 3v15M15 6v15" />,
  gem: <path d="M6 3h12l4 6-10 13L2 9zM11 3 8 9l4 13 4-13-3-6M2 9h20" />,
  sun: (
    <>
      {circle(12, 12, 4)}
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  "thumbs-up": (
    <path d="M7 10v12M15 5.9 14 10h5.8a2 2 0 0 1 1.9 2.5l-2.3 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.8a2 2 0 0 0 1.8-1.1L12 2a3.1 3.1 0 0 1 3 3.9z" />
  ),
  clock: (
    <>
      {circle(12, 12, 10)}
      <path d="M12 6v6l4 2" />
    </>
  ),
  calendar: (
    <>
      <rect x="3" y="4" width="18" height="18" rx="2" />
      <path d="M16 2v4M8 2v4M3 10h18" />
    </>
  ),
  play: <path d="m6 3 14 9-14 9z" />,
  leaf: (
    <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10zM2 21c0-3 1.9-5.4 5.6-6.4" />
  ),
};

// Emoji characters already present in data (badges, challenges, stat tiles)
// -> icon name. Variation selectors are stripped before lookup.
const EMOJI_TO_ICON: Record<string, string> = {
  "📚": "book",
  "✋": "hand",
  "🤟": "hand",
  "👋": "hand",
  "🔤": "letters",
  "🧩": "grid",
  "🏅": "award",
  "🙂": "user",
  "👤": "user",
  "🏠": "home",
  "🎥": "video",
  "🎯": "target",
  "📈": "chart-up",
  "📊": "chart-bar",
  "✨": "sparkles",
  "🎉": "sparkles",
  "🎊": "sparkles",
  "⚙": "sliders",
  "🛠": "wrench",
  "🔒": "lock",
  "⚠": "warning",
  "👥": "users",
  "⚡": "zap",
  "📝": "pencil",
  "⭐": "star",
  "🔥": "flame",
  "📷": "camera",
  "✅": "check",
  "✓": "check",
  "🏆": "trophy",
  "🧭": "compass",
  "💨": "wind",
  "🌍": "globe",
  "👑": "crown",
  "🧠": "lightbulb",
  "💡": "lightbulb",
  "🗺": "map",
  "💎": "gem",
  "☀": "sun",
  "👍": "thumbs-up",
  "⏱": "clock",
  "📅": "calendar",
  "🗓": "calendar",
  "🎮": "play",
  "🌱": "leaf",
  "🚀": "trophy",
};

const BASE_PROPS = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
  focusable: false,
} as const;

export function Icon({ name, className = "", flip = false }: { name: string; className?: string; flip?: boolean }) {
  const body = ICONS[name];
  if (!body) return null;
  return (
    <svg {...BASE_PROPS} className={`inline-block h-[1.15em] w-[1.15em] shrink-0 ${className}`} style={flip ? { transform: "scaleY(-1)" } : undefined}>
      {body}
    </svg>
  );
}

/** Renders a stored emoji character (or a plain icon name) as an SVG icon. */
export function Glyph({ value, className = "" }: { value: string; className?: string }) {
  const key = value.replace(/️/g, "").trim();
  if (key === "👎") return <Icon name="thumbs-up" className={className} flip />;
  return <Icon name={EMOJI_TO_ICON[key] ?? key} className={className} />;
}
