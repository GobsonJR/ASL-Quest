import type { SignoAccessory, SignoMood, SignoReaction } from "./types";
import "./signo.css";

const INK = "#263238";

/** SIGNO: an original sprout-headed creature built from layered SVG parts. Every
 * expression is a set of face parts that crossfade (CSS opacity/transform
 * transitions keyed off `data-mood`), so mood changes animate instead of snapping.
 * Pure presentation: it knows nothing about pages, lessons or the AI service. */
export function SignoCharacter({
  mood = "idle",
  size = 96,
  accessories = [],
  reaction = null,
  pulse = 0,
  effects = true,
  label,
  className = "",
}: {
  mood?: SignoMood;
  size?: number;
  accessories?: SignoAccessory[];
  reaction?: SignoReaction | null;
  /** Bump to replay the one-shot animation for the current mood. */
  pulse?: number;
  /** When false, all animation is disabled (the face still changes). */
  effects?: boolean;
  /** Accessible name. Omit when the character is purely decorative. */
  label?: string;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 120 120"
      width={size}
      height={size}
      className={`signo ${className}`}
      data-mood={mood}
      data-reaction={reaction ?? undefined}
      data-effects={effects ? "on" : "off"}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
    >
      <ellipse cx="60" cy="112" rx="27" ry="4" fill={INK} opacity="0.12" />

      <g className="signo-tilt">
        {/* key={pulse} remounts the body so one-shot animations (hop, wiggle) replay on each event */}
        <g className="signo-hop" key={pulse}>
          <g className="signo-breathe">
            {/* feet */}
            <ellipse cx="46" cy="104" rx="10" ry="5.5" fill="#6cc10f" />
            <ellipse cx="74" cy="104" rx="10" ry="5.5" fill="#6cc10f" />

            {/* arms (behind the body edge so they read as tucked mittens) */}
            <g className="signo-arm signo-arm-l">
              <ellipse cx="20" cy="76" rx="7.5" ry="13" fill="#86d63a" />
            </g>
            <g className="signo-arm signo-arm-r">
              <ellipse cx="100" cy="76" rx="7.5" ry="13" fill="#86d63a" />
            </g>

            {/* sprout leaves */}
            <g className="signo-leaf">
              <path d="M60 30 C52 12 38 8 31 13 C35 25 47 32 60 30Z" fill="#58cc02" />
              <path d="M60 30 C68 12 82 8 89 13 C85 25 73 32 60 30Z" fill="#46a302" />
              <path d="M60 31 C52 22 45 17 36 14" stroke="#9be548" strokeWidth="1.6" strokeLinecap="round" fill="none" />
              <path d="M60 31 C68 22 75 17 84 14" stroke="#7ee035" strokeWidth="1.6" strokeLinecap="round" fill="none" />
            </g>

            {/* body + belly */}
            <ellipse cx="60" cy="68" rx="39" ry="37" fill="#9be548" />
            <ellipse cx="60" cy="80" rx="24" ry="21" fill="#e4fadb" />

            {/* cheeks */}
            <ellipse cx="34" cy="73" rx="5.5" ry="3.4" fill="#ffb7ab" opacity="0.9" />
            <ellipse cx="86" cy="73" rx="5.5" ry="3.4" fill="#ffb7ab" opacity="0.9" />

            {/* ---- eyes: three variants crossfade ---- */}
            <g className="signo-eyes-open">
              <g className="signo-eyes">
                <ellipse cx="46" cy="58" rx="9.5" ry="11.5" fill="#fff" />
                <ellipse cx="74" cy="58" rx="9.5" ry="11.5" fill="#fff" />
                <g className="signo-pupils">
                  <circle cx="47" cy="60" r="5.6" fill={INK} />
                  <circle cx="75" cy="60" r="5.6" fill={INK} />
                  <circle cx="49" cy="57.5" r="1.9" fill="#fff" />
                  <circle cx="77" cy="57.5" r="1.9" fill="#fff" />
                </g>
              </g>
            </g>
            <g className="signo-eyes-happy">
              <path d="M37 61 Q46 49 55 61" stroke={INK} strokeWidth="3.4" strokeLinecap="round" fill="none" />
              <path d="M65 61 Q74 49 83 61" stroke={INK} strokeWidth="3.4" strokeLinecap="round" fill="none" />
            </g>
            <g className="signo-eyes-closed">
              <path d="M37 59 Q46 66 55 59" stroke={INK} strokeWidth="3.2" strokeLinecap="round" fill="none" />
              <path d="M65 59 Q74 66 83 59" stroke={INK} strokeWidth="3.2" strokeLinecap="round" fill="none" />
            </g>

            {/* brows: encouraging (both lifted, soft) and curious (one raised) */}
            <path className="signo-brow signo-brow-l" d="M37 44 Q46 39 55 45" stroke={INK} strokeWidth="2.8" strokeLinecap="round" fill="none" />
            <path className="signo-brow signo-brow-r" d="M65 45 Q74 39 83 44" stroke={INK} strokeWidth="2.8" strokeLinecap="round" fill="none" />

            {/* ---- mouths: crossfade ---- */}
            <path className="signo-mouth signo-mouth-smile" d="M52 76 Q60 84 68 76" stroke={INK} strokeWidth="3" strokeLinecap="round" fill="none" />
            <g className="signo-mouth signo-mouth-open">
              <path d="M50 73 Q60 93 70 73 Z" fill={INK} />
              <ellipse cx="60" cy="83.5" rx="5.2" ry="3" fill="#ff8f80" />
            </g>
            <ellipse className="signo-mouth signo-mouth-o" cx="60" cy="79" rx="3.6" ry="4.2" fill={INK} />
            <path className="signo-mouth signo-mouth-soft" d="M53 78 Q60 81.5 67 78" stroke={INK} strokeWidth="3" strokeLinecap="round" fill="none" />
            <path className="signo-mouth signo-mouth-flat" d="M55 78 Q60 79.5 65 78" stroke={INK} strokeWidth="2.6" strokeLinecap="round" fill="none" />

            {/* ---- accessories (cosmetic, unlocked by real badges) ---- */}
            {accessories.includes("glasses") && (
              <g fill="none" stroke={INK} strokeWidth="2.4">
                <circle cx="46" cy="58" r="14" />
                <circle cx="74" cy="58" r="14" />
                <path d="M60 57 h0" strokeLinecap="round" />
                <path d="M59.5 56 Q60 54.5 60.5 56" />
              </g>
            )}
            {accessories.includes("headband") && (
              <g>
                <path d="M24 49 Q60 34 96 49" stroke="#ff6b57" strokeWidth="6" strokeLinecap="round" fill="none" />
                <path d="M92 48 l9 -5 l-2 10 z" fill="#ff6b57" />
              </g>
            )}
            {accessories.includes("crown") && (
              <path d="M47 24 L50 8 L56 17 L60 6 L64 17 L70 8 L73 24 Z" fill="#ffb020" stroke="#e08a00" strokeWidth="1.5" strokeLinejoin="round" />
            )}
          </g>
        </g>
      </g>

      {/* mood overlays */}
      <g className="signo-zzz" fill="none" stroke="#64748b" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path className="signo-z signo-z1" d="M88 34 h9 l-9 11 h9" />
        <path className="signo-z signo-z2" d="M100 20 h6 l-6 8 h6" />
      </g>
      <g className="signo-think" fill="#64748b">
        <circle className="signo-dot signo-dot1" cx="90" cy="30" r="3" />
        <circle className="signo-dot signo-dot2" cx="99" cy="23" r="3.6" />
        <circle className="signo-dot signo-dot3" cx="109" cy="15" r="4.2" />
      </g>
      <g className="signo-sparks" key={`s${pulse}`} fill="#ffb020">
        <path className="signo-spark signo-spark1" d="M14 30 l2.2 5.5 l5.5 2.2 l-5.5 2.2 l-2.2 5.5 l-2.2 -5.5 l-5.5 -2.2 l5.5 -2.2z" />
        <path className="signo-spark signo-spark2" d="M104 38 l1.8 4.4 l4.4 1.8 l-4.4 1.8 l-1.8 4.4 l-1.8 -4.4 l-4.4 -1.8 l4.4 -1.8z" fill="#58cc02" />
        <path className="signo-spark signo-spark3" d="M100 8 l1.4 3.4 l3.4 1.4 l-3.4 1.4 l-1.4 3.4 l-1.4 -3.4 l-3.4 -1.4 l3.4 -1.4z" />
        <path className="signo-spark signo-spark4" d="M12 66 l1.4 3.4 l3.4 1.4 l-3.4 1.4 l-1.4 3.4 l-1.4 -3.4 l-3.4 -1.4 l3.4 -1.4z" fill="#ff6b57" />
      </g>
    </svg>
  );
}
