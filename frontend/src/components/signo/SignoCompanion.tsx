import { useEffect, useId, useRef } from "react";
import { SignoCharacter } from "./SignoCharacter";
import { useSigno } from "./SignoContext";
import { SignoPanel } from "./SignoPanel";
import { SignoSpeechBubble } from "./SignoSpeechBubble";

/** The persistent companion: a floating launcher + speech bubble + expandable panel.
 * Positioned clear of the mobile bottom navigation (with safe-area inset) and the
 * desktop corner, and never takes focus unless the user opens it. */
export function SignoCompanion() {
  const {
    page, mood, pulse, reaction, accessories, prefs, bubble, hasTip, announcement, open,
    setOpen, dismissBubble, react,
  } = useSigno();
  const panelId = useId();
  const launcherRef = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);

  // Return focus to the launcher when the panel closes.
  useEffect(() => {
    if (wasOpen.current && !open) launcherRef.current?.focus();
    wasOpen.current = open;
  }, [open]);

  // Smaller on Practice so it stays clear of the camera and lesson controls.
  const compact = prefs.minimized || page === "practice";
  const size = compact ? 44 : 64;
  const showBubble = bubble !== null && !open && !prefs.minimized;
  const placement =
    "fixed right-3 bottom-[calc(4.75rem+env(safe-area-inset-bottom))] md:right-6 md:bottom-6";

  return (
    <div data-signo-effects={prefs.effects ? "on" : "off"} data-testid="signo-companion">
      {/* One polite live region for important messages (not the animated bubble). */}
      <p role="status" aria-live="polite" aria-atomic="true" className="sr-only">
        {announcement}
      </p>

      {/* On very wide screens the right-hand progress sidebar is visible: the panel docks to its left so it never covers it. */}
      {open && (
        <div className="signo-dock">
          <SignoPanel id={panelId} onClose={() => setOpen(false)} />
        </div>
      )}

      <div className={`${placement} z-40 flex flex-col items-end gap-2 ${open ? "max-md:hidden" : ""}`}>
        {showBubble && bubble && <SignoSpeechBubble bubble={bubble} onDismiss={dismissBubble} />}
        <button
          ref={launcherRef}
          type="button"
          onClick={() => {
            react();
            setOpen(!open);
          }}
          aria-label={hasTip ? "Open SIGNO, your ASL companion. SIGNO has a tip" : "Open SIGNO, your ASL companion"}
          aria-expanded={open}
          aria-controls={open ? panelId : undefined}
          className="relative grid place-items-center rounded-full border-2 border-[var(--color-primary)] bg-[var(--color-surface)] p-1 shadow-[0_3px_0_0_var(--color-primary-dim)] transition-transform active:translate-y-0.5"
          style={{ width: size + 12, height: size + 12 }}
        >
          {hasTip && !prefs.minimized && (
            <>
              <span
                aria-hidden="true"
                className="signo-tip-ring pointer-events-none absolute inset-0 rounded-full border-2 border-[var(--color-primary)]"
              />
              <span
                aria-hidden="true"
                className="absolute -right-0.5 -top-0.5 h-3.5 w-3.5 rounded-full border-2 border-[var(--color-surface)] bg-[var(--color-xp)]"
              />
            </>
          )}
          <SignoCharacter
            mood={mood}
            size={size}
            accessories={accessories}
            reaction={reaction}
            pulse={pulse}
            effects={prefs.effects}
          />
        </button>
      </div>
    </div>
  );
}
