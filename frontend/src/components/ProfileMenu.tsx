import { useEffect, useId, useRef, useState } from "react";
import type { NavPage } from "../game/constants";
import { Glyph } from "./Icon";

type MenuItem = { key: string; label: string; icon: string; onSelect: () => void; tone?: "danger" };

/** The signed-in person's avatar and account dropdown in the top-right of the header: Profile, Settings, the Admin workspace
 * (administrators only) and Log out. It reuses the existing pages and the existing logout handler; it owns no data. */
export function ProfileMenu({
  name,
  email,
  isAdmin,
  currentPage,
  onNavigate,
  onLogout,
}: {
  name: string;
  email?: string;
  isAdmin: boolean;
  currentPage: NavPage;
  onNavigate: (page: NavPage) => void;
  /** Omit to hide "Log out" (demo mode has no session to end). */
  onLogout?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuId = useId();
  const initial = (name.trim()[0] ?? "?").toUpperCase();

  const items: MenuItem[] = [
    { key: "profile", label: "Profile", icon: "🙂", onSelect: () => onNavigate("profile") },
    { key: "settings", label: "Settings", icon: "⚙️", onSelect: () => onNavigate("settings") },
    ...(isAdmin ? [{ key: "admin", label: "Admin workspace", icon: "🛠️", onSelect: () => onNavigate("admin") }] : []),
    ...(onLogout ? [{ key: "logout", label: "Log out", icon: "🚪", onSelect: onLogout, tone: "danger" as const }] : []),
  ];

  const close = (returnFocus: boolean) => {
    setOpen(false);
    if (returnFocus) buttonRef.current?.focus();
  };

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: MouseEvent | TouchEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        close(true);
      }
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("touchstart", onPointer);
    document.addEventListener("keydown", onKey);
    // move focus into the menu so it is keyboard-operable straight away
    rootRef.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus();
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("touchstart", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  function onMenuKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const nodes = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('[role="menuitem"]'));
    const index = nodes.indexOf(document.activeElement as HTMLElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      nodes[(index + 1) % nodes.length]?.focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      nodes[(index - 1 + nodes.length) % nodes.length]?.focus();
    } else if (event.key === "Home") {
      event.preventDefault();
      nodes[0]?.focus();
    } else if (event.key === "End") {
      event.preventDefault();
      nodes[nodes.length - 1]?.focus();
    } else if (event.key === "Tab") {
      setOpen(false);
    }
  }

  const inProfileArea = currentPage === "profile" || currentPage === "settings";

  return (
    <div ref={rootRef} className="relative">
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={`Account menu for ${name}`}
        className={`grid h-11 w-11 place-items-center rounded-full border-2 bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-native)] font-display text-lg font-bold text-white shadow-sm transition hover:scale-105 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-primary)] ${
          open || inProfileArea ? "border-[var(--color-primary-dim)]" : "border-white"
        }`}
      >
        <span aria-hidden="true">{initial}</span>
      </button>

      {open && (
        <div
          id={menuId}
          role="menu"
          aria-label="Account"
          onKeyDown={onMenuKeyDown}
          className="absolute right-0 top-[calc(100%+0.5rem)] z-50 w-64 origin-top-right animate-pop overflow-hidden rounded-2xl border-2 border-[var(--color-border)] bg-[var(--color-surface)] shadow-xl"
        >
          <div className="flex items-center gap-3 border-b border-[var(--color-border-soft)] bg-[var(--color-surface-soft)] px-4 py-3">
            <span
              aria-hidden="true"
              className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-native)] font-display text-base font-bold text-white"
            >
              {initial}
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-bold text-[var(--color-ink)]">{name}</p>
              {email && <p className="truncate text-xs text-[var(--color-muted)]">{email}</p>}
              <p className="mt-0.5 text-[0.65rem] font-extrabold uppercase tracking-[0.12em] text-[var(--color-primary-dim)]">
                {isAdmin ? "Administrator" : "Learner"}
              </p>
            </div>
          </div>
          <div className="p-1.5">
            {items.map((item) => (
              <button
                key={item.key}
                type="button"
                role="menuitem"
                onClick={() => {
                  close(false);
                  item.onSelect();
                }}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-primary)] ${
                  item.tone === "danger"
                    ? "text-[var(--color-danger)] hover:bg-[var(--color-danger-soft)]"
                    : "text-[var(--color-ink-soft)] hover:bg-[var(--color-surface-soft)] hover:text-[var(--color-ink)]"
                }`}
              >
                <span aria-hidden="true" className="text-base">
                  <Glyph value={item.icon} />
                </span>
                {item.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
