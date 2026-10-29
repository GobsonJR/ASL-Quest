/**
 * Celebration audio feedback system using the Web Audio API.
 * Synthesizes cheerful, upbeat Duolingo-style chimes and streak celebrations
 * without requiring external media assets.
 */

const STORAGE_KEY = "asl_sound_effects_enabled";

export function isSoundEnabled(): boolean {
  if (typeof window === "undefined" || !window.localStorage) return true;
  const val = window.localStorage.getItem(STORAGE_KEY);
  return val === null ? true : val === "true";
}

export function setSoundEnabled(enabled: boolean): void {
  if (typeof window === "undefined" || !window.localStorage) return;
  window.localStorage.setItem(STORAGE_KEY, enabled ? "true" : "false");
  window.dispatchEvent(new CustomEvent("asl_sound_setting_changed", { detail: { enabled } }));
}

let audioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextClass) return null;
  if (!audioCtx || audioCtx.state === "closed") {
    try {
      audioCtx = new AudioContextClass();
    } catch {
      return null;
    }
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => undefined);
  }
  return audioCtx;
}

interface NoteOptions {
  freq: number;
  startTime: number;
  duration: number;
  type?: OscillatorType;
  volume?: number;
}

function playTone(ctx: AudioContext, { freq, startTime, duration, type = "sine", volume = 0.25 }: NoteOptions) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = type;
  osc.frequency.setValueAtTime(freq, startTime);

  gain.gain.setValueAtTime(0.0001, startTime);
  gain.gain.exponentialRampToValueAtTime(volume, startTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

  osc.connect(gain);
  gain.connect(ctx.destination);

  osc.start(startTime);
  osc.stop(startTime + duration + 0.05);
}

/**
 * Plays a cheerful Duolingo-style ascending celebration chime.
 * If streak > 1, adds an upbeat triumphant streak-fanfare shimmer.
 */
export function playCelebrationSound(options?: { streak?: number; leveledUp?: boolean; badgesUnlocked?: boolean }) {
  if (!isSoundEnabled()) return;
  const ctx = getAudioContext();
  if (!ctx) return;

  const now = ctx.currentTime;
  const isStreak = (options?.streak ?? 0) > 0;
  const isSuperCelebration = isStreak || options?.leveledUp || options?.badgesUnlocked;

  // Duolingo-style cheerful ascending major arpeggio:
  // C5 (523.25), E5 (659.25), G5 (783.99), C6 (1046.50)
  const notes = [
    { freq: 523.25, time: 0.0, dur: 0.12, type: "sine" as const, vol: 0.28 },
    { freq: 659.25, time: 0.08, dur: 0.14, type: "sine" as const, vol: 0.30 },
    { freq: 783.99, time: 0.16, dur: 0.16, type: "sine" as const, vol: 0.32 },
    { freq: 1046.50, time: 0.25, dur: 0.40, type: "triangle" as const, vol: 0.38 },
    // Shimmer harmonic overtone for the final high bell note
    { freq: 1318.51, time: 0.27, dur: 0.38, type: "sine" as const, vol: 0.20 },
  ];

  notes.forEach((n) => {
    playTone(ctx, {
      freq: n.freq,
      startTime: now + n.time,
      duration: n.dur,
      type: n.type,
      volume: n.vol,
    });
  });

  // If there's an active streak or level-up, append an ecstatic cheer/triumph chord
  if (isSuperCelebration) {
    const cheerStart = now + 0.38;
    // Triumphant burst chord: G5, C6, E6, G6
    const cheerNotes = [
      { freq: 783.99, dur: 0.45, vol: 0.22 },
      { freq: 1046.50, dur: 0.50, vol: 0.28 },
      { freq: 1318.51, dur: 0.55, vol: 0.25 },
      { freq: 1567.98, dur: 0.60, vol: 0.20 },
    ];
    cheerNotes.forEach((n) => {
      playTone(ctx, {
        freq: n.freq,
        startTime: cheerStart,
        duration: n.dur,
        type: "triangle",
        volume: n.vol,
      });
    });
  }
}
