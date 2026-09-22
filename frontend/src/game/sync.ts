import { saveProgress, recordPracticeSession, type SyncStatus } from "../auth/api";
import { saveGameState } from "./storage";
import type { GameState } from "./types";

let syncTimer: number | null = null;
let pendingState: GameState | null = null;

export function scheduleProgressSync(
  state: GameState,
  setSyncStatus: (status: SyncStatus) => void,
  delayMs = 900
): void {
  pendingState = state;
  saveGameState(state);
  if (syncTimer) window.clearTimeout(syncTimer);
  syncTimer = window.setTimeout(async () => {
    if (!pendingState) return;
    const payload = pendingState;
    pendingState = null;
    setSyncStatus("syncing");
    try {
      const saved = await saveProgress(payload);
      saveGameState(saved);
      setSyncStatus("synced");
    } catch {
      setSyncStatus("offline");
    }
  }, delayMs);
}

export async function flushProgressSync(
  state: GameState,
  setSyncStatus: (status: SyncStatus) => void
): Promise<GameState | null> {
  if (syncTimer) {
    window.clearTimeout(syncTimer);
    syncTimer = null;
  }
  pendingState = null;
  setSyncStatus("syncing");
  try {
    const saved = await saveProgress(state);
    saveGameState(saved);
    setSyncStatus("synced");
    return saved;
  } catch {
    setSyncStatus("offline");
    saveGameState(state);
    return null;
  }
}

export async function logPracticeAttempt(input: {
  letter: string;
  prediction: string | null;
  correct: boolean;
  response_time?: number;
  xp_earned?: number;
  challenge_type?: string | null;
  reason?: string | null;
}): Promise<void> {
  try {
    await recordPracticeSession(input);
  } catch {
    /* offline fallback: aggregate progress sync handles state */
  }
}
