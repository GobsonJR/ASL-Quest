import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  fetchMigrationOffer,
  loginUser,
  logoutUser,
  migrateLocalProgress,
  registerUser,
  restoreSession,
  type SyncStatus,
  type UserProfile,
} from "./api";
import { clearToken } from "./token";
import { loadGameState, saveGameState } from "../game/storage";
import type { GameState } from "../game/types";

type AuthView = "login" | "register";

type AuthContextValue = {
  user: UserProfile | null;
  loading: boolean;
  authView: AuthView;
  syncStatus: SyncStatus;
  migrationOffer: boolean;
  setAuthView: (view: AuthView) => void;
  login: (login: string, password: string) => Promise<void>;
  register: (input: { username: string; email: string; password: string }) => Promise<void>;
  logout: () => void;
  dismissMigration: () => void;
  importLocalProgress: () => Promise<GameState | null>;
  setSyncStatus: (status: SyncStatus) => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

function localProgressIsMeaningful(state: GameState): boolean {
  if (state.xp > 0 || state.totalCorrect > 0 || state.sessions > 0) return true;
  return Object.values(state.letterStats).some((stats) => stats.attempts > 0 || stats.correct > 0);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [authView, setAuthView] = useState<AuthView>("login");
  const [syncStatus, setSyncStatus] = useState<SyncStatus>("idle");
  const [migrationOffer, setMigrationOffer] = useState(false);

  useEffect(() => {
    restoreSession()
      .then(async (profile) => {
        setUser(profile);
        if (profile) {
          try {
            const offer = await fetchMigrationOffer();
            const local = loadGameState();
            setMigrationOffer(offer.should_offer && localProgressIsMeaningful(local));
          } catch {
            setSyncStatus("offline");
          }
        }
      })
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (loginValue: string, password: string) => {
    const profile = await loginUser(loginValue, password);
    setUser(profile);
    try {
      const offer = await fetchMigrationOffer();
      const local = loadGameState();
      setMigrationOffer(offer.should_offer && localProgressIsMeaningful(local));
    } catch {
      setMigrationOffer(false);
    }
  }, []);

  const register = useCallback(async (input: { username: string; email: string; password: string }) => {
    const profile = await registerUser(input);
    setUser(profile);
    const local = loadGameState();
    setMigrationOffer(localProgressIsMeaningful(local));
  }, []);

  const logout = useCallback(() => {
    logoutUser();
    clearToken();
    setUser(null);
    setMigrationOffer(false);
    setSyncStatus("idle");
    setAuthView("login");
  }, []);

  const dismissMigration = useCallback(() => setMigrationOffer(false), []);

  const importLocalProgress = useCallback(async () => {
    const local = loadGameState();
    const imported = await migrateLocalProgress(local);
    saveGameState(imported);
    setMigrationOffer(false);
    return imported;
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      authView,
      syncStatus,
      migrationOffer,
      setAuthView,
      login,
      register,
      logout,
      dismissMigration,
      importLocalProgress,
      setSyncStatus,
    }),
    [user, loading, authView, syncStatus, migrationOffer, login, register, logout, dismissMigration, importLocalProgress]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
}
