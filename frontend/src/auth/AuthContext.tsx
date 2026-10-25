import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
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
import { DEMO_MODE } from "../config";
import type { NavPage } from "../game/constants";
import { loadGameState, saveGameState } from "../game/storage";
import type { GameState } from "../game/types";

type AuthView = "login" | "register";

type AuthContextValue = {
  user: UserProfile | null;
  loading: boolean;
  authView: AuthView;
  syncStatus: SyncStatus;
  migrationOffer: boolean;
  /** Page to open right after a fresh sign-in (the Admin workspace for an administrator); consumed once. */
  landingPage: NavPage | null;
  clearLandingPage: () => void;
  setAuthView: (view: AuthView) => void;
  login: (login: string, password: string) => Promise<void>;
  register: (input: { username: string; email: string; password: string }) => Promise<void>;
  logout: () => void;
  dismissMigration: () => void;
  importLocalProgress: () => Promise<GameState | null>;
  setSyncStatus: (status: SyncStatus) => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

// Only used when DEMO_MODE is on (frontend-only deployment). Not a real account.
const DEMO_USER: UserProfile = {
  id: 0,
  username: "Demo learner",
  email: "demo@local",
  role: "user",
  created_at: new Date(0).toISOString(),
};

function localProgressIsMeaningful(state: GameState): boolean {
  if (state.xp > 0 || state.totalCorrect > 0 || state.sessions > 0) return true;
  return Object.values(state.letterStats).some((stats) => stats.attempts > 0 || stats.correct > 0);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(DEMO_MODE ? DEMO_USER : null);
  const [loading, setLoading] = useState(!DEMO_MODE);
  const [authView, setAuthView] = useState<AuthView>("login");
  const [syncStatus, setSyncStatus] = useState<SyncStatus>("idle");
  const [migrationOffer, setMigrationOffer] = useState(false);
  const [landingPage, setLandingPage] = useState<NavPage | null>(null);
  const sessionCounter = useRef(0);

  useEffect(() => {
    if (DEMO_MODE) return;
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
    // The role comes from the backend (/auth/me, read from the account row), never from anything typed into the form.
    setLandingPage(profile.role === "admin" ? "admin" : null);
    setMigrationOffer(false);
    // The import offer is a courtesy, not part of signing in: it runs after the user is in, so a slow server can never
    // keep the form on "Signing in...". The result is dropped if the user signed out or in again in the meantime.
    const session = ++sessionCounter.current;
    fetchMigrationOffer()
      .then((offer) => {
        if (session === sessionCounter.current) setMigrationOffer(offer.should_offer && localProgressIsMeaningful(loadGameState()));
      })
      .catch(() => {
        if (session === sessionCounter.current) setMigrationOffer(false);
      });
  }, []);

  const register = useCallback(async (input: { username: string; email: string; password: string }) => {
    const profile = await registerUser(input);
    setUser(profile);
    const local = loadGameState();
    setMigrationOffer(localProgressIsMeaningful(local));
  }, []);

  const logout = useCallback(() => {
    if (DEMO_MODE) return; // no session to end in demo mode
    sessionCounter.current++; // invalidates any sign-in follow-up still in flight
    logoutUser();
    clearToken();
    setUser(null);
    setLandingPage(null);
    setMigrationOffer(false);
    setSyncStatus("idle");
    setAuthView("login");
  }, []);

  const dismissMigration = useCallback(() => setMigrationOffer(false), []);
  const clearLandingPage = useCallback(() => setLandingPage(null), []);

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
      landingPage,
      clearLandingPage,
      setAuthView,
      login,
      register,
      logout,
      dismissMigration,
      importLocalProgress,
      setSyncStatus,
    }),
    [user, loading, authView, syncStatus, migrationOffer, landingPage, clearLandingPage, login, register, logout, dismissMigration, importLocalProgress]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
}
