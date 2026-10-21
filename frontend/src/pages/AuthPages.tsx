import { type FormEvent, type ReactNode, useId, useState } from "react";
import { PrimaryButton } from "../components/AppShell";
import { useAuth } from "../auth/AuthContext";
import { Glyph } from "../components/Icon";

function PasswordField({
  label,
  value,
  onChange,
  autoComplete,
  minLength,
  mismatch = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  autoComplete: string;
  minLength?: number;
  mismatch?: boolean;
}) {
  const id = useId();
  const [visible, setVisible] = useState(false);
  return (
    <div className="space-y-2 text-sm">
      <label htmlFor={id} className="block font-medium text-[var(--color-ink-soft)]">
        {label}
      </label>
      <div className="relative">
        <input
          id={id}
          type={visible ? "text" : "password"}
          className="input-field pr-20"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          autoComplete={autoComplete}
          minLength={minLength}
          aria-invalid={mismatch || undefined}
          aria-describedby={mismatch ? `${id}-error` : undefined}
          required
        />
        <button
          type="button"
          className="absolute inset-y-0 right-2 my-auto h-9 rounded-lg px-3 text-xs font-bold text-[var(--color-primary-dim)] hover:bg-[var(--color-primary-soft)]"
          onClick={() => setVisible((v) => !v)}
          aria-pressed={visible}
          aria-label={`${visible ? "Hide" : "Show"} ${label.toLowerCase()}`}
        >
          {visible ? "Hide" : "Show"}
        </button>
      </div>
      {mismatch && (
        <p id={`${id}-error`} className="text-xs font-semibold text-[var(--color-danger)]">
          Passwords do not match.
        </p>
      )}
    </div>
  );
}

function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="relative hidden flex-col justify-between overflow-hidden bg-[var(--color-primary)] p-10 text-white lg:flex">
        <div className="pointer-events-none absolute -right-16 -top-20 h-72 w-72 rounded-full bg-white/10" aria-hidden="true" />
        <div className="pointer-events-none absolute -bottom-24 -left-10 h-64 w-64 rounded-full bg-white/5" aria-hidden="true" />
        <div className="relative flex items-center gap-2.5">
          <span className="grid h-10 w-10 place-items-center rounded-2xl bg-white/15 text-lg" aria-hidden="true">
            <Glyph value="🤟" />
          </span>
          <span className="font-display text-lg font-semibold">ASL Quest</span>
        </div>
        <div className="relative">
          <p className="font-display text-4xl font-semibold leading-tight">
            Learn ASL. Build confidence.
            <br />
            One sign at a time.
          </p>
          <p className="mt-4 max-w-sm text-white/80">
            Practice the alphabet, spell real words, and learn complete native ASL signs — with instant feedback
            from your camera.
          </p>
          <div className="mt-8 flex flex-wrap gap-3 text-sm">
            <span className="rounded-full bg-white/15 px-4 py-2">A–Z recognition</span>
            <span className="rounded-full bg-white/15 px-4 py-2">Word spelling</span>
            <span className="rounded-full bg-white/15 px-4 py-2">Native signs</span>
          </div>
        </div>
        <p className="relative text-xs text-white/60">© ASL Quest — a learning game for American Sign Language.</p>
      </div>

      <div className="flex items-center justify-center px-4 py-10">
        <div className="w-full max-w-sm">
          <div className="mb-8 flex items-center gap-2.5 lg:hidden">
            <span className="grid h-10 w-10 place-items-center rounded-2xl bg-[var(--color-primary)] text-lg text-white" aria-hidden="true">
              <Glyph value="🤟" />
            </span>
            <span className="font-display text-lg font-semibold text-[var(--color-ink)]">ASL Quest</span>
          </div>
          {children}
        </div>
      </div>
    </div>
  );
}

export function LoginPage() {
  const { login, setAuthView } = useAuth();
  const [loginValue, setLoginValue] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(loginValue.trim(), password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AuthLayout>
      <h1 className="font-display text-3xl font-semibold text-[var(--color-ink)]">Welcome back</h1>
      <p className="mt-2 text-sm text-[var(--color-ink-soft)]">Sign in to sync your XP, streaks, and mastery.</p>

      <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
        <label className="block space-y-2 text-sm">
          <span className="font-medium text-[var(--color-ink-soft)]">Email or username</span>
          <input
            className="input-field"
            value={loginValue}
            onChange={(event) => setLoginValue(event.target.value)}
            autoComplete="username"
            required
          />
        </label>

        <PasswordField
          label="Password"
          value={password}
          onChange={setPassword}
          autoComplete="current-password"
        />

        {error && (
          <p role="alert" className="rounded-xl bg-[var(--color-danger-soft)] px-3 py-2 text-sm font-semibold text-[var(--color-danger)]">
            {error}
          </p>
        )}

        <PrimaryButton className="w-full" disabled={submitting} type="submit">
          {submitting ? "Signing in..." : "Continue →"}
        </PrimaryButton>
      </form>

      <p className="mt-6 text-center text-sm text-[var(--color-ink-soft)]">
        New here?{" "}
        <button
          type="button"
          className="font-semibold text-[var(--color-primary)] hover:underline"
          onClick={() => setAuthView("register")}
        >
          Create account
        </button>
      </p>
    </AuthLayout>
  );
}

export function RegisterPage() {
  const { register, setAuthView } = useAuth();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await register({ username: username.trim(), email: email.trim(), password });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AuthLayout>
      <h1 className="font-display text-3xl font-semibold text-[var(--color-ink)]">Start your journey</h1>
      <p className="mt-2 text-sm text-[var(--color-ink-soft)]">
        Create an account to save progress, streaks, and achievements.
      </p>

      <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
        <label className="block space-y-2 text-sm">
          <span className="font-medium text-[var(--color-ink-soft)]">Username</span>
          <input
            className="input-field"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            minLength={3}
            required
          />
        </label>

        <label className="block space-y-2 text-sm">
          <span className="font-medium text-[var(--color-ink-soft)]">Email</span>
          <input
            type="email"
            className="input-field"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="email"
            required
          />
        </label>

        <PasswordField
          label="Password"
          value={password}
          onChange={setPassword}
          autoComplete="new-password"
          minLength={8}
        />

        <PasswordField
          label="Confirm password"
          value={confirmPassword}
          onChange={setConfirmPassword}
          autoComplete="new-password"
          minLength={8}
          mismatch={confirmPassword.length > 0 && confirmPassword !== password}
        />

        {error && (
          <p role="alert" className="rounded-xl bg-[var(--color-danger-soft)] px-3 py-2 text-sm font-semibold text-[var(--color-danger)]">
            {error}
          </p>
        )}

        <PrimaryButton className="w-full" disabled={submitting} type="submit">
          {submitting ? "Creating account..." : "Create account →"}
        </PrimaryButton>
      </form>

      <p className="mt-6 text-center text-sm text-[var(--color-ink-soft)]">
        Already have an account?{" "}
        <button
          type="button"
          className="font-semibold text-[var(--color-primary)] hover:underline"
          onClick={() => setAuthView("login")}
        >
          Login
        </button>
      </p>
    </AuthLayout>
  );
}
