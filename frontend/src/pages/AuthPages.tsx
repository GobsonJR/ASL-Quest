import { type FormEvent, useState } from "react";
import { PageHeader, Panel, PrimaryButton } from "../components/AppShell";
import { useAuth } from "../auth/AuthContext";

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
    <div className="mx-auto max-w-md px-4 py-10">
      <Panel className="space-y-6">
        <PageHeader
          eyebrow="ASL Quest"
          title="Welcome back"
          description="Sign in to sync your XP, streaks, and mastery."
          compact
        />

        <form className="space-y-4" onSubmit={handleSubmit}>
          <label className="block space-y-2 text-sm">
            <span className="text-[var(--color-mist)]">Email or username</span>
            <input
              className="input-field"
              value={loginValue}
              onChange={(event) => setLoginValue(event.target.value)}
              autoComplete="username"
              required
            />
          </label>

          <label className="block space-y-2 text-sm">
            <span className="text-[var(--color-mist)]">Password</span>
            <input
              type="password"
              className="input-field"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />
          </label>

          {error && <p className="text-sm text-[var(--color-warm)]">{error}</p>}

          <PrimaryButton className="w-full" disabled={submitting} type="submit">
            {submitting ? "Signing in..." : "Login"}
          </PrimaryButton>
        </form>

        <p className="text-center text-sm text-[var(--color-mist)]">
          Don&apos;t have an account?{" "}
          <button
            type="button"
            className="font-semibold text-[var(--color-accent)] hover:underline"
            onClick={() => setAuthView("register")}
          >
            Create one
          </button>
        </p>
      </Panel>
    </div>
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
    <div className="mx-auto max-w-md px-4 py-10">
      <Panel className="space-y-6">
        <PageHeader
          eyebrow="ASL Quest"
          title="Start your journey"
          description="Create an account to save progress, streaks, and achievements."
          compact
        />

        <form className="space-y-4" onSubmit={handleSubmit}>
          <label className="block space-y-2 text-sm">
            <span className="text-[var(--color-mist)]">Username</span>
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
            <span className="text-[var(--color-mist)]">Email</span>
            <input
              type="email"
              className="input-field"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              required
            />
          </label>

          <label className="block space-y-2 text-sm">
            <span className="text-[var(--color-mist)]">Password</span>
            <input
              type="password"
              className="input-field"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="new-password"
              minLength={8}
              required
            />
          </label>

          <label className="block space-y-2 text-sm">
            <span className="text-[var(--color-mist)]">Confirm password</span>
            <input
              type="password"
              className="input-field"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              autoComplete="new-password"
              minLength={8}
              required
            />
          </label>

          {error && <p className="text-sm text-[var(--color-warm)]">{error}</p>}

          <PrimaryButton className="w-full" disabled={submitting} type="submit">
            {submitting ? "Creating account..." : "Create account"}
          </PrimaryButton>
        </form>

        <p className="text-center text-sm text-[var(--color-mist)]">
          Already have an account?{" "}
          <button
            type="button"
            className="font-semibold text-[var(--color-accent)] hover:underline"
            onClick={() => setAuthView("login")}
          >
            Login
          </button>
        </p>
      </Panel>
    </div>
  );
}
