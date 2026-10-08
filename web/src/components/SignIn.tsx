// Sign-in / register gate.
//
// One screen with a mode toggle rather than two routes: the dashboard is a
// single page, and a full-page navigation to switch between "I have an account"
// and "I don't" is friction for no benefit.
"use client";

import { useState } from "react";
import { api, setToken, type User } from "@/lib/api";

export function SignIn({ onSignedIn }: { onSignedIn: (user: User) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (busy) return;
    setError(null);

    // Check client-side so an obvious mistake doesn't cost a round trip.
    if (!email.trim() || !password) {
      setError("Enter an email and password.");
      return;
    }
    if (mode === "register" && password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    setBusy(true);
    const res =
      mode === "login"
        ? await api.login(email.trim(), password)
        : await api.register(email.trim(), password, name.trim() || undefined);
    setBusy(false);

    if (res.data) {
      setToken(res.data.token);
      onSignedIn(res.data.user);
    } else {
      setError(res.error);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center px-6 py-12">
      <div className="w-full max-w-sm">
        <div className="flex flex-col items-center gap-3 pb-8">
          <Logo />
          <h1 className="text-2xl font-semibold tracking-tight">Arbiter</h1>
          <p className="text-center text-sm leading-relaxed text-muted">
            AI judgment for production ML.
            <br />
            Sign in to connect your models.
          </p>
        </div>

        <div className="rounded-lg border border-border bg-bg p-6 shadow-card">
          <div className="mb-5 flex rounded-md border border-border p-0.5">
            {(["login", "register"] as const).map((m) => (
              <button
                key={m}
                onClick={() => {
                  setMode(m);
                  setError(null);
                }}
                className={`flex-1 rounded px-3 py-1.5 text-sm font-medium transition-colors ${
                  mode === m ? "bg-fg text-bg" : "text-muted hover:text-fg"
                }`}
              >
                {m === "login" ? "Sign in" : "Create account"}
              </button>
            ))}
          </div>

          <div className="flex flex-col gap-3">
            {mode === "register" && (
              <Field
                label="Name"
                value={name}
                onChange={setName}
                placeholder="Optional"
                autoComplete="name"
                onEnter={submit}
              />
            )}
            <Field
              label="Email"
              type="email"
              value={email}
              onChange={setEmail}
              placeholder="you@example.com"
              autoComplete="email"
              onEnter={submit}
            />
            <Field
              label="Password"
              type="password"
              value={password}
              onChange={setPassword}
              placeholder={mode === "register" ? "At least 8 characters" : "••••••••"}
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              onEnter={submit}
            />

            {error && (
              <p className="rounded-md border border-[#dc262633] bg-[#dc26260a] px-3 py-2 text-xs leading-relaxed text-retrain">
                {error}
              </p>
            )}

            <button
              onClick={submit}
              disabled={busy}
              className="mt-1 w-full rounded-md border border-fg bg-fg px-3.5 py-2 text-sm font-medium text-bg transition-colors hover:bg-[#333] disabled:opacity-40"
            >
              {busy
                ? mode === "login"
                  ? "Signing in…"
                  : "Creating account…"
                : mode === "login"
                  ? "Sign in"
                  : "Create account"}
            </button>
          </div>
        </div>

        <p className="pt-5 text-center text-2xs leading-relaxed text-subtle">
          Passwords are hashed with PBKDF2-SHA256 and never stored in plain text.
          <br />
          Your telemetry is visible only to your account.
        </p>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  autoComplete,
  onEnter,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  autoComplete?: string;
  onEnter?: () => void;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-2xs font-medium uppercase tracking-wider text-subtle">{label}</span>
      <input
        type={type}
        value={value}
        autoComplete={autoComplete}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onEnter?.()}
        placeholder={placeholder}
        className="rounded-md border border-border bg-bg px-3 py-2 text-sm placeholder:text-subtle focus:border-borderStrong focus:outline-none"
      />
    </label>
  );
}

function Logo() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M12 2 2 20h20L12 2Z" fill="#000" />
    </svg>
  );
}
