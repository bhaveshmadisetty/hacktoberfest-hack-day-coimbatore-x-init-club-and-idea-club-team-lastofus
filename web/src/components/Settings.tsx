// Settings: API keys, the connect snippet, email and password changes.
//
// Both account changes require the current password. That is the point of the
// screen — a session token alone should not be enough to take over an account
// by changing the address password resets would go to.
"use client";

import { useCallback, useEffect, useState } from "react";
import {
  API_URL,
  type ApiKey,
  type CreatedKey,
  type User,
  api,
  relativeTime,
  setToken,
} from "@/lib/api";
import { Button, Card, CodeBlock, Skeleton } from "./ui";

export function Settings({
  user,
  onUserChange,
  onSignOut,
}: {
  user: User;
  onUserChange: (u: User) => void;
  onSignOut: () => void;
}) {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [fresh, setFresh] = useState<CreatedKey | null>(null);
  const [label, setLabel] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);

  const load = useCallback(async () => {
    const res = await api.keys();
    setKeys(res.data?.keys ?? []);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const copy = (text: string, what: string) => {
    navigator.clipboard?.writeText(text);
    setCopied(what);
    setTimeout(() => setCopied(null), 1800);
  };

  const create = async () => {
    setBusy(true);
    const res = await api.createKey(label.trim() || "default");
    if (res.data) {
      setFresh(res.data);
      setLabel("");
      load();
    }
    setBusy(false);
  };

  const revoke = async (id: number) => {
    setBusy(true);
    await api.revokeKey(id);
    await load();
    setBusy(false);
  };

  const snippet = (key: string) => `# Point the DriftGuard SDK at Arbiter — no code changes needed
export DRIFTGUARD_API_URL="${API_URL}"
export DRIFTGUARD_API_KEY="${key}"

# Your model code, unchanged
from driftguard import DriftGuard

dg = DriftGuard(model_id="my-model-v1", drift_threshold=0.15)
model = dg.wrap(my_model, feature_extractor=lambda x: x)

model.predict(features)      # telemetry streams to Arbiter

@dg.retrainer
def retrain():               # Arbiter decides whether this should run at all
    return train_new_model()`;

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-5">
      {/* ---- keys ---- */}
      <Card>
        <Header
          title="API keys"
          subtitle="Issued by Arbiter. You do not need a key from anyone else."
        />
        <div className="flex flex-col gap-4 px-5 py-4">
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && create()}
              placeholder="Label (e.g. prod-fraud-service)"
              maxLength={80}
              className="min-w-[220px] flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm placeholder:text-subtle focus:border-borderStrong focus:outline-none"
            />
            <Button onClick={create} disabled={busy}>
              Generate key
            </Button>
          </div>

          {fresh && (
            <div className="animate-fade-in rounded-md border border-fg bg-surface p-4">
              <div className="flex items-start justify-between gap-3 pb-2">
                <div>
                  <p className="text-sm font-medium">Your new key</p>
                  <p className="text-xs text-retrain">{fresh.warning}</p>
                </div>
                <Button size="sm" variant="secondary" onClick={() => copy(fresh.key, "key")}>
                  {copied === "key" ? "Copied" : "Copy"}
                </Button>
              </div>
              <code className="block break-all rounded border border-border bg-bg px-3 py-2 font-mono text-xs">
                {fresh.key}
              </code>
            </div>
          )}

          {loading ? (
            <Skeleton className="h-12 w-full" />
          ) : keys.length ? (
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-left text-2xs uppercase tracking-wider text-subtle">
                  <th className="py-2 font-medium">Key</th>
                  <th className="py-2 font-medium">Label</th>
                  <th className="py-2 font-medium">Last used</th>
                  <th className="py-2 text-right font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {keys.map((k) => (
                  <tr key={k.id} className="border-b border-border last:border-0">
                    <td className="py-2 font-mono text-muted">{k.key_hint}</td>
                    <td className="py-2 text-muted">{k.label}</td>
                    <td className="py-2 text-subtle">{relativeTime(k.last_used)}</td>
                    <td className="py-2 text-right">
                      {k.revoked ? (
                        <span className="text-subtle">revoked</span>
                      ) : (
                        <Button size="sm" variant="ghost" onClick={() => revoke(k.id)}>
                          Revoke
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-xs text-muted">
              No keys yet. Generate one to connect a model.
            </p>
          )}
        </div>
      </Card>

      {/* ---- connect snippet ---- */}
      <Card>
        <Header title="Connect a model" subtitle="Two environment variables" />
        <div className="px-5 py-4">
          <CodeBlock
            code={snippet(fresh?.key ?? "ak_live_your_key_here")}
            onCopy={() => copy(snippet(fresh?.key ?? "ak_live_your_key_here"), "snip")}
          />
          {copied === "snip" && (
            <p className="pt-2 text-2xs text-muted">Copied to clipboard.</p>
          )}
        </div>
      </Card>

      <ChangeEmail user={user} onUserChange={onUserChange} />
      <ChangePassword onSignOut={onSignOut} />

      <Card>
        <Header title="Account" subtitle={user.email} />
        <div className="flex items-center justify-between px-5 py-4">
          <p className="text-xs text-muted">
            Signed in as <span className="font-medium text-fg">{user.email}</span>
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={async () => {
              await api.logout();
              setToken(null);
              onSignOut();
            }}
          >
            Sign out
          </Button>
        </div>
      </Card>
    </div>
  );
}

function ChangeEmail({
  user,
  onUserChange,
}: {
  user: User;
  onUserChange: (u: User) => void;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    setMsg(null);
    const res = await api.changeEmail(email.trim(), password);
    setBusy(false);
    if (res.data?.user) {
      onUserChange({ ...user, email: res.data.user.email });
      setEmail("");
      setPassword("");
      setMsg({ ok: true, text: `Email changed to ${res.data.user.email}.` });
    } else {
      setMsg({ ok: false, text: res.error ?? "Could not change email." });
    }
  };

  return (
    <Card>
      <Header title="Change email" subtitle="Requires your current password" />
      <div className="flex flex-col gap-3 px-5 py-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="New email" type="email" value={email} onChange={setEmail} />
          <Field
            label="Current password"
            type="password"
            value={password}
            onChange={setPassword}
            onEnter={submit}
          />
        </div>
        <div className="flex items-center gap-3">
          <Button onClick={submit} disabled={busy || !email.trim() || !password}>
            Update email
          </Button>
          {msg && (
            <span className={`text-xs ${msg.ok ? "text-muted" : "text-retrain"}`}>
              {msg.text}
            </span>
          )}
        </div>
      </div>
    </Card>
  );
}

function ChangePassword({ onSignOut }: { onSignOut: () => void }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    setMsg(null);
    const res = await api.changePassword(current, next);
    setBusy(false);
    if (res.data) {
      // Changing a password invalidates every session, including this one.
      setMsg({ ok: true, text: "Password changed. Signing you out…" });
      setTimeout(() => {
        setToken(null);
        onSignOut();
      }, 1400);
    } else {
      setMsg({ ok: false, text: res.error ?? "Could not change password." });
    }
  };

  return (
    <Card>
      <Header
        title="Change password"
        subtitle="Signs out every device, including this one"
      />
      <div className="flex flex-col gap-3 px-5 py-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field
            label="Current password"
            type="password"
            value={current}
            onChange={setCurrent}
          />
          <Field
            label="New password"
            type="password"
            value={next}
            onChange={setNext}
            onEnter={submit}
            hint="At least 8 characters"
          />
        </div>
        <div className="flex items-center gap-3">
          <Button onClick={submit} disabled={busy || !current || next.length < 8}>
            Update password
          </Button>
          {msg && (
            <span className={`text-xs ${msg.ok ? "text-muted" : "text-retrain"}`}>
              {msg.text}
            </span>
          )}
        </div>
      </div>
    </Card>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
  hint,
  onEnter,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  hint?: string;
  onEnter?: () => void;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-2xs font-medium uppercase tracking-wider text-subtle">
        {label}
      </span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onEnter?.()}
        className="rounded-md border border-border bg-bg px-3 py-2 text-sm placeholder:text-subtle focus:border-borderStrong focus:outline-none"
      />
      {hint && <span className="text-2xs text-subtle">{hint}</span>}
    </label>
  );
}

function Header({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="border-b border-border px-5 py-3">
      <h3 className="text-sm font-medium">{title}</h3>
      {subtitle && <p className="pt-0.5 text-xs text-muted">{subtitle}</p>}
    </div>
  );
}
