// Arbiter dashboard — full-width, model-scoped.
//
// Shape: a persistent left rail lists the account's models and switching one
// swaps the whole right side. That is the mental model the product needs —
// drift is a property of a model, not of a fleet-wide feed, and every metric,
// chart and chatbot on screen belongs to exactly one model.
//
// Overview holds the fleet-wide chatbot; each model holds its own, with its own
// context. Settings holds keys and account changes.
"use client";

import { useCallback, useEffect, useState } from "react";
import {
  type Health,
  type LiveModel,
  type User,
  api,
  getToken,
  onAuthChange,
  relativeTime,
  setToken,
} from "@/lib/api";
import { ChatPanel } from "@/components/ChatPanel";
import { ModelView } from "@/components/ModelView";
import { Settings } from "@/components/Settings";
import { SignIn } from "@/components/SignIn";
import { ActionBadge, Card, EmptyState, ErrorState, Skeleton, Stat } from "@/components/ui";

type Pane = { kind: "overview" } | { kind: "model"; id: string } | { kind: "settings" };

export default function Page() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(true);
  const [models, setModels] = useState<LiveModel[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [pane, setPane] = useState<Pane>({ kind: "overview" });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Restore a stored session on load, and react to it being cleared (an
  // expired token clears itself inside the API client).
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      if (!getToken()) {
        if (!cancelled) {
          setUser(null);
          setChecking(false);
        }
        return;
      }
      const res = await api.me();
      if (!cancelled) {
        setUser(res.data?.user ?? null);
        setChecking(false);
      }
    };
    check();
    const off = onAuthChange(() => {
      if (!getToken()) setUser(null);
    });
    return () => {
      cancelled = true;
      off();
    };
  }, []);

  const load = useCallback(async () => {
    if (!user) return;
    const [m, h] = await Promise.all([api.models(), api.health()]);
    setModels(m.data?.models ?? []);
    setHealth(h.data);
    setError(m.error || h.error);
    setLoading(false);
  }, [user]);

  useEffect(() => {
    if (user) load();
  }, [user, load]);

  // Models register themselves on first contact, so poll to surface one the
  // moment a user runs their code — that moment is the payoff of onboarding.
  useEffect(() => {
    if (!user) return;
    const t = setInterval(load, 6000);
    return () => clearInterval(t);
  }, [user, load]);

  if (checking) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Skeleton className="h-10 w-40" />
      </div>
    );
  }

  if (!user) return <SignIn onSignedIn={setUser} />;

  const active = pane.kind === "model" ? models.find((m) => m.model_id === pane.id) : null;

  return (
    <div className="flex min-h-screen w-full">
      {/* ---- left rail ---- */}
      <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-surface">
        <div className="flex items-center gap-2 border-b border-border px-4 py-4">
          <Logo />
          <span className="text-sm font-semibold tracking-tight">Arbiter</span>
          <span className="ml-auto rounded border border-border bg-bg px-1.5 py-0.5 font-mono text-2xs text-subtle">
            v{health?.version ?? "0.1.0"}
          </span>
        </div>

        <nav className="flex flex-col gap-0.5 p-2">
          <RailButton
            active={pane.kind === "overview"}
            onClick={() => setPane({ kind: "overview" })}
            label="Overview"
          />
          <RailButton
            active={pane.kind === "settings"}
            onClick={() => setPane({ kind: "settings" })}
            label="Settings"
          />
        </nav>

        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex items-center justify-between px-4 py-2">
            <span className="text-2xs font-medium uppercase tracking-wider text-subtle">
              Your models
            </span>
            <span className="nums text-2xs text-subtle">{models.length}</span>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
            {loading && !models.length ? (
              <div className="flex flex-col gap-1 px-2">
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
              </div>
            ) : models.length === 0 ? (
              <p className="px-2 py-3 text-2xs leading-relaxed text-subtle">
                None yet. Generate a key in Settings, point the SDK at Arbiter, and your
                model appears here.
              </p>
            ) : (
              <ul className="flex flex-col gap-0.5">
                {models.map((m) => {
                  const selected = pane.kind === "model" && pane.id === m.model_id;
                  const action = m.latest_verdict?.action;
                  return (
                    <li key={m.model_id}>
                      <button
                        onClick={() => setPane({ kind: "model", id: m.model_id })}
                        className={`w-full rounded-md px-2.5 py-2 text-left transition-colors ${
                          selected ? "bg-bg shadow-card" : "hover:bg-bg"
                        }`}
                      >
                        <div className="flex items-center gap-1.5">
                          <span
                            className={`truncate font-mono text-xs ${
                              selected ? "font-medium text-fg" : "text-muted"
                            }`}
                            title={m.model_id}
                          >
                            {m.model_id}
                          </span>
                          {action && (
                            <span
                              className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full"
                              style={{
                                background:
                                  action === "RETRAIN" || action === "ROLLBACK"
                                    ? "#dc2626"
                                    : "#64748b",
                              }}
                              title={action}
                            />
                          )}
                        </div>
                        <div className="nums flex gap-2 pt-0.5 text-2xs text-subtle">
                          <span>{m.telemetry_count.toLocaleString()} rows</span>
                          {m.event_count > 0 && <span>· {m.event_count} events</span>}
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>

        <div className="border-t border-border px-4 py-3">
          <GemmaStatus health={health} />
          <p className="truncate pt-1.5 text-2xs text-subtle" title={user.email}>
            {user.email}
          </p>
        </div>
      </aside>

      {/* ---- main pane, full remaining width ---- */}
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between gap-4 border-b border-border px-6 py-4">
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold tracking-tight">
              {pane.kind === "overview"
                ? "Overview"
                : pane.kind === "settings"
                  ? "Settings"
                  : pane.id}
            </h1>
            <p className="truncate text-xs text-muted">
              {pane.kind === "overview"
                ? "Every model connected to your account"
                : pane.kind === "settings"
                  ? "API keys and account"
                  : active
                    ? `${active.features.length} features · last seen ${relativeTime(
                        active.last_seen,
                      )}`
                    : ""}
            </p>
          </div>
          {pane.kind === "model" && active?.latest_verdict?.action && (
            <ActionBadge action={active.latest_verdict.action} />
          )}
        </header>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          {error && (
            <div className="pb-4">
              <ErrorState error={error} onRetry={load} />
            </div>
          )}

          {pane.kind === "settings" && (
            <Settings
              user={user}
              onUserChange={setUser}
              onSignOut={() => {
                setToken(null);
                setUser(null);
                setPane({ kind: "overview" });
              }}
            />
          )}

          {pane.kind === "overview" && (
            <Overview
              models={models}
              loading={loading}
              onOpen={(id) => setPane({ kind: "model", id })}
              onSettings={() => setPane({ kind: "settings" })}
            />
          )}

          {pane.kind === "model" &&
            (active ? (
              <ModelView key={active.model_id} model={active} />
            ) : (
              <Card>
                <EmptyState
                  title="Model not found"
                  body="It may have been removed, or it belongs to another account."
                />
              </Card>
            ))}
        </div>

        <footer className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-border px-6 py-3 text-2xs text-subtle">
          <span>
            Reasoning by{" "}
            <span className="font-mono text-muted">{health?.gemma.model ?? "gemma-4"}</span>{" "}
            (open-weight, hosted inference)
          </span>
          <span>·</span>
          <span>
            Drift detection by <span className="font-mono text-muted">driftguard-ai-sdk</span>,
            an unmodified pip dependency
          </span>
          <span>·</span>
          <span>
            Verdicts below {((health?.confidence_floor ?? 0.6) * 100).toFixed(0)}% confidence
            escalate to a human
          </span>
        </footer>
      </main>
    </div>
  );
}

function Overview({
  models,
  loading,
  onOpen,
  onSettings,
}: {
  models: LiveModel[];
  loading: boolean;
  onOpen: (id: string) => void;
  onSettings: () => void;
}) {
  const totals = models.reduce(
    (acc, m) => ({
      telemetry: acc.telemetry + m.telemetry_count,
      events: acc.events + m.event_count,
      actionable: acc.actionable + m.actionable,
      suppressed: acc.suppressed + m.suppressed,
    }),
    { telemetry: 0, events: 0, actionable: 0, suppressed: 0 },
  );
  const noise = totals.events
    ? Math.round((totals.suppressed / totals.events) * 100)
    : 0;

  if (loading && !models.length) {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (!models.length) {
    return (
      <Card>
        <EmptyState
          title="No models connected yet"
          body="Generate an API key in Settings, set DRIFTGUARD_API_URL and DRIFTGUARD_API_KEY in your own code, and run it. Your model registers itself on the first prediction and appears here."
          action={
            <button
              onClick={onSettings}
              className="rounded-md border border-fg bg-fg px-3.5 py-2 text-sm font-medium text-bg transition-colors hover:bg-[#333]"
            >
              Get an API key
            </button>
          }
        />
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <Card className="overflow-hidden">
        <div className="grid divide-y divide-border sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-5 lg:divide-x">
          <Stat label="Models" value={models.length} note="Connected via your API keys." />
          <Stat
            label="Telemetry"
            value={totals.telemetry.toLocaleString()}
            note="Predictions received in total."
          />
          <Stat
            label="Actionable"
            value={totals.actionable}
            note="Verdicts worth acting on."
          />
          <Stat
            label="Suppressed"
            value={totals.suppressed}
            note="Real drift, judged immaterial."
          />
          <Stat
            label="Noise reduction"
            value={`${noise}%`}
            note="Pages a human did not receive."
            accent={noise > 0 ? "#000" : undefined}
          />
        </div>
      </Card>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="min-w-0 xl:col-span-2">
          <div className="border-b border-border px-5 py-3">
            <h3 className="text-sm font-medium">Models</h3>
            <p className="text-xs text-muted">Select one to open its metrics and chat.</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-left text-2xs uppercase tracking-wider text-subtle">
                  <th className="px-5 py-2 font-medium">Model</th>
                  <th className="px-3 py-2 font-medium">Features</th>
                  <th className="px-3 py-2 text-right font-medium">Telemetry</th>
                  <th className="px-3 py-2 text-right font-medium">Events</th>
                  <th className="px-3 py-2 font-medium">Latest verdict</th>
                  <th className="px-5 py-2 text-right font-medium">Last seen</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr
                    key={m.model_id}
                    onClick={() => onOpen(m.model_id)}
                    className="cursor-pointer border-b border-border transition-colors last:border-0 hover:bg-surface"
                  >
                    <td className="px-5 py-2.5 font-mono font-medium text-fg">
                      {m.model_id}
                    </td>
                    <td className="px-3 py-2.5 text-muted">
                      {m.features.length > 0 ? (
                        `${m.features.length} named`
                      ) : (
                        <span className="text-investigate">none</span>
                      )}
                    </td>
                    <td className="nums px-3 py-2.5 text-right text-muted">
                      {m.telemetry_count.toLocaleString()}
                    </td>
                    <td className="nums px-3 py-2.5 text-right text-muted">
                      {m.event_count}
                    </td>
                    <td className="px-3 py-2.5">
                      {m.latest_verdict?.action ? (
                        <ActionBadge action={m.latest_verdict.action} small />
                      ) : (
                        <span className="text-subtle">—</span>
                      )}
                    </td>
                    <td className="px-5 py-2.5 text-right text-subtle">
                      {relativeTime(m.last_seen)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <ChatPanel
          title="Ask Arbiter"
          subtitle="Spans every model on your account"
          className="h-[520px] min-w-0"
        />
      </div>
    </div>
  );
}

function RailButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-2.5 py-1.5 text-left text-sm font-medium transition-colors ${
        active ? "bg-bg text-fg shadow-card" : "text-muted hover:bg-bg hover:text-fg"
      }`}
    >
      {label}
    </button>
  );
}

function GemmaStatus({ health }: { health: Health | null }) {
  if (!health) {
    return (
      <span className="flex items-center gap-1.5 text-2xs text-subtle">
        <span className="h-1.5 w-1.5 rounded-full bg-border" />
        connecting
      </span>
    );
  }
  const ok = health.gemma.available;
  return (
    <span className="flex items-center gap-1.5 text-2xs">
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: ok ? "#000" : "#ca8a04" }}
      />
      <span className={ok ? "text-muted" : "text-investigate"}>
        {ok ? "Gemma 4 live" : "Gemma unavailable"}
      </span>
    </span>
  );
}

function Logo() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M12 2 2 20h20L12 2Z" fill="#000" />
    </svg>
  );
}
