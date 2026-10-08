// Arbiter dashboard.
//
// Design rule, applied everywhere: every number on screen is paired with a
// sentence explaining it. A conventional drift dashboard shows charts; this
// shows judgment, and the layout is what makes that difference legible.
//
// Data source toggle: "Demo corpus" is the 20-event synthetic replay with
// hand-labelled causes; "Live SDK" is real telemetry arriving from a user's
// own model. They are kept visibly separate so a measured number is never
// mistaken for a scripted one.
"use client";

import { useCallback, useEffect, useState } from "react";
import {
  type EventCard,
  type EventDetail as EventDetailT,
  type Health,
  type Metrics,
  api,
} from "@/lib/api";
import { AskArbiter } from "@/components/AskArbiter";
import { Connect } from "@/components/Connect";
import { EventDetail } from "@/components/EventDetail";
import { TriageFeed } from "@/components/TriageFeed";
import { Card, ErrorState, Stat } from "@/components/ui";

type Tab = "triage" | "detail" | "ask" | "connect";
type Source = "demo" | "live";

const TABS: { id: Tab; label: string }[] = [
  { id: "triage", label: "Triage feed" },
  { id: "detail", label: "Event detail" },
  { id: "ask", label: "Ask Arbiter" },
  { id: "connect", label: "Connect" },
];

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("triage");
  const [source, setSource] = useState<Source>("demo");
  const [health, setHealth] = useState<Health | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [events, setEvents] = useState<EventCard[]>([]);
  const [detail, setDetail] = useState<EventDetailT | null>(null);
  const [selectedId, setSelectedId] = useState<string>();
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    const [h, m, e] = await Promise.all([
      api.health(),
      api.metrics(),
      source === "demo" ? api.events() : api.liveEvents(),
    ]);
    setHealth(h.data);
    setMetrics(m.data);
    setEvents(e.data?.events ?? []);
    setError(h.error || e.error);
    setLoading(false);
  }, [source]);

  useEffect(() => {
    load();
  }, [load]);

  // Live telemetry keeps arriving, so the live view refreshes itself. The demo
  // corpus is static and polling it would only burn requests.
  useEffect(() => {
    if (source !== "live") return;
    const t = setInterval(() => api.liveEvents().then((r) => setEvents(r.data?.events ?? [])), 5000);
    return () => clearInterval(t);
  }, [source]);

  const select = async (id: string) => {
    setSelectedId(id);
    setTab("detail");
    setDetailLoading(true);
    const res = source === "demo" ? await api.eventDetail(id) : await api.liveEventDetail(id);
    setDetail(res.data);
    setDetailLoading(false);
  };

  const live = events.filter((e) => !e.suppressed).length;
  const suppressed = events.length - live;
  const noise = events.length ? Math.round((suppressed / events.length) * 100) : 0;

  return (
    <div className="mx-auto flex min-h-screen max-w-[1180px] flex-col gap-6 px-6 py-8">
      {/* ---- Header ---- */}
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2.5">
            <Logo />
            <h1 className="text-xl font-semibold tracking-tight">Arbiter</h1>
            <span className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-2xs text-subtle">
              v{health?.version ?? "0.1.0"}
            </span>
          </div>
          <p className="max-w-xl text-sm leading-relaxed text-muted">
            Your monitoring tells you a number crossed a threshold. Arbiter tells you what to
            do about it.
          </p>
        </div>

        <div className="flex flex-col items-end gap-2">
          <GemmaStatus health={health} />
          <div className="flex items-center rounded-md border border-border p-0.5">
            {(["demo", "live"] as Source[]).map((s) => (
              <button
                key={s}
                onClick={() => {
                  setSource(s);
                  setDetail(null);
                  setSelectedId(undefined);
                  setTab("triage");
                }}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                  source === s ? "bg-fg text-bg" : "text-muted hover:text-fg"
                }`}
              >
                {s === "demo" ? "Demo corpus" : "Live SDK"}
              </button>
            ))}
          </div>
        </div>
      </header>

      {error && <ErrorState error={error} onRetry={load} />}

      {/* ---- Metric row. Measured, never hardcoded. ---- */}
      <Card className="overflow-hidden">
        <div className="grid divide-y divide-border sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-4 lg:divide-x">
          <Stat
            label="Alerts fired"
            value={events.length}
            note="Every one crossed DriftGuard's 0.15 threshold."
          />
          <Stat
            label="Actionable"
            value={live}
            note="Arbiter judged these worth acting on."
          />
          <Stat
            label="Suppressed"
            value={suppressed}
            note="Real drift, but immaterial or not retrainable."
          />
          <Stat
            label="Noise reduction"
            value={`${noise}%`}
            note="Pages a human did not receive."
            accent={noise > 0 ? "#000" : undefined}
          />
        </div>
        {source === "demo" && metrics && !metrics.reportable && (
          <div className="border-t border-border bg-surface px-5 py-2">
            <p className="text-2xs leading-relaxed text-investigate">
              {metrics.gemma_backed_decisions} of {events.length} verdicts came from Gemma;
              the rest used the heuristic fallback. These figures measure the fallback, not
              the product — re-run with a working GEMMA_API_KEY before quoting them.
            </p>
          </div>
        )}
        {source === "demo" && metrics?.accuracy?.taxonomy_accuracy_pct !== null &&
          metrics?.accuracy?.taxonomy_accuracy_pct !== undefined && (
            <div className="border-t border-border bg-surface px-5 py-2">
              <p className="text-2xs leading-relaxed text-muted">
                Taxonomy accuracy{" "}
                <span className="nums font-medium text-fg">
                  {metrics.accuracy.taxonomy_correct}/{metrics.accuracy.labelled_events} (
                  {metrics.accuracy.taxonomy_accuracy_pct}%)
                </span>{" "}
                against hand-labelled causes the model never sees.
              </p>
            </div>
          )}
      </Card>

      {/* ---- Tabs ---- */}
      <nav className="flex items-center gap-1 border-b border-border">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`-mb-px border-b-2 px-3 py-2.5 text-sm font-medium transition-colors ${
              tab === t.id ? "border-fg text-fg" : "border-transparent text-muted hover:text-fg"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="flex-1">
        {tab === "triage" && (
          <TriageFeed
            events={events}
            loading={loading}
            onSelect={select}
            selectedId={selectedId}
          />
        )}
        {tab === "detail" && <EventDetail detail={detail} loading={detailLoading} />}
        {tab === "ask" && <AskArbiter />}
        {tab === "connect" && <Connect />}
      </main>

      <footer className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-border pt-4 text-2xs text-subtle">
        <span>
          Reasoning by{" "}
          <span className="font-mono text-muted">{health?.gemma.model ?? "gemma-4"}</span>{" "}
          (open-weight, hosted inference)
        </span>
        <span>·</span>
        <span>
          Drift detection by{" "}
          <span className="font-mono text-muted">driftguard-ai-sdk</span>, an unmodified pip
          dependency
        </span>
        <span>·</span>
        <span>
          Verdicts below {((health?.confidence_floor ?? 0.6) * 100).toFixed(0)}% confidence
          escalate to a human
        </span>
      </footer>
    </div>
  );
}

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M12 2 2 20h20L12 2Z" fill="#000" />
    </svg>
  );
}

function GemmaStatus({ health }: { health: Health | null }) {
  if (!health) {
    return (
      <span className="flex items-center gap-1.5 text-xs text-subtle">
        <span className="h-1.5 w-1.5 rounded-full bg-border" />
        connecting
      </span>
    );
  }
  const ok = health.gemma.available;
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="flex items-center gap-1.5">
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{ background: ok ? "#000" : "#ca8a04" }}
        />
        <span className={ok ? "text-muted" : "text-investigate"}>
          {ok ? "Gemma 4 live" : "Gemma unavailable"}
        </span>
      </span>
      {health.gemma.cached_responses > 0 && (
        <span className="nums font-mono text-subtle">
          {health.gemma.cached_responses} cached
        </span>
      )}
    </div>
  );
}
