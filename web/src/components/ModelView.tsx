// One model's view: metrics, drift history, events, features, and its own chat.
//
// Each model gets a dedicated chat because the context differs — this panel is
// given that model's feature names, statistics and verdicts, so it can cite
// "transaction_amount moved 240 -> 890" instead of talking in generalities.
"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ACTION_COLOR,
  type EventCard,
  type EventDetail,
  type LiveModel,
  type ModelMetrics,
  TAXONOMY_LABEL,
  api,
  fmtNum,
  relativeTime,
} from "@/lib/api";
import { ChatPanel } from "./ChatPanel";
import { ModelDocs } from "./ModelDocs";
import {
  ActionBadge,
  Card,
  ConfidenceMeter,
  EmptyState,
  Skeleton,
  Stat,
  SuppressedNotice,
  TaxonomyBadge,
} from "./ui";

export function ModelView({ model }: { model: LiveModel }) {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [events, setEvents] = useState<EventCard[]>([]);
  const [detail, setDetail] = useState<EventDetail | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    const [m, e] = await Promise.all([
      api.modelMetrics(model.model_id),
      api.modelEvents(model.model_id),
    ]);
    setMetrics(m.data);
    setEvents(e.data?.events ?? []);
    setLoading(false);
  }, [model.model_id]);

  useEffect(() => {
    setLoading(true);
    setDetail(null);
    load();
  }, [load]);

  // Live telemetry keeps arriving and reasoning finishes in the background,
  // so this view refreshes itself rather than needing a manual reload.
  useEffect(() => {
    const t = setInterval(load, 8000);
    return () => clearInterval(t);
  }, [load]);

  const openEvent = async (id: string) => {
    const res = await api.liveEventDetail(id);
    setDetail(res.data);
  };

  return (
    <div className="flex flex-col gap-4">
      {/* ---- metric row ---- */}
      <Card className="overflow-hidden">
        {loading && !metrics ? (
          <Skeleton className="h-24 w-full" />
        ) : (
          <>
            <div className="grid divide-y divide-border sm:grid-cols-2 sm:divide-y-0 lg:grid-cols-5 lg:divide-x">
              <Stat
                label="Telemetry"
                value={(metrics?.telemetry_count ?? 0).toLocaleString()}
                note="Predictions received from your model."
              />
              <Stat
                label="Alerts fired"
                value={metrics?.alerts_fired ?? 0}
                note={`Crossed the ${metrics?.drift_threshold ?? 0.15} threshold.`}
              />
              <Stat
                label="Actionable"
                value={metrics?.actionable ?? 0}
                note="Arbiter judged these worth acting on."
              />
              <Stat
                label="Suppressed"
                value={metrics?.suppressed ?? 0}
                note="Real drift, but immaterial or not retrainable."
              />
              <Stat
                label="Noise reduction"
                value={`${metrics?.noise_reduction_pct ?? 0}%`}
                note="Pages a human did not receive."
                accent={(metrics?.noise_reduction_pct ?? 0) > 0 ? "#000" : undefined}
              />
            </div>
            {metrics && metrics.alerts_fired > 0 && !metrics.reportable && (
              <div className="border-t border-border bg-surface px-5 py-2">
                <p className="text-2xs leading-relaxed text-investigate">
                  {metrics.gemma_backed_decisions} of {metrics.alerts_fired} verdicts came
                  from Gemma; the rest used the offline fallback. These figures measure the
                  fallback, not the product.
                </p>
              </div>
            )}
          </>
        )}
      </Card>

      <div className="grid gap-4 xl:grid-cols-3">
        {/* ---- left: charts + events ---- */}
        <div className="flex min-w-0 flex-col gap-4 xl:col-span-2">
          <Card>
            <CardHeader
              title="Drift score over time"
              subtitle={`threshold ${metrics?.drift_threshold ?? 0.15}`}
            />
            <div className="px-2 py-3">
              {metrics?.drift_series?.length ? (
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={metrics.drift_series}>
                    <defs>
                      <linearGradient id="driftFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#000" stopOpacity={0.14} />
                        <stop offset="100%" stopColor="#000" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid vertical={false} stroke="#ebebeb" />
                    <XAxis dataKey="received_at" tick={false} axisLine={false} tickLine={false} />
                    <YAxis
                      domain={[0, 1]}
                      tick={{ fontSize: 11, fill: "#8f8f8f" }}
                      axisLine={false}
                      tickLine={false}
                      width={30}
                    />
                    <Tooltip
                      contentStyle={{
                        border: "1px solid #ebebeb",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                      formatter={(v: number) => [v.toFixed(3), "drift"]}
                      labelFormatter={(l: string) => new Date(l).toLocaleString()}
                    />
                    <ReferenceLine
                      y={metrics.drift_threshold}
                      stroke="#dc2626"
                      strokeDasharray="3 3"
                      label={{
                        value: "threshold",
                        fontSize: 10,
                        fill: "#dc2626",
                        position: "insideTopRight",
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="drift_score"
                      stroke="#000"
                      strokeWidth={1.5}
                      fill="url(#driftFill)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <p className="px-3 py-10 text-center text-xs text-muted">
                  No drift scores yet. They appear as your model sends predictions.
                </p>
              )}
            </div>
          </Card>

          <Card>
            <CardHeader
              title="Drift events"
              subtitle={`${events.length} reasoned by Gemma`}
            />
            {loading && !events.length ? (
              <div className="flex flex-col gap-2 p-4">
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
              </div>
            ) : events.length === 0 ? (
              <EmptyState
                title="No drift events yet"
                body={`Arbiter reasons once drift crosses ${
                  metrics?.drift_threshold ?? 0.15
                } and at least 40 predictions have accumulated. Keep the model running.`}
              />
            ) : (
              <ul className="divide-y divide-border">
                {events.map((e) => (
                  <li key={e.event_id}>
                    <button
                      onClick={() => openEvent(e.event_id)}
                      className="w-full px-5 py-4 text-left transition-colors hover:bg-surface"
                    >
                      <div className="flex flex-wrap items-center gap-2 pb-2">
                        {e.action && <ActionBadge action={e.action} small />}
                        <TaxonomyBadge taxonomy={e.taxonomy} />
                        {e.needs_human && (
                          <span className="rounded border border-[#ca8a0433] bg-[#ca8a040d] px-1.5 py-0.5 text-2xs font-medium text-investigate">
                            ESCALATED
                          </span>
                        )}
                        <span className="nums ml-auto text-2xs text-subtle">
                          drift {e.global_drift_score?.toFixed(2)} ·{" "}
                          {relativeTime(e.timestamp)}
                        </span>
                      </div>
                      <p className="text-xs leading-relaxed text-fg">{e.reasoning}</p>
                      {e.suppressed && e.driftguard_would_page && (
                        <div className="pt-2">
                          <SuppressedNotice />
                        </div>
                      )}
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pt-2">
                        <div className="w-40">
                          <ConfidenceMeter value={e.confidence} />
                        </div>
                        {e.top_features.map((f) => (
                          <span
                            key={f.name}
                            className="nums rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-2xs text-muted"
                          >
                            {f.name} {f.score.toFixed(2)}
                          </span>
                        ))}
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          {detail && <EventFeatures detail={detail} onClose={() => setDetail(null)} />}

          <ModelDocs modelId={model.model_id} />

          <Card>
            <CardHeader
              title="Registered features"
              subtitle={`${model.features.length} named`}
            />
            {model.features.length === 0 ? (
              <div className="px-5 py-4">
                <p className="text-xs leading-relaxed text-investigate">
                  No feature names registered. Narrations cannot name features without
                  them — pass <code className="font-mono">features=</code> or a{" "}
                  <code className="font-mono">feature_extractor</code> to the SDK.
                </p>
              </div>
            ) : (
              <div className="flex flex-wrap gap-1.5 px-5 py-4">
                {model.features.map((f, i) => (
                  <span
                    key={f}
                    className="rounded border border-border bg-surface px-2 py-1 font-mono text-2xs text-muted"
                    title={`positional index ${i}`}
                  >
                    {f}
                  </span>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* ---- right: taxonomy + this model's chat ---- */}
        <div className="flex min-w-0 flex-col gap-4">
          {metrics && Object.keys(metrics.by_taxonomy).length > 0 && (
            <Card>
              <CardHeader title="Drift causes" subtitle="Gemma's taxonomy" />
              <div className="flex flex-col gap-2 px-5 py-4">
                {Object.entries(metrics.by_taxonomy)
                  .sort((a, b) => b[1] - a[1])
                  .map(([tax, n]) => (
                    <div key={tax} className="flex items-center gap-3">
                      <span className="w-32 shrink-0 truncate text-xs text-muted">
                        {TAXONOMY_LABEL[tax] ?? tax}
                      </span>
                      <div className="h-4 flex-1 overflow-hidden rounded bg-surface">
                        <div
                          className="h-full rounded bg-fg transition-all duration-700"
                          style={{
                            width: `${(n / Math.max(metrics.alerts_fired, 1)) * 100}%`,
                          }}
                        />
                      </div>
                      <span className="nums w-6 shrink-0 text-right text-xs text-muted">
                        {n}
                      </span>
                    </div>
                  ))}
              </div>
            </Card>
          )}

          {metrics && Object.keys(metrics.by_action).length > 0 && (
            <Card>
              <CardHeader title="Verdicts" subtitle="What Arbiter decided" />
              <div className="px-2 py-3">
                <ResponsiveContainer width="100%" height={140}>
                  <BarChart
                    data={Object.entries(metrics.by_action).map(([a, n]) => ({
                      action: a,
                      count: n,
                    }))}
                  >
                    <CartesianGrid vertical={false} stroke="#ebebeb" />
                    <XAxis
                      dataKey="action"
                      tick={{ fontSize: 9, fill: "#8f8f8f" }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      allowDecimals={false}
                      tick={{ fontSize: 11, fill: "#8f8f8f" }}
                      axisLine={false}
                      tickLine={false}
                      width={24}
                    />
                    <Tooltip
                      contentStyle={{
                        border: "1px solid #ebebeb",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                    />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]} barSize={34}>
                      {Object.keys(metrics.by_action).map((a) => (
                        <Cell key={a} fill={ACTION_COLOR[a] ?? "#d4d4d4"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          <ChatPanel
            modelId={model.model_id}
            title={`Ask about ${model.model_id}`}
            subtitle="Sees this model's features, statistics and verdicts"
            className="h-[560px]"
          />
        </div>
      </div>
    </div>
  );
}

function EventFeatures({
  detail,
  onClose,
}: {
  detail: EventDetail;
  onClose: () => void;
}) {
  return (
    <Card>
      <div className="flex items-center justify-between border-b border-border px-5 py-3">
        <div>
          <h3 className="text-sm font-medium">Event detail · {detail.event_id}</h3>
          <p className="text-2xs text-muted">
            Reference vs. drifted window, per feature
          </p>
        </div>
        <button
          onClick={onClose}
          className="text-2xs text-subtle transition-colors hover:text-fg"
        >
          Close
        </button>
      </div>

      {detail.analysis?.hypothesis && (
        <div className="border-b border-border px-5 py-4">
          <span className="text-2xs uppercase tracking-wider text-subtle">Root cause</span>
          <p className="pt-1 text-xs leading-relaxed text-fg">
            {detail.analysis.hypothesis}
          </p>
          {detail.analysis.impact && (
            <p className="pt-2 text-xs leading-relaxed text-muted">
              {detail.analysis.impact}
            </p>
          )}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-2xs uppercase tracking-wider text-subtle">
              <th className="px-5 py-2 font-medium">Feature</th>
              <th className="px-3 py-2 text-right font-medium">Score</th>
              <th className="px-3 py-2 text-right font-medium">Reference</th>
              <th className="px-3 py-2 text-right font-medium">Drifted</th>
              <th className="px-3 py-2 text-right font-medium">Change</th>
              <th className="px-5 py-2 text-right font-medium">Nulls</th>
            </tr>
          </thead>
          <tbody>
            {detail.features.map((f) => {
              const primary = detail.primary_features?.includes(f.name);
              const nullJump = f.cur_null_rate - f.ref_null_rate > 0.05;
              return (
                <tr
                  key={f.name}
                  className={`border-b border-border last:border-0 ${
                    primary ? "bg-surface" : ""
                  }`}
                >
                  <td
                    className={`px-5 py-2 font-mono ${
                      primary ? "font-medium text-fg" : "text-muted"
                    }`}
                  >
                    {f.name}
                  </td>
                  <td className="nums px-3 py-2 text-right text-muted">
                    {f.score.toFixed(2)}
                  </td>
                  <td className="nums px-3 py-2 text-right text-muted">
                    {fmtNum(f.ref_mean)}
                  </td>
                  <td className="nums px-3 py-2 text-right text-fg">{fmtNum(f.cur_mean)}</td>
                  <td
                    className="nums px-3 py-2 text-right font-medium"
                    style={{ color: Math.abs(f.pct_change) > 50 ? "#dc2626" : "#666" }}
                  >
                    {f.pct_change > 0 ? "+" : ""}
                    {f.pct_change.toFixed(1)}%
                  </td>
                  <td
                    className="nums px-5 py-2 text-right"
                    style={{ color: nullJump ? "#dc2626" : "#8f8f8f" }}
                  >
                    {(f.cur_null_rate * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {Object.keys(detail.context || {}).length > 0 && (
        <dl className="divide-y divide-border border-t border-border">
          {Object.entries(detail.context).map(([k, v]) => (
            <div key={k} className="flex gap-4 px-5 py-2">
              <dt className="w-52 shrink-0 font-mono text-2xs text-subtle">{k}</dt>
              <dd className="text-xs text-muted">{String(v)}</dd>
            </div>
          ))}
        </dl>
      )}
    </Card>
  );
}

function CardHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-border px-5 py-3">
      <h3 className="text-sm font-medium">{title}</h3>
      {subtitle && <span className="font-mono text-2xs text-subtle">{subtitle}</span>}
    </div>
  );
}
