// Tab 2 — event detail.
//
// Left: the evidence (per-feature drift, reference vs. current distributions).
// Right: the judgment (taxonomy, causal hypothesis, confidence, action).
// Putting them side by side is the thesis made visible — a normal dashboard
// stops at the left column.
"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EventDetail as EventDetailT } from "@/lib/api";
import { ACTION_COLOR } from "@/lib/api";
import {
  ActionBadge,
  Card,
  ConfidenceMeter,
  Reasoning,
  Skeleton,
  SuppressedNotice,
  TaxonomyBadge,
} from "./ui";

export function EventDetail({
  detail,
  loading,
}: {
  detail: EventDetailT | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="grid gap-4 lg:grid-cols-5">
        <Skeleton className="h-96 lg:col-span-3" />
        <Skeleton className="h-96 lg:col-span-2" />
      </div>
    );
  }

  if (!detail) {
    return (
      <Card>
        <div className="px-6 py-16 text-center text-sm text-muted">
          Select an event from the triage feed.
        </div>
      </Card>
    );
  }

  const maxScore = Math.max(...detail.features.map((f) => f.score), 0.1);
  const chartData = detail.features.map((f) => ({
    name: f.name,
    score: f.score,
    primary: detail.primary_features?.includes(f.name) ?? false,
  }));

  return (
    <div className="grid gap-4 lg:grid-cols-5">
      {/* ---- Evidence ---- */}
      <div className="flex flex-col gap-4 lg:col-span-3">
        <Card>
          <Header title="Per-feature drift" subtitle={`${detail.features.length} features`} />
          <div className="px-2 pb-4 pt-2">
            <ResponsiveContainer width="100%" height={Math.max(180, detail.features.length * 34)}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 24 }}>
                <CartesianGrid horizontal={false} stroke="#ebebeb" />
                <XAxis
                  type="number"
                  domain={[0, Math.ceil(maxScore * 10) / 10]}
                  tick={{ fontSize: 11, fill: "#8f8f8f" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={132}
                  tick={{ fontSize: 11, fill: "#666", fontFamily: "ui-monospace, monospace" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    border: "1px solid #ebebeb",
                    borderRadius: 6,
                    fontSize: 12,
                    boxShadow: "0 4px 12px rgba(0,0,0,0.06)",
                  }}
                  formatter={(v: number) => [v.toFixed(3), "drift score"]}
                />
                <Bar dataKey="score" radius={[0, 3, 3, 0]} barSize={16}>
                  {chartData.map((d) => (
                    <Cell key={d.name} fill={d.primary ? "#000000" : "#d4d4d4"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="px-4 pt-1 text-2xs text-subtle">
              Black bars are the features Gemma identified as driving this event.
            </p>
          </div>
        </Card>

        <Card>
          <Header
            title="Reference vs. current"
            subtitle="The numbers behind the narration"
          />
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-left text-2xs uppercase tracking-wider text-subtle">
                  <th className="px-4 py-2 font-medium">Feature</th>
                  <th className="px-3 py-2 text-right font-medium">Ref mean</th>
                  <th className="px-3 py-2 text-right font-medium">Current</th>
                  <th className="px-3 py-2 text-right font-medium">Change</th>
                  <th className="px-4 py-2 text-right font-medium">Nulls</th>
                </tr>
              </thead>
              <tbody>
                {detail.features.map((f) => {
                  const isPrimary = detail.primary_features?.includes(f.name);
                  const nullJump = f.cur_null_rate - f.ref_null_rate > 0.05;
                  return (
                    <tr
                      key={f.name}
                      className={`border-b border-border last:border-0 ${
                        isPrimary ? "bg-surface" : ""
                      }`}
                    >
                      <td
                        className={`px-4 py-2 font-mono ${
                          isPrimary ? "font-medium text-fg" : "text-muted"
                        }`}
                      >
                        {f.name}
                      </td>
                      <td className="nums px-3 py-2 text-right text-muted">
                        {fmt(f.ref_mean)}
                      </td>
                      <td className="nums px-3 py-2 text-right text-fg">{fmt(f.cur_mean)}</td>
                      <td
                        className="nums px-3 py-2 text-right font-medium"
                        style={{
                          color: Math.abs(f.pct_change) > 50 ? "#dc2626" : "#666",
                        }}
                      >
                        {f.pct_change > 0 ? "+" : ""}
                        {f.pct_change.toFixed(1)}%
                      </td>
                      <td
                        className="nums px-4 py-2 text-right"
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
        </Card>

        {Object.keys(detail.context || {}).length > 0 && (
          <Card>
            <Header title="Context" subtitle="Signals the detector cannot see" />
            <dl className="divide-y divide-border">
              {Object.entries(detail.context).map(([k, v]) => (
                <div key={k} className="flex gap-4 px-4 py-2">
                  <dt className="w-44 shrink-0 font-mono text-2xs text-subtle">{k}</dt>
                  <dd className="text-xs leading-relaxed text-muted">{String(v)}</dd>
                </div>
              ))}
            </dl>
          </Card>
        )}
      </div>

      {/* ---- Judgment ---- */}
      <div className="flex flex-col gap-4 lg:col-span-2">
        <Card className="overflow-hidden">
          <div
            className="h-1 w-full"
            style={{ background: ACTION_COLOR[detail.action] }}
          />
          <div className="flex flex-col gap-4 px-5 py-4">
            <div className="flex flex-wrap items-center gap-2">
              <ActionBadge action={detail.action} />
              <TaxonomyBadge taxonomy={detail.taxonomy} />
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-2xs uppercase tracking-wider text-subtle">Confidence</span>
              <ConfidenceMeter value={detail.confidence ?? 0} />
              {detail.needs_human && (
                <p className="text-2xs leading-snug text-investigate">
                  Below the 0.60 floor — escalated to a human instead of executing.
                </p>
              )}
            </div>

            {detail.suppressed && detail.driftguard_would_page && <SuppressedNotice />}

            <div className="flex flex-col gap-1.5">
              <span className="text-2xs uppercase tracking-wider text-subtle">
                Recommended action
              </span>
              <p className="text-sm leading-relaxed text-fg">{detail.reasoning}</p>
            </div>
          </div>
        </Card>

        {detail.analysis && (
          <Card>
            <Header title="Root cause" subtitle={detail.analysis.source} />
            <div className="flex flex-col gap-3 px-5 py-4">
              <p className="text-sm leading-relaxed text-fg">{detail.analysis.hypothesis}</p>
              {detail.analysis.impact && (
                <div className="flex flex-col gap-1 border-t border-border pt-3">
                  <span className="text-2xs uppercase tracking-wider text-subtle">
                    Production impact
                  </span>
                  <p className="text-sm leading-relaxed text-muted">
                    {detail.analysis.impact}
                  </p>
                </div>
              )}
            </div>
          </Card>
        )}

        {detail.challenger && (
          <Card>
            <Header
              title="Champion vs. challenger"
              subtitle="Did it improve on the drifted segment?"
            />
            <div className="flex flex-col gap-3 px-5 py-4">
              {detail.challenger.report && (
                <div className="grid grid-cols-2 gap-3">
                  <Delta
                    label="Overall"
                    value={detail.challenger.report.overall_delta}
                  />
                  <Delta
                    label={detail.challenger.report.segment_name}
                    value={detail.challenger.report.segment_delta}
                  />
                </div>
              )}
              <p className="text-sm leading-relaxed text-fg">
                {detail.challenger.reasoning}
              </p>
              <span
                className="w-fit rounded-md border px-2 py-1 text-2xs font-medium uppercase tracking-wide"
                style={{
                  color:
                    detail.challenger.recommendation === "PROMOTE" ? "#000" : "#dc2626",
                  borderColor:
                    detail.challenger.recommendation === "PROMOTE"
                      ? "#d4d4d4"
                      : "#dc262633",
                }}
              >
                {detail.challenger.recommendation}
              </span>
            </div>
          </Card>
        )}

        {detail.quality_bar && (
          <Card>
            <Header title="Narration quality" subtitle="Checked, not assumed" />
            <ul className="flex flex-col divide-y divide-border">
              {Object.entries(detail.quality_bar).map(([criterion, passed]) => (
                <li key={criterion} className="flex items-center gap-2 px-5 py-2">
                  <span
                    className="flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full text-[9px] font-bold text-bg"
                    style={{ background: passed ? "#000" : "#d4d4d4" }}
                  >
                    {passed ? "✓" : "–"}
                  </span>
                  <span className="text-xs text-muted">
                    {criterion.replace(/_/g, " ")}
                  </span>
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    </div>
  );
}

function Header({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-border px-5 py-3">
      <h3 className="text-sm font-medium text-fg">{title}</h3>
      {subtitle && <span className="font-mono text-2xs text-subtle">{subtitle}</span>}
    </div>
  );
}

function Delta({ label, value }: { label: string; value: number | null }) {
  if (value === null) {
    return (
      <div className="flex flex-col gap-0.5 rounded-md border border-dashed border-borderStrong px-3 py-2">
        <span className="text-2xs uppercase tracking-wider text-subtle">{label}</span>
        <span className="text-sm text-muted">not measured</span>
      </div>
    );
  }
  const good = value >= 0;
  return (
    <div className="flex flex-col gap-0.5 rounded-md border border-border bg-surface px-3 py-2">
      <span className="truncate text-2xs uppercase tracking-wider text-subtle">{label}</span>
      <span
        className="nums text-lg font-semibold tracking-tight"
        style={{ color: good ? "#000" : "#dc2626" }}
      >
        {value > 0 ? "+" : ""}
        {(value * 100).toFixed(2)}%
      </span>
    </div>
  );
}

function fmt(n: number): string {
  const a = Math.abs(n);
  if (a >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 10) return n.toFixed(1);
  if (a >= 1) return n.toFixed(2);
  return n.toFixed(3);
}
