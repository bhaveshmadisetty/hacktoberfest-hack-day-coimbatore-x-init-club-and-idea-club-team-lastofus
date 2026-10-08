// Shared primitives. Monochrome by default — colour appears only on verdict
// badges and the drift bars, so anything coloured on screen carries judgment.
"use client";

import { ACTION_COLOR, type Action, TAXONOMY_LABEL } from "@/lib/api";

export function Card({
  children,
  className = "",
  hover = false,
}: {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border border-border bg-bg shadow-card ${
        hover ? "transition-shadow hover:shadow-lift" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
}

/** One metric. `note` is the sentence that explains the number — the design
 *  rule for this dashboard is that no figure appears without one. */
export function Stat({
  label,
  value,
  note,
  accent,
}: {
  label: string;
  value: string | number;
  note?: string;
  accent?: string;
}) {
  return (
    <div className="flex flex-col gap-1 px-5 py-4">
      <span className="text-2xs font-medium uppercase tracking-wider text-subtle">{label}</span>
      <span
        className="nums text-3xl font-semibold leading-none tracking-tight"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </span>
      {note && <span className="text-xs leading-snug text-muted">{note}</span>}
    </div>
  );
}

export function ActionBadge({ action, small = false }: { action: Action; small?: boolean }) {
  const color = ACTION_COLOR[action];
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-md border font-medium uppercase tracking-wide ${
        small ? "px-1.5 py-0.5 text-2xs" : "px-2 py-1 text-xs"
      }`}
      style={{ color, borderColor: `${color}33`, background: `${color}0d` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {action}
    </span>
  );
}

export function TaxonomyBadge({ taxonomy }: { taxonomy: string | null }) {
  if (!taxonomy) return null;
  return (
    <span className="inline-flex items-center rounded-md border border-border bg-surface px-2 py-0.5 text-2xs font-medium text-muted">
      {TAXONOMY_LABEL[taxonomy] ?? taxonomy}
    </span>
  );
}

/** Confidence as a bar. Below the floor it reads as an escalation, not a
 *  weak score, because that is what the backend does with it. */
export function ConfidenceMeter({
  value,
  floor = 0.6,
  showLabel = true,
}: {
  value: number;
  floor?: number;
  showLabel?: boolean;
}) {
  const pct = Math.round(value * 100);
  const below = value < floor;
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-border">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: below ? "#ca8a04" : "#000" }}
        />
      </div>
      {showLabel && (
        <span className="nums shrink-0 text-xs font-medium tabular-nums text-muted">{pct}%</span>
      )}
    </div>
  );
}

/** The most persuasive element in the UI: the page that did not happen. */
export function SuppressedNotice() {
  return (
    <div className="flex items-start gap-2 rounded-md border border-dashed border-borderStrong bg-surface px-3 py-2">
      <svg
        className="mt-0.5 h-3.5 w-3.5 shrink-0 text-subtle"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <path d="M8 1.5 14.5 13H1.5L8 1.5Z" strokeLinejoin="round" />
        <path d="M8 6v3.5M8 11.5v.5" strokeLinecap="round" />
      </svg>
      <span className="text-xs leading-snug text-muted">
        <span className="font-medium text-fg">A threshold alert would have paged on this.</span>{" "}
        Arbiter suppressed it.
      </span>
    </div>
  );
}

export function FeatureBar({
  name,
  score,
  max = 1,
  highlight = false,
}: {
  name: string;
  score: number;
  max?: number;
  highlight?: boolean;
}) {
  const pct = Math.min(100, (score / Math.max(max, 0.0001)) * 100);
  return (
    <div className="flex items-center gap-3">
      <span
        className={`w-40 shrink-0 truncate font-mono text-xs ${
          highlight ? "font-medium text-fg" : "text-muted"
        }`}
        title={name}
      >
        {name}
      </span>
      <div className="h-5 flex-1 overflow-hidden rounded bg-surface">
        <div
          className="h-full rounded transition-all duration-700"
          style={{ width: `${pct}%`, background: highlight ? "#000" : "#d4d4d4" }}
        />
      </div>
      <span className="nums w-12 shrink-0 text-right text-xs tabular-nums text-muted">
        {score.toFixed(2)}
      </span>
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  size = "md",
  disabled = false,
  type = "button",
  className = "",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}) {
  const variants = {
    primary: "bg-fg text-bg hover:bg-[#333] border-fg",
    secondary: "bg-bg text-fg border-borderStrong hover:bg-surface",
    ghost: "bg-transparent text-muted border-transparent hover:bg-surface hover:text-fg",
    danger: "bg-bg text-retrain border-[#dc262633] hover:bg-[#dc26260d]",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-2 text-sm"
      } ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

/** Makes the AI step visible. A spinner that names what it is doing reads as
 *  substance rather than latency. */
export function Reasoning({ label = "Gemma is reasoning" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-fg opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-fg" />
      </span>
      {label}…
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded ${className}`} />;
}

export function EmptyState({
  title,
  body,
  action,
}: {
  title: string;
  body: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 px-6 py-16 text-center">
      <h3 className="text-sm font-medium text-fg">{title}</h3>
      <p className="max-w-md text-sm leading-relaxed text-muted">{body}</p>
      {action}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <Card className="border-[#dc262633] bg-[#dc26260a]">
      <div className="flex items-start gap-3 px-5 py-4">
        <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-retrain" />
        <div className="flex flex-col gap-2">
          <p className="text-sm font-medium text-fg">Cannot load data</p>
          <p className="font-mono text-xs leading-relaxed text-muted">{error}</p>
          <p className="text-xs text-muted">
            Start the API with{" "}
            <code className="rounded bg-surface px-1 py-0.5 font-mono">
              uvicorn arbiter.api:app --reload
            </code>
          </p>
          {onRetry && (
            <div className="pt-1">
              <Button variant="secondary" size="sm" onClick={onRetry}>
                Retry
              </Button>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

export function CodeBlock({ code, onCopy }: { code: string; onCopy?: () => void }) {
  return (
    <div className="group relative">
      <pre className="overflow-x-auto rounded-md border border-border bg-surface px-4 py-3 font-mono text-xs leading-relaxed text-fg">
        {code}
      </pre>
      {onCopy && (
        <button
          onClick={onCopy}
          className="absolute right-2 top-2 rounded border border-border bg-bg px-2 py-1 text-2xs font-medium text-muted opacity-0 transition-opacity hover:text-fg group-hover:opacity-100"
        >
          Copy
        </button>
      )}
    </div>
  );
}
