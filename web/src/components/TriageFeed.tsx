// Tab 1 — the triage feed. The money shot.
//
// Ordering is deliberate: act-worthy events first, then escalations, then
// suppressed. A judge scanning top-to-bottom sees decisions before noise,
// which is the inverse of what a threshold-based alert list gives you.
"use client";

import { useState } from "react";
import {
  type EventCard as EventCardT,
  ACTION_COLOR,
  relativeTime,
} from "@/lib/api";
import {
  ActionBadge,
  Card,
  ConfidenceMeter,
  EmptyState,
  Skeleton,
  SuppressedNotice,
  TaxonomyBadge,
} from "./ui";

type Filter = "all" | "actionable" | "suppressed";

export function TriageFeed({
  events,
  loading,
  onSelect,
  selectedId,
}: {
  events: EventCardT[];
  loading: boolean;
  onSelect: (id: string) => void;
  selectedId?: string;
}) {
  const [filter, setFilter] = useState<Filter>("all");

  const shown = events.filter((e) =>
    filter === "all" ? true : filter === "actionable" ? !e.suppressed : e.suppressed,
  );

  const counts = {
    all: events.length,
    actionable: events.filter((e) => !e.suppressed).length,
    suppressed: events.filter((e) => e.suppressed).length,
  };

  if (loading) {
    return (
      <div className="flex flex-col gap-3">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-40 w-full" />
        ))}
      </div>
    );
  }

  if (!events.length) {
    return (
      <Card>
        <EmptyState
          title="No drift events yet"
          body="Connect a model with the DriftGuard SDK, or load the demo corpus to see Arbiter reason over drift telemetry."
        />
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-1 border-b border-border pb-px">
        {(["all", "actionable", "suppressed"] as Filter[]).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium capitalize transition-colors ${
              filter === f
                ? "border-fg text-fg"
                : "border-transparent text-muted hover:text-fg"
            }`}
          >
            {f}
            <span className="nums ml-1.5 text-xs text-subtle">{counts[f]}</span>
          </button>
        ))}
      </div>

      <div className="flex flex-col gap-3">
        {shown.map((event) => (
          <EventRow
            key={event.event_id}
            event={event}
            selected={event.event_id === selectedId}
            onSelect={() => onSelect(event.event_id)}
          />
        ))}
      </div>
    </div>
  );
}

function EventRow({
  event,
  selected,
  onSelect,
}: {
  event: EventCardT;
  selected: boolean;
  onSelect: () => void;
}) {
  const color = ACTION_COLOR[event.action];

  return (
    <Card
      hover
      className={`cursor-pointer animate-fade-in overflow-hidden ${
        selected ? "ring-1 ring-fg" : ""
      }`}
    >
      <div
        onClick={onSelect}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onSelect();
          }
        }}
        className="relative"
      >
        {/* Verdict colour as a left rule — reads at a glance when scanning. */}
        <span className="absolute inset-y-0 left-0 w-0.5" style={{ background: color }} />

        <div className="flex flex-col gap-3 px-5 py-4 pl-6">
          <div className="flex flex-wrap items-center gap-2">
            <ActionBadge action={event.action} />
            <TaxonomyBadge taxonomy={event.taxonomy} />
            {event.live && (
              <span className="inline-flex items-center gap-1 rounded-md border border-border bg-surface px-2 py-0.5 text-2xs font-medium text-muted">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-fg" />
                LIVE SDK
              </span>
            )}
            {event.needs_human && (
              <span className="rounded-md border border-[#ca8a0433] bg-[#ca8a040d] px-2 py-0.5 text-2xs font-medium text-investigate">
                ESCALATED
              </span>
            )}
            <div className="ml-auto flex items-center gap-3 text-xs text-subtle">
              <span className="font-mono">{event.model_id}</span>
              <span className="nums">drift {event.global_drift_score?.toFixed(2)}</span>
              <span>{relativeTime(event.timestamp)}</span>
            </div>
          </div>

          {/* The narration. This is the product. */}
          <p className="text-sm leading-relaxed text-fg">{event.reasoning}</p>

          {event.suppressed && event.driftguard_would_page && <SuppressedNotice />}

          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <div className="flex min-w-[180px] max-w-[260px] flex-1 flex-col gap-1">
              <span className="text-2xs uppercase tracking-wider text-subtle">Confidence</span>
              <ConfidenceMeter value={event.confidence ?? 0} />
            </div>
            {event.top_features?.length > 0 && (
              <div className="flex flex-col gap-1">
                <span className="text-2xs uppercase tracking-wider text-subtle">
                  Top drifted features
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {event.top_features.map((f) => (
                    <span
                      key={f.name}
                      className="nums rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-2xs text-muted"
                    >
                      {f.name} {f.score.toFixed(2)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
