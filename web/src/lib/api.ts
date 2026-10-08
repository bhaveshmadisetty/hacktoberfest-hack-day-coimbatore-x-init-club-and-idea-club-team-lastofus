// Typed client for the Arbiter API.
//
// Every call goes through `request`, which never throws on a network failure:
// the dashboard has to stay legible when the backend is down, so callers get
// `{ data: null, error }` and render an explicit offline state instead of a
// blank screen or an unhandled rejection.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export type Action = "RETRAIN" | "ROLLBACK" | "INVESTIGATE" | "BENIGN";

export type Taxonomy =
  | "covariate_shift"
  | "label_shift"
  | "concept_drift"
  | "schema_break"
  | "seasonal"
  | "upstream_bug";

export interface FeatureScore {
  name: string;
  score: number;
}

export interface EventCard {
  event_id: string;
  model_id: string;
  timestamp: string;
  global_drift_score: number;
  action: Action;
  confidence: number;
  taxonomy: Taxonomy | null;
  reasoning: string;
  primary_features: string[];
  top_features: FeatureScore[];
  suppressed: boolean;
  /** True when DriftGuard's fixed threshold would have paged a human. */
  driftguard_would_page: boolean;
  needs_human: boolean;
  source: string;
  live?: boolean;
}

export interface FeatureDetail {
  name: string;
  score: number;
  ref_mean: number;
  cur_mean: number;
  ref_std: number;
  cur_std: number;
  ref_null_rate: number;
  cur_null_rate: number;
  pct_change: number;
}

export interface Analysis {
  taxonomy: Taxonomy | null;
  primary_features: string[];
  hypothesis: string;
  impact: string;
  confidence: number;
  source: string;
}

export interface ChallengerVerdict {
  recommendation: "PROMOTE" | "HOLD" | "REJECT";
  confidence: number;
  reasoning: string;
  blocking_evidence: string;
  report: {
    metric_name: string;
    segment_name: string;
    champion_metric: number;
    challenger_metric: number;
    champion_segment_metric: number | null;
    challenger_segment_metric: number | null;
    overall_delta: number;
    segment_delta: number | null;
  } | null;
  source: string;
}

export interface EventDetail extends EventCard {
  features: FeatureDetail[];
  context: Record<string, unknown>;
  analysis: Analysis | null;
  quality_bar: Record<string, boolean> | null;
  challenger?: ChallengerVerdict;
}

export interface Metrics {
  alerts_fired: number;
  actionable: number;
  suppressed: number;
  noise_reduction_pct: number;
  escalated_to_human: number;
  by_action: Record<string, number>;
  by_source: Record<string, number>;
  gemma_backed_decisions: number;
  /** False when any verdict came from the heuristic fallback. */
  reportable: boolean;
  accuracy: {
    labelled_events: number;
    taxonomy_correct: number;
    taxonomy_accuracy_pct: number | null;
    skipped_heuristic?: number;
    note?: string;
  };
  corpus: {
    total_events: number;
    all_fired_by_detector: boolean;
    by_ground_truth: Record<string, number>;
    models: string[];
  };
  gemma: GemmaStats;
}

export interface GemmaStats {
  model: string;
  available: boolean;
  live_calls: number;
  cache_hits: number;
  cached_responses: number;
  last_error: string | null;
}

export interface Health {
  status: string;
  version: string;
  gemma: GemmaStats;
  replay_size: number;
  reasoned: boolean;
  confidence_floor: number;
  driftguard_threshold: number;
}

export interface ApiKey {
  id: number;
  key_hint: string;
  label: string;
  created_at: string;
  last_used: string | null;
  revoked: number;
}

export interface CreatedKey {
  id: number;
  key: string;
  key_hint: string;
  label: string;
  created_at: string;
  warning: string;
  setup: { DRIFTGUARD_API_URL: string; DRIFTGUARD_API_KEY: string };
}

export interface LiveModel {
  model_id: string;
  features: string[];
  drift_threshold: number | null;
  version: string | null;
  accuracy: number | null;
  registered_at: string;
  last_seen: string | null;
  telemetry_count: number;
  event_count: number;
}

export interface TelemetryRow {
  id: number;
  model_id: string;
  received_at: string;
  drift_score: number | null;
  features: number[] | null;
  prediction: unknown;
}

export interface Answer {
  question: string;
  answer: string;
  events_considered: number;
  source: string;
}

export interface Result<T> {
  data: T | null;
  error: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<Result<T>> {
  try {
    const res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      cache: "no-store",
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        if (body?.detail) detail = String(body.detail);
      } catch {
        // non-JSON error body; the status line is enough
      }
      return { data: null, error: detail };
    }
    return { data: (await res.json()) as T, error: null };
  } catch (err) {
    return {
      data: null,
      error:
        err instanceof Error && err.message.includes("fetch")
          ? `Cannot reach the Arbiter API at ${API_URL}`
          : err instanceof Error
            ? err.message
            : "Unknown error",
    };
  }
}

export const api = {
  health: () => request<Health>("/health"),
  metrics: () => request<Metrics>("/api/metrics"),
  events: () => request<{ events: EventCard[]; count: number }>("/api/events"),
  eventDetail: (id: string) => request<EventDetail>(`/api/events/${id}`),
  ask: (question: string) =>
    request<Answer>("/api/ask", { method: "POST", body: JSON.stringify({ question }) }),
  askExamples: () => request<{ examples: string[] }>("/api/ask/examples"),

  // Keys
  keys: () => request<{ keys: ApiKey[] }>("/api/keys"),
  createKey: (label: string) =>
    request<CreatedKey>("/api/keys", { method: "POST", body: JSON.stringify({ label }) }),
  revokeKey: (id: number) =>
    request<{ status: string }>(`/api/keys/${id}`, { method: "DELETE" }),

  // Live SDK data
  models: () => request<{ models: LiveModel[]; count: number }>("/api/models"),
  telemetry: (modelId?: string, limit = 100) =>
    request<{ telemetry: TelemetryRow[]; total: number }>(
      `/api/telemetry?limit=${limit}${modelId ? `&model_id=${encodeURIComponent(modelId)}` : ""}`,
    ),
  liveEvents: (modelId?: string) =>
    request<{ events: EventCard[]; count: number }>(
      `/api/live/events${modelId ? `?model_id=${encodeURIComponent(modelId)}` : ""}`,
    ),
  liveEventDetail: (id: string) => request<EventDetail>(`/api/live/events/${id}`),
};

// ---- presentation helpers -------------------------------------------------

export const ACTION_COLOR: Record<Action, string> = {
  RETRAIN: "#dc2626",
  ROLLBACK: "#ea580c",
  INVESTIGATE: "#ca8a04",
  BENIGN: "#64748b",
};

export const TAXONOMY_LABEL: Record<string, string> = {
  covariate_shift: "Covariate shift",
  label_shift: "Label shift",
  concept_drift: "Concept drift",
  schema_break: "Schema break",
  seasonal: "Seasonal",
  upstream_bug: "Upstream bug",
};

/** Compact number for metric tiles: 1.2k rather than 1200. */
export function compact(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

export function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}
