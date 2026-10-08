// Typed client for the Arbiter API.
//
// The session token lives in localStorage and is attached as a bearer header
// by `request`, so no call site has to remember it. A 401 clears the token and
// notifies listeners, which is what bounces the UI back to the sign-in screen
// when a session expires mid-session.
//
// Nothing here throws on a network failure: the dashboard must stay legible
// when the backend is down, so callers get `{ data: null, error }` and render
// an explicit offline state.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

const TOKEN_KEY = "arbiter.session";

type Listener = () => void;
const authListeners = new Set<Listener>();

export function onAuthChange(fn: Listener): () => void {
  authListeners.add(fn);
  return () => authListeners.delete(fn);
}

function notifyAuth() {
  authListeners.forEach((fn) => fn());
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  if (typeof window === "undefined") return;
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Private browsing can throw on write; the session simply won't persist.
  }
  notifyAuth();
}

// ---- types ---------------------------------------------------------------

export type Action = "RETRAIN" | "ROLLBACK" | "INVESTIGATE" | "BENIGN";

export type Taxonomy =
  | "covariate_shift"
  | "label_shift"
  | "concept_drift"
  | "schema_break"
  | "seasonal"
  | "upstream_bug";

export interface User {
  id: number;
  email: string;
  name: string | null;
  created_at?: string;
  last_login?: string | null;
}

export interface Session {
  token: string;
  user: User;
}

export interface FeatureScore {
  name: string;
  score: number;
}

export interface EventCard {
  event_id: string;
  model_id: string;
  timestamp: string;
  global_drift_score: number | null;
  action: Action | null;
  confidence: number;
  taxonomy: Taxonomy | null;
  reasoning: string | null;
  primary_features: string[];
  top_features: FeatureScore[];
  suppressed: boolean;
  driftguard_would_page: boolean;
  needs_human: boolean;
  source: string | null;
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

export interface EventDetail extends EventCard {
  features: FeatureDetail[];
  context: Record<string, unknown>;
  analysis: Analysis | null;
  quality_bar: Record<string, boolean> | null;
  challenger?: ChallengerVerdict;
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
  actionable: number;
  suppressed: number;
  latest_verdict: { action?: Action; confidence?: number; taxonomy?: string } | null;
  latest_drift_score: number | null;
}

export interface ModelMetrics {
  model_id: string;
  features: string[];
  drift_threshold: number;
  version: string | null;
  reported_accuracy: number | null;
  telemetry_count: number;
  alerts_fired: number;
  actionable: number;
  suppressed: number;
  noise_reduction_pct: number;
  escalated_to_human: number;
  mean_confidence: number | null;
  by_action: Record<string, number>;
  by_taxonomy: Record<string, number>;
  gemma_backed_decisions: number;
  reportable: boolean;
  drift_series: { drift_score: number; received_at: string }[];
  latest_drift_score: number | null;
}

export interface TelemetryRow {
  id: number;
  model_id: string;
  received_at: string;
  drift_score: number | null;
  features: number[] | null;
  prediction: unknown;
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
  warning: string;
  setup: { DRIFTGUARD_API_URL: string; DRIFTGUARD_API_KEY: string };
}

export interface ModelDoc {
  model_id: string;
  markdown: string;
  generated_at: string;
  incidents: number;
  source: string;
}

export interface ChatReply {
  answer: string;
  scope: "model" | "fleet";
  model_id: string | null;
  context_events: number;
  source: string;
  suggestions: string[];
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
  confidence_floor: number;
  driftguard_threshold: number;
  has_users: boolean;
}

export interface Result<T> {
  data: T | null;
  error: string | null;
}

// ---- transport -----------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<Result<T>> {
  const token = getToken();
  try {
    const res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init?.headers || {}),
      },
      cache: "no-store",
    });

    // A 401 from an auth endpoint means "those credentials are wrong", not
    // "your session expired" — rewriting it hid the real message behind a
    // confusing one on the sign-in screen, which read as registration being
    // broken. Only treat a 401 as an expired session on an authenticated
    // route, where a token was actually sent.
    if (res.status === 401 && !path.startsWith("/auth/") && token) {
      setToken(null);
      return { data: null, error: "Your session expired. Sign in again." };
    }

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        if (body?.detail) detail = String(body.detail);
      } catch {
        // non-JSON body; the status is enough
      }
      return { data: null, error: detail };
    }
    return { data: (await res.json()) as T, error: null };
  } catch (err) {
    return {
      data: null,
      error:
        err instanceof Error && /fetch|network/i.test(err.message)
          ? `Cannot reach the Arbiter API at ${API_URL}`
          : err instanceof Error
            ? err.message
            : "Unknown error",
    };
  }
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });

export const api = {
  health: () => request<Health>("/health"),

  // auth
  register: (email: string, password: string, name?: string) =>
    post<Session>("/auth/register", { email, password, name }),
  login: (email: string, password: string) => post<Session>("/auth/login", { email, password }),
  logout: () => post<{ status: string }>("/auth/logout"),
  me: () => request<{ user: User }>("/auth/me"),
  changeEmail: (new_email: string, password: string) =>
    request<{ user: User }>("/auth/email", {
      method: "PATCH",
      body: JSON.stringify({ new_email, password }),
    }),
  changePassword: (current_password: string, new_password: string) =>
    request<{ status: string; reauth_required: boolean }>("/auth/password", {
      method: "PATCH",
      body: JSON.stringify({ current_password, new_password }),
    }),

  // keys
  keys: () => request<{ keys: ApiKey[] }>("/api/keys"),
  createKey: (label: string) => post<CreatedKey>("/api/keys", { label }),
  revokeKey: (id: number) => request<{ status: string }>(`/api/keys/${id}`, { method: "DELETE" }),

  // the user's models
  models: () => request<{ models: LiveModel[]; count: number }>("/api/models"),
  modelMetrics: (id: string) =>
    request<ModelMetrics>(`/api/models/${encodeURIComponent(id)}/metrics`),
  modelEvents: (id: string) =>
    request<{ events: EventCard[]; count: number }>(
      `/api/models/${encodeURIComponent(id)}/events`,
    ),
  modelTelemetry: (id: string, limit = 100) =>
    request<{ telemetry: TelemetryRow[]; total: number }>(
      `/api/models/${encodeURIComponent(id)}/telemetry?limit=${limit}`,
    ),
  liveEventDetail: (id: string) => request<EventDetail>(`/api/live/events/${id}`),
  modelDocs: (id: string) =>
    request<ModelDoc>(`/api/models/${encodeURIComponent(id)}/docs`),

  // chat
  chatModel: (id: string, question: string, history: { role: string; content: string }[] = []) =>
    post<ChatReply>(`/api/models/${encodeURIComponent(id)}/chat`, { question, history }),
  chatFleet: (question: string, history: { role: string; content: string }[] = []) =>
    post<ChatReply>("/api/chat", { question, history }),

  // demo corpus
  demoEvents: () => request<{ events: EventCard[]; count: number }>("/api/events"),
  demoEventDetail: (id: string) => request<EventDetail>(`/api/events/${id}`),
};

// ---- presentation --------------------------------------------------------

export const ACTION_COLOR: Record<string, string> = {
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
  undetermined: "Undetermined",
};

export function relativeTime(iso: string | null): string {
  if (!iso) return "never";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** Adaptive precision — a ratio feature printed as 0.3 hides its movement. */
export function fmtNum(n: number): string {
  const a = Math.abs(n);
  if (a >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 10) return n.toFixed(1);
  if (a >= 1) return n.toFixed(2);
  return n.toFixed(3);
}
