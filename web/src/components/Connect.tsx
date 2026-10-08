// Tab 4 — Connect. The onboarding flow, and the answer to "where do I get a key?"
//
// The key is minted here, in this dashboard. Nothing is requested from
// DriftGuard or any third party: Arbiter implements the SDK's wire protocol,
// so pointing DRIFTGUARD_API_URL at Arbiter is the entire integration.
//
// The plaintext key exists only in the create response, so it is shown once
// with a copy button and never fetched again.
"use client";

import { useCallback, useEffect, useState } from "react";
import {
  API_URL,
  type ApiKey,
  type CreatedKey,
  type LiveModel,
  api,
  relativeTime,
} from "@/lib/api";
import { Button, Card, CodeBlock, EmptyState, ErrorState, Skeleton } from "./ui";

export function Connect() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [models, setModels] = useState<LiveModel[]>([]);
  const [fresh, setFresh] = useState<CreatedKey | null>(null);
  const [label, setLabel] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [k, m] = await Promise.all([api.keys(), api.models()]);
    if (k.error) setError(k.error);
    else {
      setError(null);
      setKeys(k.data?.keys ?? []);
      setModels(m.data?.models ?? []);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    // Poll so a model connecting from the user's own code appears without a
    // manual refresh — that moment is the payoff of this whole tab.
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
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
    } else {
      setError(res.error);
    }
    setBusy(false);
  };

  const revoke = async (id: number) => {
    setBusy(true);
    await api.revokeKey(id);
    await load();
    setBusy(false);
  };

  if (error && !keys.length) return <ErrorState error={error} onRetry={load} />;

  const snippet = (key: string) => `# 1. Point the DriftGuard SDK at Arbiter
export DRIFTGUARD_API_URL="${API_URL}"
export DRIFTGUARD_API_KEY="${key}"

# 2. Your existing code needs no changes
from driftguard import DriftGuard

dg = DriftGuard(
    model_id="fraud-detector-v2",
    drift_threshold=0.15,
    api_key="${key}",          # or read from DRIFTGUARD_API_KEY
)
model = dg.wrap(my_model, feature_extractor=lambda x: x)

# 3. Predict as usual. Telemetry streams to Arbiter.
model.predict(features)

# 4. Let Arbiter decide whether a retrain is warranted
@dg.retrainer
def retrain():
    return train_new_model()`;

  return (
    <div className="flex flex-col gap-6">
      {/* ---- Step 1: key ---- */}
      <Card>
        <div className="border-b border-border px-5 py-3">
          <h3 className="text-sm font-medium">1 · Generate an Arbiter API key</h3>
          <p className="pt-0.5 text-xs text-muted">
            Issued here. You do not need a key from DriftGuard or anyone else — Arbiter
            implements the SDK&apos;s protocol directly.
          </p>
        </div>

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
          ) : keys.length > 0 ? (
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
                    <td className="py-2 text-subtle">
                      {k.last_used ? relativeTime(k.last_used) : "never"}
                    </td>
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
              No keys yet. Until one exists, Arbiter accepts unauthenticated telemetry so
              you can try it immediately.
            </p>
          )}
        </div>
      </Card>

      {/* ---- Step 2: wire it up ---- */}
      <Card>
        <div className="border-b border-border px-5 py-3">
          <h3 className="text-sm font-medium">2 · Paste it into your code</h3>
          <p className="pt-0.5 text-xs text-muted">
            Two environment variables. No SDK fork, no vendored source.
          </p>
        </div>
        <div className="px-5 py-4">
          <CodeBlock
            code={snippet(fresh?.key ?? "ak_live_your_key_here")}
            onCopy={() => copy(snippet(fresh?.key ?? "ak_live_your_key_here"), "snippet")}
          />
          {copied === "snippet" && (
            <p className="pt-2 text-2xs text-muted">Copied to clipboard.</p>
          )}
        </div>
      </Card>

      {/* ---- Step 3: watch it arrive ---- */}
      <Card>
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <div>
            <h3 className="text-sm font-medium">3 · Connected models</h3>
            <p className="pt-0.5 text-xs text-muted">
              Updates every 5 seconds as telemetry arrives.
            </p>
          </div>
          <span className="flex items-center gap-1.5 text-2xs text-subtle">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-fg" />
            watching
          </span>
        </div>

        {models.length === 0 ? (
          <EmptyState
            title="No models connected yet"
            body="Run your code with the two environment variables above. The model registers itself on the first prediction and appears here."
          />
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border text-left text-2xs uppercase tracking-wider text-subtle">
                <th className="px-5 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">Features</th>
                <th className="px-3 py-2 text-right font-medium">Telemetry</th>
                <th className="px-3 py-2 text-right font-medium">Events</th>
                <th className="px-5 py-2 text-right font-medium">Last seen</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.model_id} className="border-b border-border last:border-0">
                  <td className="px-5 py-2.5 font-mono font-medium text-fg">{m.model_id}</td>
                  <td className="px-3 py-2.5 text-muted">
                    {m.features.length > 0 ? (
                      <span title={m.features.join(", ")}>{m.features.length} named</span>
                    ) : (
                      <span className="text-investigate" title="Narrations cannot name features">
                        none registered
                      </span>
                    )}
                  </td>
                  <td className="nums px-3 py-2.5 text-right text-muted">
                    {m.telemetry_count.toLocaleString()}
                  </td>
                  <td className="nums px-3 py-2.5 text-right text-muted">{m.event_count}</td>
                  <td className="px-5 py-2.5 text-right text-subtle">
                    {m.last_seen ? relativeTime(m.last_seen) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
