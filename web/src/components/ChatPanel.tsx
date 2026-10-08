// Chat panel — used for both the per-model and the fleet chatbot.
//
// One component, two scopes: `modelId` decides which endpoint is called and
// therefore which context the model is given. The distinction matters, so the
// header states it rather than leaving the user to guess what the bot can see.
"use client";

import { useEffect, useRef, useState } from "react";
import { api, type ChatReply } from "@/lib/api";

interface Turn {
  role: "user" | "assistant";
  content: string;
  source?: string;
  pending?: boolean;
  error?: boolean;
}

export function ChatPanel({
  modelId,
  title,
  subtitle,
  className = "",
}: {
  modelId?: string;
  title: string;
  subtitle: string;
  className?: string;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  // Switching models must not carry the previous model's conversation over —
  // the context changed, so the history would be misleading.
  useEffect(() => {
    setTurns([]);
    setDraft("");
    setSuggestions(
      modelId
        ? [
            "Why did this model drift?",
            "Should I retrain it?",
            "Is this a pipeline bug or a real shift?",
          ]
        : [
            "Which models degraded and why?",
            "Do any of these share a root cause?",
            "Where should I look first?",
          ],
    );
  }, [modelId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const send = async (text: string) => {
    const question = text.trim();
    if (!question || busy) return;

    // Send prior turns so follow-ups ("and the second one?") resolve.
    const history = turns
      .filter((t) => !t.pending && !t.error)
      .map((t) => ({ role: t.role, content: t.content }));

    setDraft("");
    setBusy(true);
    setTurns((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", pending: true },
    ]);

    const res = modelId
      ? await api.chatModel(modelId, question, history)
      : await api.chatFleet(question, history);

    setTurns((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (res.data) {
        last.content = res.data.answer;
        last.source = res.data.source;
      } else {
        last.content = res.error ?? "Request failed.";
        last.error = true;
      }
      last.pending = false;
      return next;
    });
    setBusy(false);
  };

  return (
    <div
      className={`flex flex-col overflow-hidden rounded-lg border border-border bg-bg ${className}`}
    >
      <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-medium">{title}</h3>
          <p className="truncate text-2xs text-muted">{subtitle}</p>
        </div>
        {turns.length > 0 && (
          <button
            onClick={() => setTurns([])}
            className="shrink-0 text-2xs text-subtle transition-colors hover:text-fg"
          >
            Clear
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {turns.length === 0 ? (
          <div className="flex flex-col gap-2">
            <span className="text-2xs uppercase tracking-wider text-subtle">Try asking</span>
            {suggestions.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="rounded-md border border-border bg-surface px-3 py-2 text-left text-xs text-muted transition-colors hover:border-borderStrong hover:text-fg"
              >
                {s}
              </button>
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {turns.map((t, i) =>
              t.role === "user" ? (
                <div key={i} className="flex justify-end">
                  <p className="max-w-[85%] rounded-lg rounded-br-sm bg-fg px-3 py-2 text-xs leading-relaxed text-bg">
                    {t.content}
                  </p>
                </div>
              ) : (
                <div key={i} className="flex flex-col gap-1">
                  {t.pending ? (
                    <span className="flex items-center gap-2 text-xs text-muted">
                      <span className="relative flex h-1.5 w-1.5">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-fg opacity-60" />
                        <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-fg" />
                      </span>
                      Gemma is reasoning…
                    </span>
                  ) : (
                    <>
                      <p
                        className={`whitespace-pre-wrap text-xs leading-relaxed ${
                          t.error ? "text-retrain" : "text-fg"
                        }`}
                      >
                        {t.content}
                      </p>
                      {t.source && (
                        <span className="font-mono text-2xs text-subtle">{t.source}</span>
                      )}
                    </>
                  )}
                </div>
              ),
            )}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <div className="flex gap-2 border-t border-border px-4 py-3">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(draft)}
          placeholder={modelId ? "Ask about this model…" : "Ask about your models…"}
          disabled={busy}
          className="min-w-0 flex-1 rounded-md border border-border bg-bg px-3 py-2 text-xs placeholder:text-subtle focus:border-borderStrong focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={() => send(draft)}
          disabled={busy || !draft.trim()}
          className="shrink-0 rounded-md border border-fg bg-fg px-3 py-2 text-xs font-medium text-bg transition-colors hover:bg-[#333] disabled:opacity-40"
        >
          {busy ? "…" : "Ask"}
        </button>
      </div>
    </div>
  );
}
