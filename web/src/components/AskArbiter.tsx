// Tab 3 — Ask Arbiter (UC 11).
//
// A judge gets to type their own question. The seeded chips exist so the tab
// never starts empty, but the real value is the free-text box: an answer to an
// unrehearsed question is worth more than any scripted demo beat.
"use client";

import { useEffect, useRef, useState } from "react";
import { type Answer, api } from "@/lib/api";
import { Button, Card, Reasoning } from "./ui";

interface Turn {
  question: string;
  answer: string | null;
  source: string | null;
  error?: string;
}

export function AskArbiter() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [examples, setExamples] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.askExamples().then((r) => setExamples(r.data?.examples ?? []));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, busy]);

  const submit = async (q: string) => {
    const text = q.trim();
    if (!text || busy) return;

    setQuestion("");
    setBusy(true);
    setTurns((prev) => [...prev, { question: text, answer: null, source: null }]);

    const res = await api.ask(text);
    setTurns((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (res.data) {
        last.answer = res.data.answer;
        last.source = res.data.source;
      } else {
        last.error = res.error ?? "Request failed";
      }
      return next;
    });
    setBusy(false);
  };

  return (
    <div className="flex flex-col gap-4">
      <Card className="flex flex-col">
        <div className="border-b border-border px-5 py-3">
          <h3 className="text-sm font-medium">Ask Arbiter</h3>
          <p className="pt-0.5 text-xs text-muted">
            Natural-language questions over the drift telemetry. Answers come only from the
            events Arbiter has reasoned about.
          </p>
        </div>

        <div className="flex min-h-[280px] flex-col gap-5 px-5 py-5">
          {turns.length === 0 && !busy && (
            <div className="flex flex-col gap-3">
              <span className="text-2xs uppercase tracking-wider text-subtle">
                Try one of these
              </span>
              <div className="flex flex-col items-start gap-2">
                {examples.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => submit(ex)}
                    className="rounded-md border border-border bg-surface px-3 py-1.5 text-left text-xs text-muted transition-colors hover:border-borderStrong hover:text-fg"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((turn, i) => (
            <div key={i} className="flex animate-fade-in flex-col gap-2">
              <p className="text-sm font-medium text-fg">{turn.question}</p>
              {turn.error ? (
                <p className="text-sm text-retrain">{turn.error}</p>
              ) : turn.answer === null ? (
                <Reasoning />
              ) : (
                <>
                  <p className="text-sm leading-relaxed text-muted">{turn.answer}</p>
                  {turn.source && (
                    <span className="font-mono text-2xs text-subtle">{turn.source}</span>
                  )}
                </>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>

        <div className="flex gap-2 border-t border-border px-5 py-4">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit(question)}
            placeholder="Which models degraded this week and why?"
            disabled={busy}
            className="flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm placeholder:text-subtle focus:border-borderStrong focus:outline-none disabled:opacity-50"
          />
          <Button onClick={() => submit(question)} disabled={busy || !question.trim()}>
            {busy ? "Reasoning…" : "Ask"}
          </Button>
        </div>
      </Card>

      <p className="px-1 text-2xs leading-relaxed text-subtle">
        Arbiter answers only from telemetry it holds. Asked something the data cannot
        support, it says so rather than estimating.
      </p>
    </div>
  );
}
