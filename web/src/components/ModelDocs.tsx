// Gemma-written incident documentation for one model.
//
// Generated on demand rather than on page load: it costs a model call and
// takes a few seconds, so it runs when the user asks for it and the state of
// that request is made visible.
//
// The Markdown renderer here is deliberately small — headings, lists, code,
// bold and inline code are the whole vocabulary the report prompt asks for,
// and a full Markdown library for that would be a dependency for nothing
// (AGENTS.md §2).
"use client";

import { useState } from "react";
import { type ModelDoc, api } from "@/lib/api";
import { Button, Card, Reasoning } from "./ui";

export function ModelDocs({ modelId }: { modelId: string }) {
  const [doc, setDoc] = useState<ModelDoc | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const generate = async () => {
    setBusy(true);
    setError(null);
    const res = await api.modelDocs(modelId);
    setBusy(false);
    if (res.data) setDoc(res.data);
    else setError(res.error);
  };

  const copy = () => {
    if (!doc) return;
    navigator.clipboard?.writeText(doc.markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const download = () => {
    if (!doc) return;
    // A generated report is most useful in the repo or a ticket, so make it
    // a file rather than something to select and copy by hand.
    const blob = new Blob([doc.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${modelId}-incident-report.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3">
        <div>
          <h3 className="text-sm font-medium">Incident documentation</h3>
          <p className="text-2xs text-muted">
            {doc
              ? `${doc.incidents} incident(s) · generated ${new Date(
                  doc.generated_at,
                ).toLocaleString()} · ${doc.source}`
              : "A written report of every incident on this model, drafted by Gemma from stored telemetry"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {doc && (
            <>
              <Button size="sm" variant="ghost" onClick={copy}>
                {copied ? "Copied" : "Copy"}
              </Button>
              <Button size="sm" variant="ghost" onClick={download}>
                Download .md
              </Button>
            </>
          )}
          <Button size="sm" variant={doc ? "secondary" : "primary"} onClick={generate} disabled={busy}>
            {busy ? "Generating…" : doc ? "Regenerate" : "Generate report"}
          </Button>
        </div>
      </div>

      <div className="px-5 py-4">
        {busy && <Reasoning label="Gemma is writing the report" />}

        {error && !busy && (
          <p className="text-xs leading-relaxed text-retrain">{error}</p>
        )}

        {!busy && !error && !doc && (
          <p className="text-xs leading-relaxed text-muted">
            Generates a structured report — overview, incident history with the
            features and magnitudes involved, root-cause patterns, recommended
            actions, and what to watch next. Built from recorded telemetry only;
            no incident is invented.
          </p>
        )}

        {doc && !busy && <Markdown source={doc.markdown} />}
      </div>
    </Card>
  );
}

/** Minimal Markdown renderer covering what the report prompt produces. */
function Markdown({ source }: { source: string }) {
  const blocks: React.ReactNode[] = [];
  const lines = source.split("\n");
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code
    if (line.trimStart().startsWith("```")) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trimStart().startsWith("```")) {
        body.push(lines[i]);
        i++;
      }
      i++;
      blocks.push(
        <pre
          key={key++}
          className="overflow-x-auto rounded-md border border-border bg-surface px-3 py-2 font-mono text-2xs leading-relaxed text-muted"
        >
          {body.join("\n")}
        </pre>,
      );
      continue;
    }

    // Headings
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const depth = heading[1].length;
      const text = heading[2];
      const cls =
        depth <= 1
          ? "pt-1 text-base font-semibold tracking-tight"
          : depth === 2
            ? "border-t border-border pt-4 text-sm font-semibold"
            : "pt-2 font-mono text-xs font-medium text-fg";
      blocks.push(
        <h4 key={key++} className={cls}>
          {inline(text)}
        </h4>,
      );
      i++;
      continue;
    }

    // Blockquote (the fallback report uses one as its disclaimer)
    if (line.startsWith(">")) {
      const body: string[] = [];
      while (i < lines.length && lines[i].startsWith(">")) {
        body.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      blocks.push(
        <blockquote
          key={key++}
          className="border-l-2 border-borderStrong pl-3 text-xs italic leading-relaxed text-muted"
        >
          {inline(body.join(" "))}
        </blockquote>,
      );
      continue;
    }

    // Lists — ordered and unordered render the same way, with their marker.
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const items: string[] = [];
      const ordered = /^\s*\d+\./.test(line);
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ""));
        i++;
      }
      blocks.push(
        ordered ? (
          <ol key={key++} className="list-decimal space-y-1 pl-5 text-xs leading-relaxed text-fg">
            {items.map((t, n) => (
              <li key={n}>{inline(t)}</li>
            ))}
          </ol>
        ) : (
          <ul key={key++} className="list-disc space-y-1 pl-5 text-xs leading-relaxed text-fg">
            {items.map((t, n) => (
              <li key={n}>{inline(t)}</li>
            ))}
          </ul>
        ),
      );
      continue;
    }

    if (!line.trim()) {
      i++;
      continue;
    }

    // Paragraph: gather until a blank line or a new block starts.
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,4}\s|>|\s*([-*]|\d+\.)\s)/.test(lines[i]) &&
      !lines[i].trimStart().startsWith("```")
    ) {
      para.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={key++} className="text-xs leading-relaxed text-fg">
        {inline(para.join(" "))}
      </p>,
    );
  }

  return <div className="flex flex-col gap-3">{blocks}</div>;
}

/** Inline `code` and **bold**. */
function inline(text: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;

  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const token = m[0];
    if (token.startsWith("`")) {
      out.push(
        <code
          key={key++}
          className="rounded border border-border bg-surface px-1 py-0.5 font-mono text-[0.95em]"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else {
      out.push(
        <strong key={key++} className="font-semibold">
          {token.slice(2, -2)}
        </strong>,
      );
    }
    last = m.index + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
