"""Replay runner — produces the measured numbers the README cites.

Run it, read the numbers off stdout, paste them into the README. That is the
whole workflow, and it is the reason no number in this project is hardcoded
(AGENTS.md §2).

    python -m arbiter.run_replay              # 20 events, prints metrics
    python -m arbiter.run_replay --verbose    # plus every narration
    python -m arbiter.run_replay --json out.json

It also doubles as the demo pre-warm: every response lands in the cache, so a
subsequent run — or a live demo — replays offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .evaluator import decide, measure_accuracy, measure_suppression
from .gemma_client import get_client
from .reasoner import meets_quality_bar
from .scenarios import generate_replay, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Arbiter replay corpus.")
    parser.add_argument("-n", "--events", type=int, default=20, help="corpus size")
    parser.add_argument("--seed", type=int, default=20261008, help="corpus seed")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every narration")
    parser.add_argument("--json", metavar="PATH", help="write full results to PATH")
    args = parser.parse_args(argv)

    client = get_client()
    print(f"model: {client.model}")
    print(f"gemma available: {client.available}")
    if not client.available:
        print(
            "\n  WARNING: no usable GEMMA_API_KEY. Verdicts will come from the\n"
            "  heuristic fallback and are NOT reportable results. Set the key in\n"
            "  .env and re-run before quoting any number from this output.\n"
        )

    events = generate_replay(args.events, seed=args.seed)
    print(f"\ncorpus: {json.dumps(summarize(events))}\n")

    decisions = []
    for i, event in enumerate(events, 1):
        decision = decide(event, client)
        decisions.append(decision)
        v = decision.verdict
        flag = "SUPPRESSED" if v.suppressed else "ACT"
        human = " ESCALATED" if decision.needs_human else ""
        print(
            f"[{i:2}/{len(events)}] {event.event_id} {event.model_id:20} "
            f"{v.action:12} {v.confidence:.2f} {str(v.taxonomy):16} "
            f"(truth: {str(event.ground_truth):16}) {flag}{human}"
        )
        if args.verbose:
            print(f"         {v.reasoning}")
            if decision.analysis:
                bar = meets_quality_bar(decision.analysis, event)
                print(f"         quality bar: {bar}")
            print()

    suppression = measure_suppression(decisions)
    accuracy = measure_accuracy(decisions, events)

    print("\n" + "=" * 68)
    print("MEASURED RESULTS" if suppression["reportable"] else "UNREPORTABLE RUN (heuristic fallback)")
    print("=" * 68)
    print(f"  Alerts fired (DriftGuard threshold) : {suppression['alerts_fired']}")
    print(f"  Actionable                          : {suppression['actionable']}")
    print(f"  Suppressed                          : {suppression['suppressed']}")
    print(f"  Noise reduction                     : {suppression['noise_reduction_pct']}%")
    print(f"  Escalated to human                  : {suppression['escalated_to_human']}")
    print(f"  By action                           : {suppression['by_action']}")
    if accuracy["taxonomy_accuracy_pct"] is not None:
        print(
            f"  Taxonomy accuracy                   : "
            f"{accuracy['taxonomy_correct']}/{accuracy['labelled_events']} "
            f"({accuracy['taxonomy_accuracy_pct']}%)"
        )
    else:
        print(f"  Taxonomy accuracy                   : not measurable — {accuracy.get('note', '')}")

    # Quality bar across the run — the differentiator, measured not assumed.
    bars = [
        meets_quality_bar(d.analysis, e)
        for d, e in zip(decisions, events)
        if d.analysis and d.analysis.source.startswith("gemma")
    ]
    if bars:
        print("\n  Narration quality bar (Gemma-backed narrations only):")
        for criterion in bars[0]:
            hits = sum(1 for b in bars if b[criterion])
            print(f"    {criterion:26} {hits}/{len(bars)}")

    print(f"\n  gemma: {json.dumps(client.stats())}")
    print("=" * 68)

    if not suppression["reportable"]:
        print(
            "\n  Do not quote these numbers. They measure the heuristic fallback,\n"
            "  not Gemma's reasoning (AGENTS.md §2)."
        )

    if args.json:
        payload: dict[str, Any] = {
            "model": client.model,
            "corpus": summarize(events),
            "suppression": suppression,
            "accuracy": accuracy,
            "decisions": [d.to_dict() for d in decisions],
            "gemma": client.stats(),
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
