from __future__ import annotations

import argparse
import os

from agent import TRACE_FILE, run_agent
from coach import CHECK_NAMES, compute_features, evaluate_checks, todays_conflicts
from compare_models import scenarios
from pawpal_system import Scheduler
from retriever import format_notes

DEMO_SCENARIOS = ["Calm day, one pet", "Medication moved by a walk", "Task with no room today"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--traces", action="store_true",
                        help="append each run's reasoning trace to ai_interactions.md")
    args = parser.parse_args(argv)

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("NOTE: GEMINI_API_KEY is not set, so the coach will use its built-in fallback.\n")

    owners = dict(scenarios())
    for name in DEMO_SCENARIOS:
        owner = owners[name]
        scheduler = Scheduler(owner)
        plan = scheduler.build_daily_plan()
        conflicts = todays_conflicts(scheduler.detect_conflicts())
        result = run_agent(
            owner, plan, conflicts, trace_path=TRACE_FILE if args.traces else None
        )

        # Re-score the final answer. The fallback is deterministic and never cites
        # a note, so the citation check is skipped for it.
        sources = None if result.source == "fallback" else [c.source for c in result.notes]
        checks = evaluate_checks(
            result.text, compute_features(plan, conflicts), result.plan_text, sources
        )
        passed = sum(1 for problem in checks.values() if problem is None)
        caught = "; ".join(result.problems) or "none"

        print(f"=== Scenario: {name} ===\n")
        print("INPUT (today's plan from the Scheduler):")
        print(result.plan_text, "\n")
        print("CARE NOTES RETRIEVED:")
        print(format_notes(result.notes), "\n")
        print("COACH ANSWER:")
        print(result.text, "\n")
        print(
            f"SOURCE: {result.source} | MODEL CALLS: {result.attempts} | "
            f"RULE VIOLATIONS CAUGHT: {caught} | "
            f"GUARDRAIL CHECKS PASSED: {passed}/{len(CHECK_NAMES)}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
