"""Test harness for the PawPal Coach.

Part A: guardrail suite (OFFLINE, deterministic, no API key needed)
    Runs the real agent against a scripted "model" that misbehaves in 12
    specific ways (dosage language, invented times, invented sources, ...).
    Each case states what the guardrail SHOULD do and the harness checks it.

Part B: live scenarios (needs GEMINI_API_KEY and google-genai)
    Runs the full agent with real Gemini on the 8 scenarios from
    compare_models.py and reports how often Gemini passed the guardrails.

Usage:
    python evaluate.py                       # Part A, then Part B if a key is set
    python evaluate.py --offline             # Part A only
    python evaluate.py --out results/evaluation.md
    python evaluate.py --traces              # also append Part B traces to ai_interactions.md

Exit code is 1 if any Part A expectation fails, so it can run in CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent import TRACE_FILE, AgentResult, run_agent
from coach import CHECK_NAMES, CoachError, fallback_explanation, gemini_llm, todays_conflicts
from compare_models import SAMPLE_SCENARIO, scenarios
from pawpal_system import Owner, Scheduler
from retriever import retrieve


# ----------------------------------------------------------------------
# A scripted model for Part A
# ----------------------------------------------------------------------

def scripted_llm(replies: list):
    """Return a fake model that answers from a list (an Exception item is raised)."""
    queue = list(replies)

    def llm(system: str, user: str) -> str:
        if not queue:
            raise CoachError("script exhausted")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return llm


@dataclass
class Case:
    name: str
    what_goes_wrong: str
    replies: list  # what the scripted model says, in order
    expect_source: str
    expect_attempts: int
    expect_caught: Optional[str]  # text that must appear in the caught problems
    empty_plan: bool = False


def build_cases(good: str) -> list[Case]:
    """Define the 12 guardrail cases. `good` is a reply that passes every check."""
    return [
        Case("Clean reply", "nothing", [good], "gemini", 1, None),
        Case("Dosage language", "reply gives a dose amount",
             [good + " Give 5 mg twice daily.", good], "gemini", 2, "diagnosis or dosage"),
        Case("Invented time", "reply mentions a time that is not in the plan",
             [good + " Plan something at 23:59.", good], "gemini", 2, "not in the plan"),
        Case("Invented source", "reply cites a document that was never retrieved",
             [good + " [cats.md]", good], "gemini", 2, "not provided"),
        Case("Missing citation", "Tip is not grounded in a care note",
             [_strip_citation(good), good], "gemini", 2, "not grounded"),
        Case("No vet mention", "medication day without mentioning the vet",
             [good.replace("your vet", "a friend"), good], "gemini", 2, "mention the vet"),
        Case("Markdown and bullets", "reply uses markdown formatting",
             ["**" + good + "**", good], "gemini", 2, "markdown"),
        Case("Too long", "reply is over 120 words",
             [good + " word" * 130, good], "gemini", 2, "too long"),
        Case("Wrong format", "reply ignores the required format",
             ["Here is your schedule for today!", good], "gemini", 2, "does not start"),
        Case("Never complies", "model keeps ignoring the rules",
             ["no format", "still no format"], "fallback", 2, "does not start"),
        Case("Model outage", "Gemini raises an error",
             [CoachError("simulated outage")], "fallback", 1, None),
        Case("Empty plan", "nothing to explain, so no model call",
             [], "guardrail", 0, None, empty_plan=True),
    ]


def _strip_citation(text: str) -> str:
    return re.sub(r"\s*\[[\w.\-]+\.md\]", "", text)


# ----------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------

@dataclass
class CaseResult:
    case: Case
    result: AgentResult
    passed: bool
    note: str


def run_guardrail_suite() -> list[CaseResult]:
    """Run every guardrail case through the real agent and compare with expectations."""
    owner = dict(scenarios())[SAMPLE_SCENARIO]
    scheduler = Scheduler(owner)
    plan = scheduler.build_daily_plan()
    conflicts = todays_conflicts(scheduler.detect_conflicts())
    notes = retrieve(owner, plan)
    good = fallback_explanation(plan, conflicts, cite=notes[0].source if notes else None)

    results = []
    for case in build_cases(good):
        if case.empty_plan:
            result = run_agent(Owner("Tester"), [], [], llm=scripted_llm(case.replies))
        else:
            result = run_agent(owner, plan, conflicts, llm=scripted_llm(case.replies))

        problems = []
        if result.source != case.expect_source:
            problems.append(f"source {result.source!r}, expected {case.expect_source!r}")
        if result.attempts != case.expect_attempts:
            problems.append(f"{result.attempts} model calls, expected {case.expect_attempts}")
        caught = "; ".join(result.problems)
        if case.expect_caught and case.expect_caught not in caught:
            problems.append(f"did not catch {case.expect_caught!r}")
        if case.expect_caught is None and result.problems:
            problems.append(f"unexpected problems: {caught}")
        results.append(CaseResult(case, result, not problems, "; ".join(problems) or "ok"))
    return results


def render_suite(results: list[CaseResult]) -> str:
    """Markdown table for Part A."""
    passed = sum(r.passed for r in results)
    lines = [
        "## Part A: guardrail suite (scripted model, no network)\n",
        "| # | Case | What goes wrong | Guardrail outcome | Calls | Result |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        outcome = {"gemini": "caught, retried, accepted" if r.result.problems
                   else "accepted first try",
                   "fallback": "fell back to built-in explanation",
                   "guardrail": "short-circuited"}[r.result.source]
        lines.append(
            f"| {i} | {r.case.name} | {r.case.what_goes_wrong} | {outcome} | "
            f"{r.result.attempts} | {'PASS' if r.passed else 'FAIL: ' + r.note} |"
        )
    lines.append(f"\n**Guardrail suite: {passed}/{len(results)} expectations met.**\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------

def run_live(save_traces: bool) -> list[tuple[str, AgentResult]]:
    """Run the full agent with real Gemini on every scenario."""
    runs = []
    for name, owner in scenarios():
        scheduler = Scheduler(owner)
        plan = scheduler.build_daily_plan()
        conflicts = todays_conflicts(scheduler.detect_conflicts())
        result = run_agent(
            owner, plan, conflicts, llm=gemini_llm,
            trace_path=TRACE_FILE if save_traces else None,
        )
        runs.append((name, result))
    return runs


def render_live(runs: list[tuple[str, AgentResult]]) -> str:
    """Markdown table and summary for Part B."""
    lines = [
        "## Part B: live Gemini through the full agent\n",
        "| Scenario | Answer source | Model calls | Rule violations caught | Words |",
        "|---|---|---|---|---|",
    ]
    for name, r in runs:
        caught = "; ".join(r.problems) or "none"
        lines.append(f"| {name} | {r.source} | {r.attempts} | {caught} | {len(r.text.split())} |")
    n = len(runs)
    gemini = sum(1 for _, r in runs if r.source == "gemini")
    first_try = sum(1 for _, r in runs if r.source == "gemini" and r.attempts == 1)
    fallback = sum(1 for _, r in runs if r.source == "fallback")
    avg_calls = sum(r.attempts for _, r in runs) / n
    lines.append(
        f"\n**Gemini passed all {len(CHECK_NAMES)} guardrail checks in {gemini}/{n} scenarios "
        f"({first_try} on the first try); {fallback} used the fallback; "
        f"average {avg_calls:.1f} model calls per scenario.**\n"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--offline", action="store_true", help="run Part A only")
    parser.add_argument("--traces", action="store_true",
                        help="append Part B agent traces to ai_interactions.md")
    parser.add_argument("--out", help="also write the markdown report to this file")
    args = parser.parse_args(argv)

    suite = run_guardrail_suite()
    report = render_suite(suite)

    if args.offline:
        report += "\nPart B skipped (--offline).\n"
    else:
        try:
            gemini_llm("You are a test.", "Say OK.")
        except CoachError as error:
            report += f"\nPart B skipped: cannot reach Gemini ({error}).\n"
        else:
            report += "\n" + render_live(run_live(args.traces))

    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Saved report to {args.out}")
    return 0 if all(r.passed for r in suite) else 1


if __name__ == "__main__":
    sys.exit(main())