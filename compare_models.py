"""Compare three versions of the coach on the same plans.

    baseline          generic prompt, no rules, no examples
    specialized       rules + few-shot examples (no care notes)
    specialized+RAG   rules + few-shot examples + retrieved care notes

All three get the SAME plan text for each scenario. Each reply is scored with
the same ten guardrail checks (coach.evaluate_checks), so differences are
measurable rather than a matter of opinion. The report also prints one scenario
side by side, which is the before/after example for the RAG stretch feature.

Usage:
    export GEMINI_API_KEY=your-key
    python compare_models.py             # real Gemini calls
    python compare_models.py --out results/comparison.md
    python compare_models.py --dry-run   # offline wiring check with a STUB model

--dry-run does not call any model. Its numbers are not real results and must
never be pasted into model_card.md.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from coach import (
    CHECK_NAMES,
    CoachError,
    baseline_explain,
    compute_features,
    evaluate_checks,
    gemini_llm,
    plan_to_text,
    specialized_explain,
    todays_conflicts,
)
from pawpal_system import Owner, Pet, Scheduler, Task
from retriever import format_notes, retrieve

VERSIONS = ["baseline", "specialized", "specialized+RAG"]
SAMPLE_SCENARIO = "Medication moved by a walk"


# ----------------------------------------------------------------------
# Test scenarios (different pets and times from the few-shot examples)
# ----------------------------------------------------------------------

def _owner(*pets: tuple[str, str, list[tuple]]) -> Owner:
    """Build an owner. Each task tuple: (description, time, duration, priority)."""
    owner = Owner("Naviyah")
    for name, species, tasks in pets:
        pet = Pet(name, species)
        for description, at, duration, priority in tasks:
            pet.add_task(Task(description, at, duration_minutes=duration, priority=priority))
        owner.add_pet(pet)
    return owner


def scenarios() -> list[tuple[str, Owner]]:
    """Return (name, owner) pairs covering normal, conflict, and edge cases."""
    return [
        ("Calm day, one pet", _owner(
            ("Daisy", "dog", [("Morning walk", "07:00", 30, "high"),
                              ("Lunch feeding", "12:00", 15, "medium"),
                              ("Evening walk", "18:30", 30, "medium")]))),
        ("Calm day, two pets", _owner(
            ("Milo", "cat", [("Breakfast", "07:30", 10, "high"),
                             ("Laser play", "19:00", 15, "low")]),
            ("Scout", "dog", [("Morning walk", "08:00", 30, "high")]))),
        ("Overlapping walks", _owner(
            ("Daisy", "dog", [("Morning walk", "08:00", 45, "high")]),
            ("Scout", "dog", [("Training", "08:30", 30, "medium")]))),
        (SAMPLE_SCENARIO, _owner(
            ("Oscar", "cat", [("Give medication", "08:15", 10, "high")]),
            ("Daisy", "dog", [("Morning walk", "08:00", 30, "high")]))),
        ("Task with no room today", _owner(
            ("Scout", "dog", [("Park trip", "17:00", 150, "high"),
                              ("Training", "19:30", 60, "high"),
                              ("Grooming", "18:30", 60, "low")]))),
        ("All low priority", _owner(
            ("Pebbles", "turtle", [("Feed turtle", "10:00", 10, "low"),
                                   ("Clean tank", "16:00", 40, "low")]))),
        ("Busy three-pet day", _owner(
            ("Daisy", "dog", [("Morning walk", "07:00", 30, "high"),
                              ("Training", "17:00", 30, "medium")]),
            ("Milo", "cat", [("Flea treatment", "07:15", 10, "high"),
                             ("Brushing", "20:00", 15, "low")]),
            ("Clover", "rabbit", [("Fresh hay and water", "07:30", 15, "medium")]))),
        ("Single quick task", _owner(
            ("Oscar", "cat", [("Play session", "16:00", 15, "medium")]))),
    ]


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

@dataclass
class Row:
    scenario: str
    version: str
    words: int
    passed: int
    failed_checks: list[str]
    text: str


def evaluate(llm) -> list[Row]:
    """Run all three versions on every scenario and score the replies."""
    rows: list[Row] = []
    for name, owner in scenarios():
        scheduler = Scheduler(owner)
        plan = scheduler.build_daily_plan()
        conflicts = todays_conflicts(scheduler.detect_conflicts())
        features = compute_features(plan, conflicts)
        plan_text = plan_to_text(owner, plan, conflicts)
        notes = retrieve(owner, plan)
        sources = [c.source for c in notes]

        calls = {
            # version: (function that gets the reply, sources the checks should expect)
            "baseline": (lambda: baseline_explain(plan_text, llm), []),
            "specialized": (
                lambda: specialized_explain(plan_text, features, llm), []),
            "specialized+RAG": (
                lambda: specialized_explain(
                    plan_text, features, llm, notes_text=format_notes(notes)),
                sources),
        }
        for version in VERSIONS:
            call, expected_sources = calls[version]
            try:
                text = call()
            except CoachError as error:
                text = f"[error: {error}]"
                failed = list(CHECK_NAMES)
            else:
                results = evaluate_checks(text, features, plan_text, expected_sources)
                failed = [check for check, problem in results.items() if problem]
            rows.append(Row(name, version, len(text.split()),
                            len(CHECK_NAMES) - len(failed), failed, text))
    return rows


def render_markdown(rows: list[Row], dry_run: bool) -> str:
    """Build the comparison tables as markdown."""
    out = []
    if dry_run:
        out.append("> DRY RUN: stub model, NOT real Gemini output. Do not use these numbers.\n")
    out.append("## Baseline vs specialized vs specialized+RAG (per scenario)\n")
    out.append("| Scenario | Version | Words | Checks passed | Failed checks |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        failed = ", ".join(r.failed_checks) or "none"
        out.append(f"| {r.scenario} | {r.version} | {r.words} | {r.passed}/{len(CHECK_NAMES)} | {failed} |")

    out.append("\n## Summary\n")
    out.append("| Version | Avg words | Checks passed | Scenarios passing all checks |")
    out.append("|---|---|---|---|")
    for version in VERSIONS:
        mine = [r for r in rows if r.version == version]
        total = len(mine) * len(CHECK_NAMES)
        passed = sum(r.passed for r in mine)
        perfect = sum(1 for r in mine if r.passed == len(CHECK_NAMES))
        avg_words = sum(r.words for r in mine) / len(mine)
        out.append(f"| {version} | {avg_words:.0f} | {passed}/{total} ({passed / total:.0%}) | {perfect}/{len(mine)} |")
    return "\n".join(out) + "\n"


def render_samples(rows: list[Row]) -> str:
    """Show one scenario's three replies together (the before/after example)."""
    lines = [f"\n## Before/after sample: {SAMPLE_SCENARIO}\n"]
    for r in rows:
        if r.scenario == SAMPLE_SCENARIO:
            lines.append(f"**{r.version}**\n\n```text\n{r.text}\n```\n")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Dry-run stub
# ----------------------------------------------------------------------

def stub_llm(system: str, user: str) -> str:
    """Fake model for --dry-run: chatty for the baseline, terse for the coach."""
    if "PawPal Coach" in system:
        tip = " [dog.md]" if "CARE NOTES:\n- [" in user else ""
        return f"Top priority: Daisy, Milo, Oscar, Scout and the rest are covered.\nTip: Start with the first task.{tip}"
    return "# Your schedule\n\nHere is a long, friendly breakdown of everything. " * 12


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="also write the markdown report to this file")
    parser.add_argument("--dry-run", action="store_true", help="use a stub model (offline wiring check)")
    args = parser.parse_args(argv)

    llm = stub_llm if args.dry_run else gemini_llm
    if not args.dry_run:
        try:  # fail early with a clear message instead of 24 error rows
            gemini_llm("You are a test.", "Say OK.")
        except CoachError as error:
            print(f"Cannot reach Gemini: {error}", file=sys.stderr)
            print("Set GEMINI_API_KEY, install google-genai, or use --dry-run.", file=sys.stderr)
            return 1

    rows = evaluate(llm)
    report = render_markdown(rows, args.dry_run) + render_samples(rows)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Saved report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())