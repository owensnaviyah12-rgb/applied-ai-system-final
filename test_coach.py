import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach import (  # noqa: E402
    EXAMPLES_FILE,
    CoachError,
    check_output,
    compute_features,
    explain_plan,
    fallback_explanation,
    plan_to_text,
    select_examples,
    todays_conflicts,
)
from pawpal_system import Owner, Pet, Scheduler, Task  # noqa: E402


def make_owner(*tasks):
    """One dog named Max with the given (description, time, duration, priority) tasks."""
    owner = Owner("Tester")
    pet = Pet("Max", "dog")
    for description, at, duration, priority in tasks:
        pet.add_task(Task(description, at, duration_minutes=duration, priority=priority))
    owner.add_pet(pet)
    return owner


def plan_for(owner):
    scheduler = Scheduler(owner)
    return scheduler.build_daily_plan(), todays_conflicts(scheduler.detect_conflicts())


GOOD = "Top priority: Max's morning walk at 08:00 comes first.\nTip: Keep a water bowl by the door."


# --- the few-shot dataset ------------------------------------------------

def test_every_example_follows_its_own_rules():
    examples = json.loads(Path(EXAMPLES_FILE).read_text(encoding="utf-8"))
    assert len(examples) >= 10
    for example in examples:
        needs = bool({"conflict", "moved", "unplaced"} & set(example["tags"]))
        top_pet = example["output"].split("Top priority: ")[1].split("'")[0].split(" ")[0]
        assert check_output(example["output"], top_pet, needs) == [], example["tags"]


def test_select_examples_prefers_matching_tags():
    owner = make_owner(("Give medication", "08:00", 10, "high"))
    plan, conflicts = plan_for(owner)
    features = compute_features(plan, conflicts)
    examples = json.loads(Path(EXAMPLES_FILE).read_text(encoding="utf-8"))
    chosen = select_examples(features, examples, k=3)
    assert len(chosen) == 3
    assert "medication" in chosen[0]["tags"]


# --- checks ---------------------------------------------------------------

def test_check_output_accepts_a_good_reply():
    assert check_output(GOOD, "Max", needs_heads_up=False) == []


@pytest.mark.parametrize(
    "text, needs, fragment",
    [
        ("Hello! " + "word " * 130, False, "too long"),
        ("Tip: only a tip about Max", False, "Top priority"),
        ("Top priority: Max's walk.", False, "Tip"),
        (GOOD, True, "Heads-up"),
        (GOOD.replace("Tip:", "Heads-up: x\nTip:"), False, "unneeded"),
        (GOOD + " Give 5 mg twice daily.", False, "diagnosis or dosage"),
        ("Top priority: The walk is first.\nTip: Hydrate.", False, "top-priority pet"),
    ],
)
def test_check_output_flags_each_rule(text, needs, fragment):
    problems = check_output(text, "Max", needs)
    assert any(fragment in p for p in problems), problems


# --- text rendering ----------------------------------------------------------

def test_plan_to_text_marks_moved_and_unplaced_tasks():
    owner = make_owner(
        ("Park trip", "17:00", 150, "high"),
        ("Training", "19:30", 60, "high"),
        ("Grooming", "18:30", 60, "low"),
    )
    plan, conflicts = plan_for(owner)
    text = plan_to_text(owner, plan, conflicts)
    assert "NO ROOM TODAY" in text
    assert "Max (dog)" in text
    assert "Conflict on" in text


# --- explain_plan guardrails -----------------------------------------------------

def test_empty_plan_makes_no_model_call():
    calls = []
    result = explain_plan(Owner("Tester"), [], [], llm=lambda s, u: calls.append(1) or GOOD)
    assert result.source == "guardrail"
    assert calls == []


def test_good_reply_is_accepted_first_try():
    owner = make_owner(("Morning walk", "08:00", 30, "high"))
    plan, conflicts = plan_for(owner)
    result = explain_plan(owner, plan, conflicts, llm=lambda s, u: GOOD)
    assert result.source == "gemini"
    assert result.problems == []


def test_bad_reply_is_retried_then_accepted():
    owner = make_owner(("Morning walk", "08:00", 30, "high"))
    plan, conflicts = plan_for(owner)
    replies = iter(["Here is a long chatty answer with no format.", GOOD])
    result = explain_plan(owner, plan, conflicts, llm=lambda s, u: next(replies))
    assert result.source == "gemini"
    assert result.problems  # the first reply's violations were recorded


def test_retry_prompt_names_the_broken_rules():
    owner = make_owner(("Morning walk", "08:00", 30, "high"))
    plan, conflicts = plan_for(owner)
    seen = []

    def llm(system, user):
        seen.append(user)
        return "no format" if len(seen) == 1 else GOOD

    explain_plan(owner, plan, conflicts, llm=llm)
    assert "broke these rules" in seen[1]


def test_persistent_bad_reply_uses_fallback():
    owner = make_owner(("Morning walk", "08:00", 30, "high"))
    plan, conflicts = plan_for(owner)
    result = explain_plan(owner, plan, conflicts, llm=lambda s, u: "no format at all")
    assert result.source == "fallback"
    assert result.text.startswith("Top priority:")


def test_model_error_uses_fallback_instead_of_crashing():
    owner = make_owner(("Morning walk", "08:00", 30, "high"))
    plan, conflicts = plan_for(owner)

    def broken(system, user):
        raise CoachError("no network")

    result = explain_plan(owner, plan, conflicts, llm=broken)
    assert result.source == "fallback"


def test_fallback_passes_its_own_checks():
    owner = make_owner(
        ("Park trip", "17:00", 150, "high"),
        ("Training", "19:30", 60, "high"),
        ("Grooming", "18:30", 60, "low"),
    )
    plan, conflicts = plan_for(owner)
    features = compute_features(plan, conflicts)
    text = fallback_explanation(plan, conflicts)
    assert check_output(text, features.top_pet, features.needs_heads_up) == []
