import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach import (  # noqa: E402
    CHECK_NAMES,
    EXAMPLES_FILE,
    PlanFeatures,
    build_user_message,
    check_output,
    compute_features,
    evaluate_checks,
    fallback_explanation,
    plan_to_text,
    select_examples,
    times_in,
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


PLAN_TEXT = "- 08:00-08:30 | Max (dog) | Morning walk | high | on time"
FEATURES = PlanFeatures(n_tasks=1, n_pets=1, top_pet="Max")
GOOD = "Top priority: Max's morning walk at 08:00 comes first.\nTip: Keep a water bowl by the door."


# --- the few-shot dataset ------------------------------------------------

def load_examples_raw():
    return json.loads(Path(EXAMPLES_FILE).read_text(encoding="utf-8"))


def test_every_example_passes_every_guardrail_check():
    examples = load_examples_raw()
    assert len(examples) >= 10
    for e in examples:
        tags = set(e["tags"])
        features = PlanFeatures(
            top_pet=e["top_pet"],
            has_conflicts="conflict" in tags,
            has_moved="moved" in tags,
            has_unplaced="unplaced" in tags,
            has_medication="medication" in tags,
        )
        sources = re.findall(r"- \[([\w.\-]+)\]", e["care_notes"])
        assert check_output(e["output"], features, e["plan_text"], sources) == [], e["tags"]


def test_examples_cover_both_cited_and_uncited_tips():
    examples = load_examples_raw()
    assert any(e["care_notes"] == "none" for e in examples)
    assert any(e["care_notes"] != "none" for e in examples)


def test_select_examples_prefers_matching_tags():
    owner = make_owner(("Give medication", "08:00", 10, "high"))
    plan, conflicts = plan_for(owner)
    features = compute_features(plan, conflicts)
    chosen = select_examples(features, load_examples_raw(), k=3)
    assert len(chosen) == 3
    assert "medication" in chosen[0]["tags"]


# --- the ten checks, one by one ---------------------------------------------

def test_check_names_are_the_ten_documented_checks():
    assert len(CHECK_NAMES) == 10
    assert set(evaluate_checks(GOOD, FEATURES)) == set(CHECK_NAMES)


def test_good_reply_passes_all_checks():
    assert check_output(GOOD, FEATURES, PLAN_TEXT, sources=[]) == []


@pytest.mark.parametrize(
    "text, check",
    [
        ("Top priority: Max walks. " + "word " * 130 + "\nTip: x", "length"),
        ("Tip: only a tip about Max", "opener"),
        ("Top priority: Max's walk.", "tip"),
        (GOOD + " Give 5 mg twice daily.", "no medical advice"),
        ("Top priority: The walk is first.\nTip: Hydrate.", "names top pet"),
        (GOOD + " Do it again at 23:59.", "grounded times"),
        (GOOD.replace("Tip:", "**Tip:**"), "plain text"),
        (GOOD + "\n- a bullet point", "plain text"),
    ],
)
def test_each_check_catches_its_problem(text, check):
    results = evaluate_checks(text, FEATURES, PLAN_TEXT, sources=[])
    assert results[check] is not None, results


def test_heads_up_is_required_only_when_something_needs_flagging():
    needs = PlanFeatures(top_pet="Max", has_moved=True)
    assert evaluate_checks(GOOD, needs)["heads-up"] is not None
    assert evaluate_checks(GOOD.replace("Tip:", "Heads-up: x\nTip:"), FEATURES)["heads-up"] is not None


def test_vet_mention_is_required_for_medication_plans():
    med = PlanFeatures(top_pet="Max", has_medication=True)
    assert evaluate_checks(GOOD, med)["vet mention"] is not None
    assert evaluate_checks(GOOD + " Ask your vet.", med)["vet mention"] is None
    assert evaluate_checks(GOOD + " Ask the vet.", FEATURES)["vet mention"] is None


def test_citations_must_be_provided_and_present():
    cited = GOOD + " [dog.md]"
    assert evaluate_checks(cited, FEATURES, sources=["dog.md"])["citation"] is None
    assert evaluate_checks(GOOD, FEATURES, sources=["dog.md"])["citation"] is not None  # not grounded
    assert evaluate_checks(cited, FEATURES, sources=["cat.md"])["citation"] is not None  # invented
    assert evaluate_checks(cited, FEATURES, sources=[])["citation"] is not None  # none were provided
    assert evaluate_checks(cited, FEATURES, sources=None)["citation"] is None  # check switched off


def test_times_in_finds_hh_mm():
    assert times_in("Walk 08:00-08:30, then 9:15") == {"08:00", "08:30", "9:15"}


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


def test_build_user_message_includes_notes_and_feedback():
    plain = build_user_message("PLAN TEXT", "- [dog.md] x")
    assert "CARE NOTES:\n- [dog.md] x" in plain
    retry = build_user_message("PLAN TEXT", "none", ["too long (130 words, limit 120)"])
    assert "broke these rules: too long" in retry
    assert retry.rstrip().endswith("EXPLANATION:")


# --- fallback -------------------------------------------------------------------

def test_fallback_passes_its_own_checks():
    owner = make_owner(
        ("Park trip", "17:00", 150, "high"),
        ("Training", "19:30", 60, "high"),
        ("Grooming", "18:30", 60, "low"),
    )
    plan, conflicts = plan_for(owner)
    features = compute_features(plan, conflicts)
    text = fallback_explanation(plan, conflicts)
    assert check_output(text, features, plan_to_text(owner, plan, conflicts)) == []


def test_fallback_with_a_medication_task_mentions_the_vet():
    owner = make_owner(("Give medication", "08:00", 10, "high"))
    plan, conflicts = plan_for(owner)
    assert "vet" in fallback_explanation(plan, conflicts).lower()
