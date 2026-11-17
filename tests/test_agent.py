"""Tests for the agent workflow. The model is scripted, so no network or key is needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import TRACE_HEADING, run_agent, save_trace  # noqa: E402
from coach import CoachError, fallback_explanation, todays_conflicts  # noqa: E402
from pawpal_system import Owner, Pet, Scheduler, Task  # noqa: E402
from retriever import retrieve  # noqa: E402


def setup(*tasks):
    owner = Owner("Tester")
    pet = Pet("Max", "dog")
    for description, at, duration, priority in tasks:
        pet.add_task(Task(description, at, duration_minutes=duration, priority=priority))
    owner.add_pet(pet)
    scheduler = Scheduler(owner)
    plan = scheduler.build_daily_plan()
    conflicts = todays_conflicts(scheduler.detect_conflicts())
    notes = retrieve(owner, plan)
    good = fallback_explanation(plan, conflicts, cite=notes[0].source if notes else None)
    return owner, plan, conflicts, good


def scripted(replies):
    queue = list(replies)

    def llm(system, user):
        llm.calls.append(user)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    llm.calls = []
    return llm


def step_names(result):
    return [s.name for s in result.trace]


def test_empty_plan_makes_no_model_call():
    llm = scripted([])
    result = run_agent(Owner("Tester"), [], [], llm=llm)
    assert result.source == "guardrail"
    assert result.attempts == 0
    assert llm.calls == []
    assert step_names(result) == ["short_circuit"]


def test_good_reply_is_accepted_first_try():
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    result = run_agent(owner, plan, conflicts, llm=scripted([good]))
    assert result.source == "gemini"
    assert result.attempts == 1
    assert result.problems == []
    assert step_names(result) == [
        "inspect_plan", "make_plan", "retrieve_care_notes", "select_examples", "draft", "self_check",
    ]


def test_retrieved_notes_are_shown_to_the_model():
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    llm = scripted([good])
    result = run_agent(owner, plan, conflicts, llm=llm)
    assert result.notes and result.notes[0].source == "dog.md"
    assert "- [dog.md]" in llm.calls[0]


def test_bad_reply_is_revised_with_feedback_then_accepted():
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    llm = scripted(["no format at all", good])
    result = run_agent(owner, plan, conflicts, llm=llm)
    assert result.source == "gemini"
    assert result.attempts == 2
    assert result.problems
    assert "broke these rules" in llm.calls[1]
    assert "revise" in step_names(result)


def test_persistent_bad_reply_uses_fallback():
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    result = run_agent(owner, plan, conflicts, llm=scripted(["bad", "still bad"]))
    assert result.source == "fallback"
    assert result.text.startswith("Top priority:")
    assert step_names(result)[-1] == "fallback"


def test_model_error_uses_fallback_instead_of_crashing():
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    result = run_agent(owner, plan, conflicts, llm=scripted([CoachError("no network")]))
    assert result.source == "fallback"
    assert result.attempts == 1


def test_invented_citation_is_caught():
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    result = run_agent(owner, plan, conflicts, llm=scripted([good + " [cats.md]", good]))
    assert any("not provided" in p for p in result.problems)
    assert result.source == "gemini"


def test_rag_off_skips_retrieval_and_forbids_citations():
    owner, plan, conflicts, _ = setup(("Morning walk", "08:00", 30, "high"))
    plain = fallback_explanation(plan, conflicts)  # no citation
    result = run_agent(owner, plan, conflicts, llm=scripted([plain]), use_rag=False)
    assert result.source == "gemini"
    assert result.notes == []
    retrieval = next(s for s in result.trace if s.name == "retrieve_care_notes")
    assert "skipped" in retrieval.detail


def test_retrieval_failure_does_not_break_the_agent():
    owner, plan, conflicts, _ = setup(("Morning walk", "08:00", 30, "high"))

    def broken(owner, plan):
        raise RuntimeError("disk error")

    plain = fallback_explanation(plan, conflicts)
    result = run_agent(owner, plan, conflicts, llm=scripted([plain]), retriever=broken)
    assert result.source == "gemini"
    assert any("FAILED" in s.detail for s in result.trace)


def test_medication_day_adds_a_vet_requirement_to_the_plan_step():
    owner, plan, conflicts, good = setup(("Give medication", "08:00", 10, "high"))
    result = run_agent(owner, plan, conflicts, llm=scripted([good]))
    make_plan = next(s for s in result.trace if s.name == "make_plan")
    assert "vet mention" in make_plan.detail


# --- saved traces ----------------------------------------------------------------

def test_trace_is_appended_to_markdown_without_touching_existing_text(tmp_path):
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    path = tmp_path / "ai_interactions.md"
    path.write_text("# My earlier notes\n\nSome existing content.\n", encoding="utf-8")

    run_agent(owner, plan, conflicts, llm=scripted([good]), trace_path=path)
    run_agent(owner, plan, conflicts, llm=scripted(["bad", good]), trace_path=path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("# My earlier notes\n\nSome existing content.\n")
    assert text.count(TRACE_HEADING) == 1  # heading added once
    assert text.count("### Run ") == 2
    assert "Step 1: inspect_plan" in text
    assert "FAIL:" in text  # the failed self-check of the second run is recorded


def test_unwritable_trace_path_never_breaks_the_agent(tmp_path):
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    bad_path = tmp_path / "no_such_folder" / "ai_interactions.md"
    result = run_agent(owner, plan, conflicts, llm=scripted([good]), trace_path=bad_path)
    assert result.source == "gemini"


def test_save_trace_adds_heading_to_a_new_file(tmp_path):
    owner, plan, conflicts, good = setup(("Morning walk", "08:00", 30, "high"))
    result = run_agent(owner, plan, conflicts, llm=scripted([good]))
    path = tmp_path / "new.md"
    save_trace(result, path)
    assert TRACE_HEADING in path.read_text(encoding="utf-8")
