"""Tests for the care-note retriever (RAG)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pawpal_system import Owner, Pet, Scheduler, Task  # noqa: E402
from retriever import format_notes, load_chunks, retrieve, species_key, tokenize  # noqa: E402


def plan_for(*pets):
    """pets: (name, species, [(description, time, priority), ...])"""
    owner = Owner("Tester")
    for name, species, tasks in pets:
        pet = Pet(name, species)
        for description, at, priority in tasks:
            pet.add_task(Task(description, at, priority=priority))
        owner.add_pet(pet)
    return owner, Scheduler(owner).build_daily_plan()


def test_documents_load_into_sections():
    chunks = load_chunks()
    sources = {c.source for c in chunks}
    assert {"dog.md", "cat.md", "rabbit.md", "fish.md", "turtle.md", "general.md", "medication.md"} <= sources
    assert all(c.heading and c.text for c in chunks)
    assert len(chunks) >= 15


def test_species_aliases_map_free_text():
    assert species_key("Golden Retriever") == "dog"
    assert species_key("kitten") == "cat"
    assert species_key("Bunny") == "rabbit"
    assert species_key("axolotl") == ""


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("Morning walks") == {"walk"}
    assert "walk" in tokenize("walking the dog")


def test_dog_walk_retrieves_dog_notes():
    owner, plan = plan_for(("Max", "dog", [("Morning walk", "08:00", "high")]))
    notes = retrieve(owner, plan)
    assert notes and notes[0].source == "dog.md"


def test_species_documents_are_never_used_for_other_species():
    owner, plan = plan_for(("Luna", "cat", [("Play session", "08:00", "medium")]))
    notes = retrieve(owner, plan)
    assert notes
    assert all(n.source != "dog.md" for n in notes)
    assert notes[0].source == "cat.md"


def test_medication_task_gets_medication_notes_even_on_a_busy_day():
    owner, plan = plan_for(
        ("Max", "dog", [("Morning walk", "07:00", "high"), ("Training", "17:00", "medium")]),
        ("Luna", "cat", [("Give medication", "07:15", "high")]),
    )
    sources = [n.source for n in retrieve(owner, plan)]
    assert "medication.md" in sources


def test_no_match_returns_nothing():
    owner, plan = plan_for(("Zed", "axolotl", [("Zzz", "08:00", "low")]))
    assert retrieve(owner, plan) == []


def test_results_are_capped_and_unique():
    owner, plan = plan_for(
        ("Max", "dog", [("Morning walk", "07:00", "high"), ("Evening walk", "18:00", "high"),
                        ("Training", "12:00", "medium"), ("Lunch feeding", "13:00", "medium")]),
    )
    notes = retrieve(owner, plan, k=2)
    assert len(notes) == 2
    assert len({(n.source, n.heading) for n in notes}) == 2


def test_format_notes():
    assert format_notes([]) == "none"
    owner, plan = plan_for(("Max", "dog", [("Morning walk", "08:00", "high")]))
    text = format_notes(retrieve(owner, plan))
    assert text.startswith("- [dog.md]")
