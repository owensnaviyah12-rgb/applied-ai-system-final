from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from pawpal_system import PRIORITY_ORDER, Owner, PlanItem

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

EXAMPLES_FILE = Path(__file__).parent / "data" / "examples.json"
LOG_FILE = Path(__file__).parent / "coach.log"
MAX_WORDS = 120
DEFAULT_MODEL = "gemini-2.5-flash"  # override with the GEMINI_MODEL env var
MEDICATION_HINTS = ("medic", "pill", "insulin", "antibiotic", "flea", "tick")

# Phrases the coach must never produce (it explains plans, it is not a vet).
BANNED_PATTERNS = (
    r"\b\d+(\.\d+)?\s?(mg|mcg|ml|cc|iu|units?)\b",  # dose amounts
    r"\bdiagnos",
    r"\bdosage\b",
    r"\bdose\b",
    r"\bprescri",
)

LLM = Callable[[str, str], str]  # (system_prompt, user_text) -> reply text

LOG = logging.getLogger("pawpal.coach")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    try:
        _handler: logging.Handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    except OSError:  # read-only folder: log to the console instead
        _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(_handler)


class CoachError(Exception):
    """Raised when the language model cannot produce a usable reply."""


# ----------------------------------------------------------------------
# Gemini call
# ----------------------------------------------------------------------

def gemini_llm(system: str, user: str) -> str:
    """Call Gemini and return the reply text. Raises CoachError on any failure.

    Needs the `google-genai` package and a GEMINI_API_KEY environment variable.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise CoachError("GEMINI_API_KEY is not set.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise CoachError("The google-genai package is not installed.") from error

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        text = (response.text or "").strip()
    except Exception as error:  # network, quota, auth, safety blocks...
        raise CoachError(f"Gemini request failed: {error}") from error

    if not text:
        raise CoachError("Gemini returned an empty reply.")
    return text


# ----------------------------------------------------------------------
# Turning the scheduler's output into text the model can read
# ----------------------------------------------------------------------

@dataclass
class PlanFeatures:
    """Facts about a plan, used to pick examples and to validate replies."""

    n_tasks: int = 0
    n_pets: int = 0
    has_conflicts: bool = False
    has_moved: bool = False
    has_unplaced: bool = False
    has_medication: bool = False
    top_pet: str = ""
    top_desc: str = ""

    @property
    def needs_heads_up(self) -> bool:
        """True when the reply must include a Heads-up line."""
        return self.has_conflicts or self.has_moved or self.has_unplaced

    @property
    def tags(self) -> set[str]:
        """Tags used to match few-shot examples (same names as examples.json)."""
        tags = {"multi_pet" if self.n_pets > 1 else "single_pet"}
        if self.has_conflicts:
            tags.add("conflict")
        if self.has_moved:
            tags.add("moved")
        if self.has_unplaced:
            tags.add("unplaced")
        if self.has_medication:
            tags.add("medication")
        return tags


def todays_conflicts(conflicts: list[str], day: Optional[date] = None) -> list[str]:
    """Keep only the scheduler's conflict warnings for one day (default today)."""
    marker = f"Conflict on {day or date.today()}:"
    return [c for c in conflicts if c.startswith(marker)]


def compute_features(plan: list[PlanItem], conflicts: list[str]) -> PlanFeatures:
    """Summarize a plan so we can match examples and check the model's reply."""
    features = PlanFeatures(
        n_tasks=len(plan),
        n_pets=len({item.task.pet_name for item in plan}),
        has_conflicts=bool(conflicts),
        has_moved=any(item.moved for item in plan),
        has_unplaced=any(item.start is None for item in plan),
        has_medication=any(
            hint in item.task.description.lower()
            for item in plan
            for hint in MEDICATION_HINTS
        ),
    )
    placed = [item for item in plan if item.start is not None]
    candidates = placed or plan
    if candidates:
        top = min(
            candidates,
            key=lambda i: (PRIORITY_ORDER[i.task.priority], i.start or "99:99"),
        )
        features.top_pet = top.task.pet_name
        features.top_desc = top.task.description
    return features


def plan_to_text(owner: Owner, plan: list[PlanItem], conflicts: list[str]) -> str:
    """Render the plan in the fixed text format used by the prompts and examples."""
    lines = [f"Today's plan for {date.today()} (owner: {owner.name})", "Tasks:"]
    for item in plan:
        pet = owner.get_pet(item.task.pet_name)
        species = pet.species if pet else "pet"
        who = f"{item.task.pet_name} ({species})"
        priority = item.task.priority
        if item.start is None:
            lines.append(
                f"- no slot | {who} | {item.task.description} | {priority} | "
                f"NO ROOM TODAY (wanted {item.task.time})"
            )
        else:
            note = f"moved from {item.task.time}" if item.moved else "on time"
            lines.append(
                f"- {item.start}-{item.end} | {who} | {item.task.description} | "
                f"{priority} | {note}"
            )
    lines.append("Overlapping requests:")
    if conflicts:
        lines.extend(f"- {c}" for c in conflicts)
    else:
        lines.append("none")
    return "\n".join(lines)


def times_in(text: str) -> set[str]:
    """Every HH:MM time that appears in a piece of text."""
    return set(re.findall(r"\b\d{1,2}:\d{2}\b", text))


# ----------------------------------------------------------------------
# Few-shot examples
# ----------------------------------------------------------------------

def load_examples(path: Path = EXAMPLES_FILE) -> list[dict]:
    """Load the synthetic example dataset (empty list if unreadable)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOG.warning("Could not load examples from %s: %s", path, error)
        return []


def select_examples(features: PlanFeatures, examples: list[dict], k: int = 3) -> list[dict]:
    """Pick the k examples whose tags overlap most with today's plan.

    Ties keep the file order, so the choice is deterministic.
    """
    wanted = features.tags
    scored = sorted(
        enumerate(examples),
        key=lambda pair: (-len(wanted & set(pair[1].get("tags", []))), pair[0]),
    )
    return [example for _, example in scored[:k]]


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

BASELINE_SYSTEM = "You are a helpful assistant."

SPECIALIZED_SYSTEM = """You are PawPal Coach, a calm, practical assistant that explains a pet owner's daily care plan.
The plan was already built by a scheduling engine. Never change times and never invent tasks; only explain it.

Reply in exactly this format:
Top priority: <one sentence naming the pet and the most important task and when it happens>
Heads-up: <one or two sentences, ONLY if a task was moved, has no room today, or requests overlapped>
Tip: <one practical, low-risk tip for today>

Rules:
- Under 120 words in total. Plain text only: no markdown, no emojis, no bullet points.
- Warm but brief. Speak to the owner as "you".
- Use only the pets, tasks and times that appear in the plan. Never mention a time that is not in the plan.
- Never diagnose, never give medication doses or amounts, never suggest changing a prescribed routine.
- If the plan includes any medication or treatment task, mention the vet in the Heads-up or Tip.
- If nothing needs flagging, leave out the Heads-up line completely.
- CARE NOTES are trusted reference text. If notes are provided, base the Tip on one of them and end the Tip
  with that note's source in square brackets exactly as shown, for example [dog.md]. Do not use facts that
  are not in the notes and never invent a source. If the notes are "none", write a general low-risk Tip and
  do not cite anything.
"""


def build_specialized_system(examples: list[dict]) -> str:
    """Combine the rules with the chosen few-shot examples."""
    shots = "\n\n".join(
        f"PLAN:\n{e['plan_text']}\n\nCARE NOTES:\n{e.get('care_notes', 'none')}"
        f"\n\nEXPLANATION:\n{e['output']}"
        for e in examples
    )
    if not shots:
        return SPECIALIZED_SYSTEM
    return f"{SPECIALIZED_SYSTEM}\nExamples of ideal replies:\n\n{shots}\n"


def build_user_message(
    plan_text: str, notes_text: str = "none", feedback: Optional[list[str]] = None
) -> str:
    """The per-request message. `feedback` is the list of rules a previous reply broke."""
    note = ""
    if feedback:
        note = (
            f"Your previous reply broke these rules: {'; '.join(feedback)}.\n"
            "Rewrite it so it follows every rule.\n\n"
        )
    return f"PLAN:\n{plan_text}\n\nCARE NOTES:\n{notes_text}\n\n{note}EXPLANATION:"


def baseline_explain(plan_text: str, llm: LLM = gemini_llm) -> str:
    """Generic prompt, no rules, no examples. Used only as the comparison baseline."""
    return llm(BASELINE_SYSTEM, f"Explain this pet care schedule.\n\n{plan_text}")


def specialized_explain(
    plan_text: str,
    features: PlanFeatures,
    llm: LLM = gemini_llm,
    examples: Optional[list[dict]] = None,
    notes_text: str = "none",
) -> str:
    """Rules + few-shot examples (+ care notes). One raw model call, no guardrails."""
    chosen = select_examples(
        features, load_examples() if examples is None else examples
    )
    return llm(
        build_specialized_system(chosen), build_user_message(plan_text, notes_text)
    )


# ----------------------------------------------------------------------
# OUTPUT GUARDRAIL: ten named checks
# ----------------------------------------------------------------------

CHECK_NAMES = [
    "length",
    "opener",
    "tip",
    "heads-up",
    "no medical advice",
    "names top pet",
    "grounded times",
    "plain text",
    "vet mention",
    "citation",
]

_CITATION = re.compile(r"\[([\w.\-]+\.md)\]")
_VET = re.compile(r"\bvet(erinarian)?s?\b", re.IGNORECASE)
_MARKDOWN_OR_EMOJI = re.compile(
    r"[*#`]|^\s*[-•]\s|[\U0001F300-\U0001FAFF☀-➿]", re.MULTILINE
)


def evaluate_checks(
    text: str,
    features: PlanFeatures,
    plan_text: Optional[str] = None,
    sources: Optional[list[str]] = None,
) -> dict[str, Optional[str]]:
    """Run every guardrail check. Maps check name -> problem message (None = passed).

    plan_text: when given, every time in the reply must appear in the plan.
    sources:   file names of the care notes the model was shown. When given
               (even as an empty list) citations are checked against it.
    """
    results: dict[str, Optional[str]] = {name: None for name in CHECK_NAMES}

    words = len(text.split())
    if words > MAX_WORDS:
        results["length"] = f"too long ({words} words, limit {MAX_WORDS})"
    if not text.lstrip().startswith("Top priority:"):
        results["opener"] = "does not start with 'Top priority:'"
    if "Tip:" not in text:
        results["tip"] = "missing 'Tip:' line"
    if features.needs_heads_up and "Heads-up:" not in text:
        results["heads-up"] = "missing 'Heads-up:' line for a moved/overlapping/unplaced task"
    elif not features.needs_heads_up and "Heads-up:" in text:
        results["heads-up"] = "has an unneeded 'Heads-up:' line"
    if any(re.search(p, text, re.IGNORECASE) for p in BANNED_PATTERNS):
        results["no medical advice"] = "contains diagnosis or dosage language"
    if features.top_pet and features.top_pet.lower() not in text.lower():
        results["names top pet"] = "does not name the top-priority pet"
    if plan_text is not None:
        invented = sorted(times_in(text) - times_in(plan_text))
        if invented:
            results["grounded times"] = f"mentions times that are not in the plan: {', '.join(invented)}"
    if _MARKDOWN_OR_EMOJI.search(text):
        results["plain text"] = "uses markdown, bullets or emoji (plain text only)"
    if features.has_medication and not _VET.search(text):
        results["vet mention"] = "medication plan does not mention the vet"
    if sources is not None:
        cited = _CITATION.findall(text)
        invalid = [c for c in cited if c not in sources]
        if invalid:
            results["citation"] = f"cites sources that were not provided: {', '.join(invalid)}"
        elif sources and not cited:
            results["citation"] = "Tip is not grounded in the care notes (no [source] citation)"
    return results


def check_output(
    text: str,
    features: PlanFeatures,
    plan_text: Optional[str] = None,
    sources: Optional[list[str]] = None,
) -> list[str]:
    """Return the list of rule violations (an empty list means the reply is good)."""
    return [m for m in evaluate_checks(text, features, plan_text, sources).values() if m]


# ----------------------------------------------------------------------
# Rule-based fallback
# ----------------------------------------------------------------------

def fallback_explanation(
    plan: list[PlanItem], conflicts: list[str], cite: Optional[str] = None
) -> str:
    """A deterministic explanation in the same format, used when Gemini fails.

    `cite` (a care-note file name) is only used by the tests and the evaluation
    harness, to script a compliant "model" reply. The app never passes it.
    """
    features = compute_features(plan, conflicts)
    top = next(
        (i for i in plan
         if i.task.pet_name == features.top_pet
         and i.task.description == features.top_desc),
        plan[0],
    )
    when = f" at {top.start}" if top.start else ""
    lines = [
        f"Top priority: {features.top_pet}'s {features.top_desc.lower()}{when} "
        "is the most important task today."
    ]

    flags = []
    for item in (i for i in plan if i.moved):
        flags.append(
            f"{item.task.pet_name}'s {item.task.description.lower()} moved from "
            f"{item.task.time} to {item.start} to avoid an overlap."
        )
    for item in (i for i in plan if i.start is None):
        flags.append(
            f"{item.task.pet_name}'s {item.task.description.lower()} has no room today."
        )
    if features.has_conflicts and not flags:
        flags.append("Some of your requested times overlapped.")
    if flags:
        lines.append("Heads-up: " + " ".join(flags))

    if features.has_medication:
        tip = "Tip: Keep medication on schedule and ask your vet if anything about it is unclear."
    else:
        tip = "Tip: Do the highest-priority task first so the rest of the day stays flexible."
    if cite:
        tip += f" [{cite}]"
    lines.append(tip)
    return "\n".join(lines)
