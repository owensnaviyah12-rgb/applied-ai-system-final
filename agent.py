from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from coach import (
    LLM,
    LOG,
    CoachError,
    build_specialized_system,
    build_user_message,
    check_output,
    compute_features,
    fallback_explanation,
    gemini_llm,
    load_examples,
    plan_to_text,
    select_examples,
)
from pawpal_system import Owner, PlanItem
from retriever import Chunk, format_notes, retrieve

TRACE_FILE = Path(__file__).parent / "ai_interactions.md"
TRACE_HEADING = "## PawPal Coach agent reasoning traces"
EMPTY_PLAN_MESSAGE = (
    "You have nothing planned for today. Add a task above and I'll walk you through the plan."
)


@dataclass
class TraceStep:
    """One step of the agent's reasoning."""

    n: int
    name: str
    tool: str
    detail: str


@dataclass
class AgentResult:
    """The agent's answer plus everything needed to audit how it got there."""

    text: str
    source: str  # "gemini", "fallback", or "guardrail"
    attempts: int  # model calls made
    problems: list[str] = field(default_factory=list)  # rule violations caught along the way
    trace: list[TraceStep] = field(default_factory=list)
    notes: list[Chunk] = field(default_factory=list)  # care notes shown to the model
    plan_text: str = ""


def run_agent(
    owner: Owner,
    plan: list[PlanItem],
    conflicts: list[str],
    llm: LLM = gemini_llm,
    use_rag: bool = True,
    max_revisions: int = 1,
    trace_path: Optional[Path] = None,
    retriever: Callable[[Owner, list[PlanItem]], list[Chunk]] = retrieve,
) -> AgentResult:
    """Explain today's plan. Never raises: worst case it returns the fallback text.

    trace_path: when given, the run is appended to that markdown file.
    """
    trace: list[TraceStep] = []

    def step(name: str, tool: str, detail: str) -> None:
        trace.append(TraceStep(len(trace) + 1, name, tool, detail))

    # --- Step 1: short-circuit on an empty plan -------------------------------
    if not plan:
        step("short_circuit", "guardrail", "The plan is empty, so no model call is made.")
        LOG.info("Empty plan: skipped model call")
        result = AgentResult(EMPTY_PLAN_MESSAGE, "guardrail", 0, trace=trace)
        _save(result, trace_path)
        return result

    # --- Steps 2-3: look at the day and decide what to do ----------------------
    features = compute_features(plan, conflicts)
    plan_text = plan_to_text(owner, plan, conflicts)
    step(
        "inspect_plan",
        "compute_features",
        f"{features.n_tasks} tasks, {features.n_pets} pet(s); conflicts={features.has_conflicts}, "
        f"moved={features.has_moved}, no_room={features.has_unplaced}, "
        f"medication={features.has_medication}; top priority: "
        f"{features.top_pet} - {features.top_desc}",
    )

    chain = ["retrieve care notes" if use_rag else "skip retrieval", "draft", "self-check",
             f"revise up to {max_revisions}x if a rule is broken", "fall back if still failing"]
    if features.has_medication:
        chain.insert(3, "require a vet mention (medication day)")
    step("make_plan", "planner", " -> ".join(chain))

    # --- Step 4: retrieval ------------------------------------------------------
    notes: list[Chunk] = []
    if use_rag:
        try:
            notes = retriever(owner, plan)
        except Exception as error:  # a broken doc folder must not break the coach
            LOG.error("Retrieval failed: %s", error)
            step("retrieve_care_notes", "retriever", f"FAILED ({error}); continuing without notes")
        else:
            if notes:
                step("retrieve_care_notes", "retriever",
                     "; ".join(f"{c.source} / {c.heading} (score {c.score})" for c in notes))
            else:
                step("retrieve_care_notes", "retriever", "no relevant notes found; Tip will be general")
    else:
        step("retrieve_care_notes", "retriever", "skipped (RAG turned off)")
    sources = [c.source for c in notes]

    # --- Step 5: few-shot examples ------------------------------------------------
    examples = select_examples(features, load_examples())
    step("select_examples", "select_examples",
         f"{len(examples)} examples; tags: " + " | ".join(",".join(e["tags"]) for e in examples))
    system = build_specialized_system(examples)

    # --- Steps 6-8: draft, check, revise ----------------------------------------------
    notes_text = format_notes(notes)
    feedback: Optional[list[str]] = None
    caught: list[str] = []
    attempts = 0

    for attempt in range(max_revisions + 1):
        label = "draft" if attempt == 0 else "revise"
        attempts += 1
        try:
            reply = llm(system, build_user_message(plan_text, notes_text, feedback))
        except CoachError as error:
            LOG.error("Model call failed (attempt %d): %s", attempts, error)
            step(label, "gemini", f"FAILED: {error}")
            break

        step(label, "gemini", f"{len(reply.split())} words\n{reply}")
        problems = check_output(reply, features, plan_text, sources)
        if not problems:
            step("self_check", "check_output", "PASS (all 10 checks)")
            LOG.info("Accepted reply on attempt %d (%d words)", attempts, len(reply.split()))
            result = AgentResult(reply, "gemini", attempts, caught, trace, notes, plan_text)
            _save(result, trace_path)
            return result

        step("self_check", "check_output", "FAIL: " + "; ".join(problems))
        LOG.warning("Reply rejected on attempt %d: %s", attempts, "; ".join(problems))
        caught.extend(problems)
        feedback = problems

    # --- Last resort ----------------------------------------------------------------------
    step("fallback", "fallback_explanation",
         "Model unavailable or kept breaking the rules; using the built-in explanation.")
    LOG.warning("Using rule-based fallback")
    result = AgentResult(
        fallback_explanation(plan, conflicts), "fallback", attempts, caught, trace, notes, plan_text
    )
    _save(result, trace_path)
    return result


# ----------------------------------------------------------------------
# Saving the reasoning trace
# ----------------------------------------------------------------------

def format_trace(result: AgentResult) -> str:
    """Render one agent run as markdown."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"### Run {stamp} (answer source: {result.source}, model calls: {result.attempts})",
        "",
    ]
    if result.plan_text:
        lines += ["**Input plan**", "", "```text", result.plan_text, "```", ""]
    lines += ["**Reasoning trace**", "", "```text"]
    for s in result.trace:
        lines.append(f"Step {s.n}: {s.name}  [{s.tool}]")
        lines.extend(f"    {part}" for part in s.detail.splitlines())
    lines += ["```", "", "**Final answer**", "", "```text", result.text, "```", ""]
    return "\n".join(lines)


def save_trace(result: AgentResult, path: Path = TRACE_FILE) -> None:
    """Append a run to the markdown file (adds the section heading the first time)."""
    path = Path(path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    chunk = ""
    if TRACE_HEADING not in existing:
        chunk += (
            f"\n\n{TRACE_HEADING}\n\n"
            "Each run below is one pass of the PawPal Coach agent: what it inspected, "
            "which care notes it retrieved, what Gemini drafted, and which guardrail "
            "checks passed or failed.\n\n"
        )
    chunk += format_trace(result) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(chunk)


def _save(result: AgentResult, path: Optional[Path]) -> None:
    """Save a trace without ever letting a file problem break the app."""
    if path is None:
        return
    try:
        save_trace(result, path)
    except OSError as error:
        LOG.error("Could not save trace to %s: %s", path, error)
