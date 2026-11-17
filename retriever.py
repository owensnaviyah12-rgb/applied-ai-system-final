from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path

from pawpal_system import PRIORITY_ORDER, Owner, PlanItem

LOG = logging.getLogger("pawpal.retriever")

GUIDELINES_DIR = Path(__file__).parent / "data" / "care_guidelines"

# Which document key a free-text species belongs to ("Golden retriever" -> dog).
SPECIES_ALIASES = {
    "dog": ("dog", "puppy", "canine", "retriever", "labrador", "poodle", "terrier", "beagle"),
    "cat": ("cat", "kitten", "feline"),
    "rabbit": ("rabbit", "bunny"),
    "fish": ("fish", "goldfish", "betta"),
    "turtle": ("turtle", "tortoise"),
}

STOPWORDS = {
    "the", "and", "for", "with", "give", "your", "his", "her", "its", "our",
    "morning", "evening", "midday", "night", "time", "session", "today", "daily",
    "from", "into", "that", "this", "some", "new",
}


@dataclass
class Chunk:
    """One section of a care document."""

    source: str  # file name, for example "dog.md"
    heading: str
    text: str
    score: int = 0

    def as_note(self) -> str:
        """The one-line form that goes into the prompt."""
        return f"[{self.source}] {self.heading}: {self.text}"


def _stem(word: str) -> str:
    """A very small stemmer so 'walks', 'walking' and 'walk' all match."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> set[str]:
    """Lowercase, drop stopwords and short words, and stem."""
    words = re.findall(r"[a-z]+", text.lower())
    return {_stem(w) for w in words if len(w) >= 3 and w not in STOPWORDS}


def species_key(species: str) -> str:
    """Map a pet's free-text species to a document key, or '' if unknown."""
    lowered = species.lower()
    for key, aliases in SPECIES_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return key
    return ""


def load_chunks(docs_dir: Path = GUIDELINES_DIR) -> list[Chunk]:
    """Read every .md file and split it into one chunk per '## ' section."""
    chunks: list[Chunk] = []
    try:
        files = sorted(Path(docs_dir).glob("*.md"))
    except OSError as error:
        LOG.warning("Cannot read %s: %s", docs_dir, error)
        return chunks
    for path in files:
        heading, lines = None, []
        for raw in path.read_text(encoding="utf-8").splitlines() + ["## "]:
            line = raw.strip()
            if line.startswith("## "):
                if heading and lines:
                    chunks.append(Chunk(path.name, heading, " ".join(lines)))
                heading, lines = line[3:].strip(), []
            elif line and not line.startswith("#") and heading:
                lines.append(line)
    if not chunks:
        LOG.warning("No care guideline chunks found in %s", docs_dir)
    return chunks


def retrieve(
    owner: Owner,
    plan: list[PlanItem],
    k: int = 3,
    chunks: list[Chunk] | None = None,
) -> list[Chunk]:
    """Return up to k care notes: the best chunk for each task, high priority first."""
    chunks = load_chunks() if chunks is None else chunks
    picked: dict[tuple[str, str], Chunk] = {}

    ordered = sorted(
        plan, key=lambda i: (PRIORITY_ORDER[i.task.priority], i.start or "99:99")
    )
    for item in ordered:
        pet = owner.get_pet(item.task.pet_name)
        species = species_key(pet.species) if pet else ""
        query = tokenize(item.task.description)

        best: Chunk | None = None
        for chunk in chunks:
            doc = Path(chunk.source).stem
            if doc in SPECIES_ALIASES and doc != species:
                continue  # never show cat notes for a dog
            overlap = len(query & tokenize(f"{chunk.heading} {chunk.text}"))
            if overlap == 0:
                continue
            score = overlap + (3 if doc == species else 0)
            if best is None or score > best.score:
                best = replace(chunk, score=score)

        if best is not None:
            picked.setdefault((best.source, best.heading), best)
        if len(picked) >= k:
            break
    return list(picked.values())


def format_notes(chunks: list[Chunk]) -> str:
    """Render notes for the prompt ('none' when nothing relevant was found)."""
    if not chunks:
        return "none"
    return "\n".join(f"- {c.as_note()}" for c in chunks)
