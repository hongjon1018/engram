"""Rule-based query-intent classification for typed-memory retrieval filters."""

from __future__ import annotations

import re
from dataclasses import dataclass

from engram.models import MemorySystem


@dataclass(frozen=True)
class QueryMemoryIntent:
    """Typed-memory filters inferred from a retrieval query."""

    memory_systems: tuple[MemorySystem, ...] | None = None
    memory_subtypes: tuple[str, ...] | None = None
    tags: tuple[str, ...] | None = None
    reason: str = "no typed query intent detected"


class RuleBasedQueryIntentClassifier:
    """Conservative deterministic query classifier for typed-memory recall."""

    def classify(self, query: str) -> QueryMemoryIntent:
        text = query.lower()
        normalized = re.sub(r"[^a-z0-9_\s-]", " ", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return QueryMemoryIntent()

        for classifier in (
            _preference_intent,
            _procedural_intent,
            _prospective_intent,
            _episodic_intent,
            _working_intent,
            _semantic_intent,
        ):
            intent = classifier(normalized)
            if intent is not None:
                return intent
        return QueryMemoryIntent()


def _preference_intent(query: str) -> QueryMemoryIntent | None:
    if not _has_any(
        query,
        (
            "prefer",
            "preference",
            "should i use",
            "do i use",
            "package manager",
            "communication style",
            "how detailed",
            "concise or detailed",
        ),
    ):
        return None
    subtype = None
    tags: tuple[str, ...] | None = None
    if _has_any(query, ("package manager", "pnpm", "npm", "yarn", "bun")):
        subtype = "tool_preference"
        tags = tuple(tag for tag in ("pnpm", "npm", "yarn", "bun") if tag in query) or None
    elif _has_any(query, ("communication", "style", "concise", "detailed", "explain")):
        subtype = "communication_style"
    return QueryMemoryIntent(
        memory_systems=(MemorySystem.PREFERENCE,),
        memory_subtypes=(subtype,) if subtype is not None else None,
        tags=tags,
        reason="query asks for a user/project preference",
    )


def _procedural_intent(query: str) -> QueryMemoryIntent | None:
    if not _has_any(
        query,
        (
            "workflow",
            "runbook",
            "procedure",
            "process",
            "checklist",
            "steps",
            "how do i",
            "how should i",
            "release",
            "deploy",
            "validation",
        ),
    ):
        return None
    subtype = None
    if _has_any(query, ("release", "deploy", "changelog", "version", "ship")):
        subtype = "release_process"
    elif _has_any(query, ("test", "validation", "validate", "check")):
        subtype = "validation_recipe"
    return QueryMemoryIntent(
        memory_systems=(MemorySystem.PROCEDURAL,),
        memory_subtypes=(subtype,) if subtype is not None else None,
        reason="query asks for a reusable procedure",
    )


def _prospective_intent(query: str) -> QueryMemoryIntent | None:
    if not _has_any(
        query,
        ("todo", "next", "remind", "reminder", "follow up", "deadline", "planned"),
    ):
        return None
    return QueryMemoryIntent(
        memory_systems=(MemorySystem.PROSPECTIVE,),
        memory_subtypes=("trigger",),
        reason="query asks for future-oriented work",
    )


def _episodic_intent(query: str) -> QueryMemoryIntent | None:
    if not _has_any(
        query,
        (
            "what happened",
            "when did",
            "last time",
            "yesterday",
            "earlier",
            "incident",
            "debugged",
            "audit",
            "history",
        ),
    ):
        return None
    return QueryMemoryIntent(
        memory_systems=(MemorySystem.EPISODIC,),
        reason="query asks for a past event or episode",
    )


def _working_intent(query: str) -> QueryMemoryIntent | None:
    if not _has_any(query, ("current", "right now", "in progress", "scratch", "temporary")):
        return None
    return QueryMemoryIntent(
        memory_systems=(MemorySystem.WORKING,),
        reason="query asks for current working-memory state",
    )


def _semantic_intent(query: str) -> QueryMemoryIntent | None:
    if not _has_any(query, ("what is", "define", "meaning", "architecture", "schema")):
        return None
    return QueryMemoryIntent(
        memory_systems=(MemorySystem.SEMANTIC,),
        reason="query asks for durable factual knowledge",
    )


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
