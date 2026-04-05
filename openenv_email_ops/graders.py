"""Deterministic graders for scoring agent actions in openenv-email-ops."""

from __future__ import annotations

import re

from openenv_email_ops.models import Email

# Stop words to filter out when extracting subject keywords
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "from", "and", "or", "but",
    "not", "no", "it", "its", "this", "that", "i", "you", "we", "they",
    "he", "she", "my", "your", "our", "their", "your",
})

# Greeting patterns (case-insensitive)
_GREETINGS = [
    "hi", "hello", "dear", "good morning", "good afternoon",
    "good evening", "greetings", "hey",
]

# Placeholder patterns that indicate an incomplete reply
_PLACEHOLDER_PATTERNS = [
    "[",
    "TODO",
    "PLACEHOLDER",
    "INSERT",
    "YOUR NAME",
    "FILL IN",
]


# Semantic adjacency for classification labels.
# Adjacent labels receive partial credit (0.5) instead of 0.0.
_CLASSIFICATION_ADJACENCY: dict[str, set[str]] = {
    "important": {"promotion"},
    "promotion": {"important"},
    "spam": set(),
}

# Priority adjacency: one level apart = adjacent (0.5); two levels = 0.0
_PRIORITY_ADJACENCY: dict[str, set[str]] = {
    "high": {"medium"},
    "medium": {"high", "low"},
    "low": {"medium"},
}


class ClassificationGrader:
    """Scores classify_email actions against ground truth labels.

    Returns:
        1.0 — exact match
        0.5 — semantically adjacent label (e.g. "promotion" when correct is "important")
        0.0 — completely wrong label
    """

    def score(self, predicted: str, ground_truth: str) -> float:
        p, g = predicted.strip().lower(), ground_truth.strip().lower()
        if p == g:
            return 1.0
        if p in _CLASSIFICATION_ADJACENCY.get(g, set()):
            return 0.5
        return 0.0


class PrioritizationGrader:
    """Scores prioritize_email actions against ground truth labels.

    Returns:
        1.0 — exact match
        0.5 — adjacent priority level (high↔medium or medium↔low)
        0.0 — two-level mismatch (high↔low)
    """

    def score(self, predicted: str, ground_truth: str) -> float:
        p, g = predicted.strip().lower(), ground_truth.strip().lower()
        if p == g:
            return 1.0
        if p in _PRIORITY_ADJACENCY.get(g, set()):
            return 0.5
        return 0.0


class RoutingGrader:
    """Scores route_email actions against ground truth labels."""

    def score(self, predicted: str, ground_truth: str) -> float:
        """Return 1.0 if predicted matches ground_truth (case-insensitive), else 0.0."""
        return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


class ReplyGrader:
    """Heuristic grader for generate_reply actions. Returns a float in [0.0, 1.0]."""

    def score(self, reply: str, email: Email) -> float:
        """Score a reply against 5 criteria, each worth 0.2.

        Criteria:
          1. Reply length >= 30 characters
          2. Reply contains a greeting
          3. Reply contains at least one keyword from the email subject
          4. Reply does NOT contain placeholder text
          5. Reply length >= 80 characters (substantive content)
        """
        criteria_met = 0

        # Criterion 1: minimum length
        if len(reply) >= 30:
            criteria_met += 1

        # Criterion 2: greeting present
        reply_lower = reply.lower()
        if any(greeting in reply_lower for greeting in _GREETINGS):
            criteria_met += 1

        # Criterion 3: subject keyword overlap
        subject_words = re.findall(r"\w+", email.subject.lower())
        keywords = [w for w in subject_words if w not in _STOP_WORDS and len(w) > 1]
        if keywords and any(kw in reply_lower for kw in keywords):
            criteria_met += 1

        # Criterion 4: no placeholder text
        reply_upper = reply.upper()
        has_placeholder = any(p.upper() in reply_upper for p in _PLACEHOLDER_PATTERNS)
        if not has_placeholder:
            criteria_met += 1

        # Criterion 5: substantive length
        if len(reply) >= 80:
            criteria_met += 1

        return criteria_met * 0.2

    def score_hard(self, reply: str, email: Email) -> float:
        """Score a reply for hard-task emails, adding dominant-intent check.

        Calls score() then adds +0.25 if email.dominant_intent keyword appears
        in the reply. Result is capped at 1.0. Falls back to score() if
        dominant_intent is None.
        """
        base = self.score(reply, email)
        if email.dominant_intent and email.dominant_intent.lower() in reply.lower():
            return min(1.0, base + 0.25)
        return base
