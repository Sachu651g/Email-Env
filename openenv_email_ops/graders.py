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


class ClassificationGrader:
    """Scores classify_email actions against ground truth labels."""

    def score(self, predicted: str, ground_truth: str) -> float:
        """Return 1.0 if predicted matches ground_truth (case-insensitive), else 0.0."""
        return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


class PrioritizationGrader:
    """Scores prioritize_email actions against ground truth labels."""

    def score(self, predicted: str, ground_truth: str) -> float:
        """Return 1.0 if predicted matches ground_truth (case-insensitive), else 0.0."""
        return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


class RoutingGrader:
    """Scores route_email actions against ground truth labels."""

    def score(self, predicted: str, ground_truth: str) -> float:
        """Return 1.0 if predicted matches ground_truth (case-insensitive), else 0.0."""
        return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


class ReplyGrader:
    """Heuristic grader for generate_reply actions. Returns a float in [0.0, 1.0]."""

    def score(self, reply: str, email: Email) -> float:
        """Score a reply against 4 criteria, each worth 0.25.

        Criteria:
          1. Reply length >= 20 characters
          2. Reply contains a greeting
          3. Reply contains at least one keyword from the email subject
          4. Reply does NOT contain placeholder text
        """
        criteria_met = 0

        # Criterion 1: minimum length
        if len(reply) >= 20:
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

        return criteria_met * 0.25
