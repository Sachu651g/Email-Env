"""Property-based tests for InboxGenerator.

Properties covered:
  - Property 3: Seeded episode determinism
  - Property 4: Inbox sender_type coverage invariant
  - Property 5: Ground truth completeness invariant
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openenv_email_ops.inbox_generator import InboxGenerator

_VALID_CLASSIFICATIONS = {"spam", "important", "promotion"}
_VALID_PRIORITIES = {"low", "medium", "high"}
_VALID_ROUTES = {"support", "sales", "escalation"}
_ALL_SENDER_TYPES = {"customer", "spammer", "VIP", "internal"}


# ---------------------------------------------------------------------------
# Property 3: Seeded episode determinism
# ---------------------------------------------------------------------------

# Feature: openenv-email-ops, Property 3: Seeded episode determinism
@pytest.mark.property
@settings(max_examples=100)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    size=st.integers(min_value=1, max_value=20),
)
def test_seeded_determinism(seed: int, size: int) -> None:
    """Same seed produces identical email sequences.

    Validates: Requirements 2.2
    """
    gen = InboxGenerator()
    emails1 = gen.generate(size=size, seed=seed)
    emails2 = gen.generate(size=size, seed=seed)

    assert len(emails1) == len(emails2)
    for e1, e2 in zip(emails1, emails2):
        assert e1.id == e2.id
        assert e1.subject == e2.subject
        assert e1.body == e2.body
        assert e1.sender_type == e2.sender_type
        assert e1.urgency_score == e2.urgency_score
        assert e1.ground_truth == e2.ground_truth


# ---------------------------------------------------------------------------
# Property 4: Inbox sender_type coverage invariant
# ---------------------------------------------------------------------------

# Feature: openenv-email-ops, Property 4: Inbox sender_type coverage invariant
@pytest.mark.property
@settings(max_examples=100)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    size=st.integers(min_value=4, max_value=20),
)
def test_sender_type_coverage(seed: int, size: int) -> None:
    """Inbox with size >= 4 always contains all 4 sender types.

    Validates: Requirements 2.4
    """
    gen = InboxGenerator()
    emails = gen.generate(size=size, seed=seed)

    sender_types = {e.sender_type for e in emails}
    assert sender_types == _ALL_SENDER_TYPES, (
        f"Missing sender types: {_ALL_SENDER_TYPES - sender_types}"
    )


# ---------------------------------------------------------------------------
# Property 5: Ground truth completeness invariant
# ---------------------------------------------------------------------------

# Feature: openenv-email-ops, Property 5: Ground truth completeness invariant
@pytest.mark.property
@settings(max_examples=100)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    size=st.integers(min_value=1, max_value=20),
)
def test_ground_truth_completeness(seed: int, size: int) -> None:
    """Every generated email has all 3 ground truth fields populated with valid values.

    Validates: Requirements 2.5
    """
    gen = InboxGenerator()
    emails = gen.generate(size=size, seed=seed)

    for email in emails:
        gt = email.ground_truth
        assert gt is not None, f"Email {email.id} has no ground_truth"
        assert gt.correct_classification in _VALID_CLASSIFICATIONS, (
            f"Invalid classification: {gt.correct_classification}"
        )
        assert gt.correct_priority in _VALID_PRIORITIES, (
            f"Invalid priority: {gt.correct_priority}"
        )
        assert gt.correct_route in _VALID_ROUTES, (
            f"Invalid route: {gt.correct_route}"
        )
