"""Tests for graders.py — unit tests and property-based tests."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openenv_email_ops.graders import (
    ClassificationGrader,
    PrioritizationGrader,
    ReplyGrader,
    RoutingGrader,
)
from openenv_email_ops.models import Email, GroundTruth

# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

_CLASSIFICATION_VALUES = ["spam", "important", "promotion"]
_PRIORITY_VALUES = ["low", "medium", "high"]
_ROUTE_VALUES = ["support", "sales", "escalation"]
_SENDER_TYPES = ["customer", "spammer", "VIP", "internal"]


def _make_email(subject: str = "Test subject") -> Email:
    return Email(
        id="test-id",
        subject=subject,
        body="Test body content.",
        sender_type="customer",
        urgency_score=0.5,
        ground_truth=GroundTruth(
            correct_classification="important",
            correct_priority="medium",
            correct_route="support",
        ),
    )


@st.composite
def email_strategy(draw):
    subject = draw(st.text(min_size=1, max_size=80))
    body = draw(st.text(min_size=0, max_size=200))
    sender_type = draw(st.sampled_from(_SENDER_TYPES))
    urgency_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    classification = draw(st.sampled_from(_CLASSIFICATION_VALUES))
    priority = draw(st.sampled_from(_PRIORITY_VALUES))
    route = draw(st.sampled_from(_ROUTE_VALUES))
    return Email(
        id="gen-id",
        subject=subject,
        body=body,
        sender_type=sender_type,
        urgency_score=urgency_score,
        ground_truth=GroundTruth(
            correct_classification=classification,
            correct_priority=priority,
            correct_route=route,
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classification_correct_reward():
    """Correct prediction returns 1.0 (maps to +0.2 reward)."""
    grader = ClassificationGrader()
    assert grader.score("spam", "spam") == 1.0
    assert grader.score("important", "important") == 1.0
    assert grader.score("promotion", "promotion") == 1.0


@pytest.mark.unit
def test_classification_incorrect_penalty():
    """Wrong prediction returns 0.0 (maps to -0.2 reward); adjacent returns 0.5."""
    grader = ClassificationGrader()
    assert grader.score("spam", "important") == 0.0
    assert grader.score("promotion", "spam") == 0.0
    # "important" is adjacent to "promotion" → partial credit 0.5
    assert grader.score("important", "promotion") == 0.5


@pytest.mark.unit
def test_classification_case_insensitive():
    grader = ClassificationGrader()
    assert grader.score("SPAM", "spam") == 1.0
    assert grader.score("Important", "important") == 1.0


@pytest.mark.unit
def test_reply_grader_criteria():
    """Test each of the 4 ReplyGrader criteria independently."""
    grader = ReplyGrader()
    email = _make_email(subject="Invoice payment overdue")

    # All 5 criteria met → 1.0
    # Need: >= 30 chars, greeting, keyword, no placeholder, >= 80 chars
    full_reply = "Hello, I wanted to follow up on the invoice payment that is overdue. We will process it right away and keep you updated on the status."
    assert grader.score(full_reply, email) == 1.0

    # Criterion 1 fails: reply too short (< 30 chars), also fails criterion 5 (< 80 chars)
    short_reply = "Hi, invoice."  # has greeting + keyword + no placeholder, but < 30 chars
    score_short = grader.score(short_reply, email)
    assert score_short == pytest.approx(0.6)  # criteria 2,3,4 met (3 of 5)

    # Criterion 2 fails: no greeting
    no_greeting = "I wanted to follow up on the invoice payment that is overdue today and will ensure it is resolved promptly for you."
    score_no_greeting = grader.score(no_greeting, email)
    assert score_no_greeting == 0.8  # length>=30, keyword, no placeholder, length>=80

    # Criterion 3 fails: no subject keyword in reply
    no_keyword = "Hello, please let me know if you need any assistance with your request. We are happy to help you resolve this matter quickly and efficiently."
    score_no_kw = grader.score(no_keyword, email)
    assert score_no_kw == 0.8  # length>=30, greeting, no placeholder, length>=80

    # Criterion 4 fails: contains placeholder
    placeholder_reply = "Hello, please pay the invoice by [DATE] or contact [NAME] for help with the overdue invoice payment issue that you raised."
    score_placeholder = grader.score(placeholder_reply, email)
    assert score_placeholder == 0.8  # length>=30, greeting, keyword, length>=80

    # All 5 criteria fail → 0.0
    bad_reply = "TODO [NAME]"
    assert grader.score(bad_reply, email) == 0.0


@pytest.mark.unit
def test_reply_grader_empty_reply():
    grader = ReplyGrader()
    email = _make_email()
    score = grader.score("", email)
    # Empty: no length, no greeting, no keyword, but also no placeholder → 0.2 (criterion 4 only)
    assert score == 0.2


@pytest.mark.unit
def test_prioritization_grader():
    grader = PrioritizationGrader()
    assert grader.score("high", "high") == 1.0
    assert grader.score("low", "high") == 0.0
    # Adjacent levels return 0.5
    assert grader.score("medium", "high") == 0.5
    assert grader.score("medium", "low") == 0.5


@pytest.mark.unit
def test_routing_grader():
    grader = RoutingGrader()
    assert grader.score("support", "support") == 1.0
    assert grader.score("sales", "escalation") == 0.0


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


# Property 11: All graders return scores in [0.0, 1.0]
# Validates: Requirements 9.1, 9.2, 9.3
@pytest.mark.property
@settings(max_examples=100)
@given(predicted=st.text(), ground_truth=st.text())
def test_classification_grader_range(predicted: str, ground_truth: str):
    """**Validates: Requirements 9.1**"""
    grader = ClassificationGrader()
    score = grader.score(predicted, ground_truth)
    assert 0.0 <= score <= 1.0


@pytest.mark.property
@settings(max_examples=100)
@given(predicted=st.text(), ground_truth=st.text())
def test_prioritization_grader_range(predicted: str, ground_truth: str):
    """**Validates: Requirements 9.2**"""
    grader = PrioritizationGrader()
    score = grader.score(predicted, ground_truth)
    assert 0.0 <= score <= 1.0


@pytest.mark.property
@settings(max_examples=100)
@given(predicted=st.text(), ground_truth=st.text())
def test_routing_grader_range(predicted: str, ground_truth: str):
    """**Validates: Requirements 9.3**"""
    grader = RoutingGrader()
    score = grader.score(predicted, ground_truth)
    assert 0.0 <= score <= 1.0


# Property 10: Reply grader score is in valid range
# Validates: Requirements 5.4, 9.4
@pytest.mark.property
@settings(max_examples=100)
@given(reply=st.text(), email=email_strategy())
def test_reply_grader_range(reply: str, email: Email):
    """**Validates: Requirements 5.4, 9.4**"""
    grader = ReplyGrader()
    score = grader.score(reply, email)
    assert 0.0 <= score <= 1.0


# Property 12: Grader determinism
# Validates: Requirements 9.5
@pytest.mark.property
@settings(max_examples=100)
@given(predicted=st.text(), ground_truth=st.text())
def test_classification_grader_determinism(predicted: str, ground_truth: str):
    """**Validates: Requirements 9.5**"""
    grader = ClassificationGrader()
    assert grader.score(predicted, ground_truth) == grader.score(predicted, ground_truth)


@pytest.mark.property
@settings(max_examples=100)
@given(predicted=st.text(), ground_truth=st.text())
def test_prioritization_grader_determinism(predicted: str, ground_truth: str):
    """**Validates: Requirements 9.5**"""
    grader = PrioritizationGrader()
    assert grader.score(predicted, ground_truth) == grader.score(predicted, ground_truth)


@pytest.mark.property
@settings(max_examples=100)
@given(predicted=st.text(), ground_truth=st.text())
def test_routing_grader_determinism(predicted: str, ground_truth: str):
    """**Validates: Requirements 9.5**"""
    grader = RoutingGrader()
    assert grader.score(predicted, ground_truth) == grader.score(predicted, ground_truth)


@pytest.mark.property
@settings(max_examples=100)
@given(reply=st.text(), email=email_strategy())
def test_reply_grader_determinism(reply: str, email: Email):
    """**Validates: Requirements 9.5**"""
    grader = ReplyGrader()
    assert grader.score(reply, email) == grader.score(reply, email)
