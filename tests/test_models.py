"""Property-based and unit tests for openenv_email_ops/models.py."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from openenv_email_ops.models import (
    Action,
    Email,
    GroundTruth,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = [
    "classify_email",
    "prioritize_email",
    "route_email",
    "generate_reply",
    "defer_email",
]

invalid_action_type_strategy = st.text().filter(
    lambda s: s not in VALID_ACTION_TYPES and len(s) > 0
)

valid_ground_truth_strategy = st.builds(
    GroundTruth,
    correct_classification=st.sampled_from(["spam", "important", "promotion"]),
    correct_priority=st.sampled_from(["low", "medium", "high"]),
    correct_route=st.sampled_from(["support", "sales", "escalation"]),
)

valid_email_strategy = st.builds(
    Email,
    id=st.uuids().map(str),
    subject=st.text(min_size=1, max_size=100),
    body=st.text(min_size=1, max_size=500),
    sender_type=st.sampled_from(["customer", "spammer", "VIP", "internal"]),
    urgency_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ground_truth=valid_ground_truth_strategy,
)

# ---------------------------------------------------------------------------
# Property 8: Invalid action type raises ValidationError
# Feature: openenv-email-ops, Property 8: Invalid action type raises ValidationError
# Validates: Requirements 1.6, 4.5, 10.5
# ---------------------------------------------------------------------------

@pytest.mark.property
@settings(max_examples=100)
@given(invalid_action_type_strategy)
def test_invalid_action_type_raises_validation_error(invalid_type: str):
    """Property 8: Any string not in the five valid action types raises ValidationError."""
    with pytest.raises(ValidationError):
        Action(action_type=invalid_type)


# ---------------------------------------------------------------------------
# Property 16: Out-of-range field values raise ValidationError
# Feature: openenv-email-ops, Property 16: Out-of-range field values raise ValidationError
# Validates: Requirements 10.4
# ---------------------------------------------------------------------------

out_of_range_urgency_strategy = st.one_of(
    st.floats(max_value=-0.0001, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.0001, allow_nan=False, allow_infinity=False),
)


@pytest.mark.property
@settings(max_examples=100)
@given(out_of_range_urgency_strategy)
def test_out_of_range_urgency_score_raises_validation_error(bad_score: float):
    """Property 16: urgency_score outside [0.0, 1.0] raises ValidationError."""
    gt = GroundTruth(
        correct_classification="spam",
        correct_priority="low",
        correct_route="support",
    )
    with pytest.raises(ValidationError) as exc_info:
        Email(
            id="test-id",
            subject="Test",
            body="Test body",
            sender_type="customer",
            urgency_score=bad_score,
            ground_truth=gt,
        )
    # Verify the error message references the field
    errors = exc_info.value.errors()
    assert any("urgency_score" in str(e) for e in errors)
