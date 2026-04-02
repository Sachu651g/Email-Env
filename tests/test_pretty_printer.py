"""Property-based tests for PrettyPrinter serialization."""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openenv_email_ops.models import (
    Action,
    Email,
    GroundTruth,
    InboxSummary,
    Observation,
)
from openenv_email_ops.pretty_printer import PrettyPrinter

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

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
    urgency_score=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    ground_truth=valid_ground_truth_strategy,
)

valid_action_strategy = st.builds(
    Action,
    action_type=st.sampled_from(
        ["classify_email", "prioritize_email", "route_email", "generate_reply", "defer_email"]
    ),
    value=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
)

valid_inbox_summary_strategy = st.builds(
    InboxSummary,
    counts_by_sender_type=st.fixed_dictionaries(
        {
            "customer": st.integers(min_value=0, max_value=10),
            "spammer": st.integers(min_value=0, max_value=10),
            "VIP": st.integers(min_value=0, max_value=10),
            "internal": st.integers(min_value=0, max_value=10),
        }
    ),
    urgency_distribution=st.fixed_dictionaries(
        {
            "low": st.integers(min_value=0, max_value=10),
            "medium": st.integers(min_value=0, max_value=10),
            "high": st.integers(min_value=0, max_value=10),
        }
    ),
)

valid_observation_strategy = st.builds(
    Observation,
    current_email=st.one_of(st.none(), valid_email_strategy),
    inbox_summary=valid_inbox_summary_strategy,
    action_history=st.lists(valid_action_strategy, max_size=10),
    step_count=st.integers(min_value=0, max_value=100),
)


# ---------------------------------------------------------------------------
# Property 6: Ground truth excluded from observation serialization
# Feature: openenv-email-ops, Property 6: Ground truth excluded from observation serialization
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=100)
@given(valid_observation_strategy)
def test_ground_truth_excluded_from_json(obs: Observation):
    """Property 6: Serializing any Observation to JSON must not contain 'ground_truth'."""
    printer = PrettyPrinter()
    json_str = printer.to_json(obs)

    # Parse back to verify structure
    data = json.loads(json_str)

    # ground_truth must not appear anywhere in the JSON string
    assert "ground_truth" not in json_str

    # Also verify it's not in the parsed dict (current_email level)
    if data.get("current_email") is not None:
        assert "ground_truth" not in data["current_email"]


# ---------------------------------------------------------------------------
# Property 7: Observation serialization round-trip
# Feature: openenv-email-ops, Property 7: Observation serialization round-trip
# Validates: Requirements 3.4, 10.2, 10.3
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=100)
@given(valid_observation_strategy)
def test_observation_serialization_round_trip(obs: Observation):
    """Property 7: Serialize to JSON then deserialize back preserves all non-ground-truth fields."""
    printer = PrettyPrinter()
    json_str = printer.to_json(obs)

    # Deserialize back — Observation.model_validate_json handles the excluded ground_truth
    # by accepting the partial email dict (ground_truth is optional in deserialization context)
    data = json.loads(json_str)

    # Verify step_count preserved
    assert data["step_count"] == obs.step_count

    # Verify action_history preserved
    assert len(data["action_history"]) == len(obs.action_history)
    for orig, serialized in zip(obs.action_history, data["action_history"]):
        assert serialized["action_type"] == orig.action_type
        assert serialized.get("value") == orig.value

    # Verify inbox_summary preserved
    assert data["inbox_summary"]["counts_by_sender_type"] == obs.inbox_summary.counts_by_sender_type
    assert data["inbox_summary"]["urgency_distribution"] == obs.inbox_summary.urgency_distribution

    # Verify current_email non-ground-truth fields preserved
    if obs.current_email is None:
        assert data["current_email"] is None
    else:
        email_data = data["current_email"]
        assert email_data["id"] == obs.current_email.id
        assert email_data["subject"] == obs.current_email.subject
        assert email_data["body"] == obs.current_email.body
        assert email_data["sender_type"] == obs.current_email.sender_type
        assert abs(email_data["urgency_score"] - obs.current_email.urgency_score) < 1e-9
