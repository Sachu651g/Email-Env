"""Tests for MetricsTracker — unit tests and property-based tests."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openenv_email_ops.metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_initial_metrics_all_zero():
    tracker = MetricsTracker()
    metrics = tracker.get_metrics()
    assert metrics["total_reward"] == 0.0
    assert metrics["classification_accuracy"] == 0.0
    assert metrics["prioritization_accuracy"] == 0.0
    assert metrics["routing_accuracy"] == 0.0
    assert metrics["vip_handling_rate"] == 0.0
    assert metrics["deferral_count"] == 0


@pytest.mark.unit
def test_classification_accuracy_all_correct():
    tracker = MetricsTracker()
    for _ in range(5):
        tracker.record_classification(True)
    assert tracker.get_metrics()["classification_accuracy"] == 1.0


@pytest.mark.unit
def test_classification_accuracy_none_correct():
    tracker = MetricsTracker()
    for _ in range(3):
        tracker.record_classification(False)
    assert tracker.get_metrics()["classification_accuracy"] == 0.0


@pytest.mark.unit
def test_classification_accuracy_partial():
    tracker = MetricsTracker()
    tracker.record_classification(True)
    tracker.record_classification(False)
    assert tracker.get_metrics()["classification_accuracy"] == 0.5


@pytest.mark.unit
def test_prioritization_accuracy():
    tracker = MetricsTracker()
    tracker.record_prioritization(True)
    tracker.record_prioritization(True)
    tracker.record_prioritization(False)
    assert pytest.approx(tracker.get_metrics()["prioritization_accuracy"]) == 2 / 3


@pytest.mark.unit
def test_routing_accuracy():
    tracker = MetricsTracker()
    tracker.record_routing(False)
    assert tracker.get_metrics()["routing_accuracy"] == 0.0


@pytest.mark.unit
def test_vip_handling_rate():
    tracker = MetricsTracker()
    tracker.record_vip_handled(True)
    tracker.record_vip_handled(False)
    assert tracker.get_metrics()["vip_handling_rate"] == 0.5


@pytest.mark.unit
def test_vip_handling_rate_no_vips():
    tracker = MetricsTracker()
    assert tracker.get_metrics()["vip_handling_rate"] == 0.0


@pytest.mark.unit
def test_deferral_count():
    tracker = MetricsTracker()
    tracker.record_deferral()
    tracker.record_deferral()
    tracker.record_deferral()
    assert tracker.get_metrics()["deferral_count"] == 3


@pytest.mark.unit
def test_total_reward_accumulates():
    tracker = MetricsTracker()
    tracker.record_reward(0.2)
    tracker.record_reward(-0.05)
    tracker.record_reward(0.3)
    assert pytest.approx(tracker.get_metrics()["total_reward"]) == 0.45


@pytest.mark.unit
def test_reset_clears_all_state():
    tracker = MetricsTracker()
    tracker.record_classification(True)
    tracker.record_prioritization(False)
    tracker.record_routing(True)
    tracker.record_reward(1.5)
    tracker.record_deferral()
    tracker.record_vip_handled(True)

    tracker.reset()
    metrics = tracker.get_metrics()
    assert metrics["total_reward"] == 0.0
    assert metrics["classification_accuracy"] == 0.0
    assert metrics["prioritization_accuracy"] == 0.0
    assert metrics["routing_accuracy"] == 0.0
    assert metrics["vip_handling_rate"] == 0.0
    assert metrics["deferral_count"] == 0


@pytest.mark.unit
def test_get_metrics_returns_all_six_fields():
    tracker = MetricsTracker()
    metrics = tracker.get_metrics()
    required_keys = {
        "total_reward",
        "classification_accuracy",
        "prioritization_accuracy",
        "routing_accuracy",
        "vip_handling_rate",
        "deferral_count",
    }
    assert required_keys == set(metrics.keys())


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


# Feature: openenv-email-ops, Property 19: Episode metrics completeness
@pytest.mark.property
@settings(max_examples=100)
@given(
    rewards=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False), max_size=20),
    classifications=st.lists(st.booleans(), max_size=20),
    prioritizations=st.lists(st.booleans(), max_size=20),
    routings=st.lists(st.booleans(), max_size=20),
    vip_handlings=st.lists(st.booleans(), max_size=10),
    deferrals=st.integers(min_value=0, max_value=20),
)
def test_p19_episode_metrics_completeness(
    rewards, classifications, prioritizations, routings, vip_handlings, deferrals
):
    """
    **Validates: Requirements 14.2, 14.3**

    For any completed episode, the metrics dict SHALL contain all six required fields:
    total_reward, classification_accuracy, prioritization_accuracy, routing_accuracy,
    vip_handling_rate, and deferral_count.
    """
    tracker = MetricsTracker()

    for r in rewards:
        tracker.record_reward(r)
    for c in classifications:
        tracker.record_classification(c)
    for p in prioritizations:
        tracker.record_prioritization(p)
    for ro in routings:
        tracker.record_routing(ro)
    for v in vip_handlings:
        tracker.record_vip_handled(v)
    for _ in range(deferrals):
        tracker.record_deferral()

    metrics = tracker.get_metrics()

    # All six required fields must be present
    assert "total_reward" in metrics
    assert "classification_accuracy" in metrics
    assert "prioritization_accuracy" in metrics
    assert "routing_accuracy" in metrics
    assert "vip_handling_rate" in metrics
    assert "deferral_count" in metrics

    # Accuracy fields must be in [0.0, 1.0]
    assert 0.0 <= metrics["classification_accuracy"] <= 1.0
    assert 0.0 <= metrics["prioritization_accuracy"] <= 1.0
    assert 0.0 <= metrics["routing_accuracy"] <= 1.0
    assert 0.0 <= metrics["vip_handling_rate"] <= 1.0

    # deferral_count must match what was recorded
    assert metrics["deferral_count"] == deferrals

    # total_reward must match sum of recorded rewards
    assert pytest.approx(metrics["total_reward"], abs=1e-6) == sum(rewards)


@pytest.mark.property
@settings(max_examples=100)
@given(
    n_correct=st.integers(min_value=0, max_value=50),
    n_incorrect=st.integers(min_value=0, max_value=50),
)
def test_accuracy_bounds_classification(n_correct, n_incorrect):
    """Classification accuracy is always in [0.0, 1.0] and matches ratio."""
    tracker = MetricsTracker()
    for _ in range(n_correct):
        tracker.record_classification(True)
    for _ in range(n_incorrect):
        tracker.record_classification(False)

    acc = tracker.get_metrics()["classification_accuracy"]
    assert 0.0 <= acc <= 1.0

    total = n_correct + n_incorrect
    if total == 0:
        assert acc == 0.0
    else:
        assert pytest.approx(acc) == n_correct / total


@pytest.mark.property
@settings(max_examples=100)
@given(
    n_correct=st.integers(min_value=0, max_value=50),
    n_incorrect=st.integers(min_value=0, max_value=50),
)
def test_vip_handling_rate_bounds(n_correct, n_incorrect):
    """VIP handling rate is always in [0.0, 1.0] and matches ratio."""
    tracker = MetricsTracker()
    for _ in range(n_correct):
        tracker.record_vip_handled(True)
    for _ in range(n_incorrect):
        tracker.record_vip_handled(False)

    rate = tracker.get_metrics()["vip_handling_rate"]
    assert 0.0 <= rate <= 1.0

    total = n_correct + n_incorrect
    if total == 0:
        assert rate == 0.0
    else:
        assert pytest.approx(rate) == n_correct / total


@pytest.mark.property
@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=100))
def test_deferral_count_matches_records(n):
    """Deferral count always equals the number of record_deferral() calls."""
    tracker = MetricsTracker()
    for _ in range(n):
        tracker.record_deferral()
    assert tracker.get_metrics()["deferral_count"] == n


@pytest.mark.property
@settings(max_examples=100)
@given(
    rewards=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False), min_size=1, max_size=30),
)
def test_total_reward_is_sum(rewards):
    """total_reward equals the sum of all recorded rewards."""
    tracker = MetricsTracker()
    for r in rewards:
        tracker.record_reward(r)
    assert pytest.approx(tracker.get_metrics()["total_reward"], abs=1e-6) == sum(rewards)
