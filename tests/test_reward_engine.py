"""Unit and property tests for RewardEngine."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openenv_email_ops.memory_tracker import MemoryTracker
from openenv_email_ops.models import Action, Email, GroundTruth, TaskConfig
from openenv_email_ops.reward_engine import RewardEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_email(
    email_id: str = "email-1",
    sender_type: str = "customer",
    classification: str = "important",
    priority: str = "high",
    route: str = "support",
    urgency: float = 0.5,
) -> Email:
    return Email(
        id=email_id,
        subject="Test subject meeting",
        body="Hello, please help with this issue.",
        sender_type=sender_type,
        urgency_score=urgency,
        ground_truth=GroundTruth(
            correct_classification=classification,
            correct_priority=priority,
            correct_route=route,
        ),
    )


def make_task(difficulty: str, components: list[str]) -> TaskConfig:
    return TaskConfig(
        task_id=f"task-{difficulty}",
        description=f"{difficulty} task",
        difficulty=difficulty,
        max_steps=20,
        inbox_size=5,
        reward_components=components,
    )


EASY_TASK = make_task("easy", ["classification"])
MEDIUM_TASK = make_task("medium", ["classification", "prioritization", "routing"])
HARD_TASK = make_task("hard", ["classification", "prioritization", "routing", "reply"])


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vip_ignore_penalty():
    """VIP email not classified as important within 3 steps → -0.3 penalty."""
    engine = RewardEngine()
    tracker = MemoryTracker()
    vip_email = make_email("vip-1", sender_type="VIP")

    # Record email received at step 0, but never classify it as important
    tracker.record_email_received("vip-1", step=0)
    # Record a non-classification action
    tracker.record_action("vip-1", Action(action_type="defer_email"), step=1)

    adjustment, breakdown = engine.finalize_episode(tracker, [vip_email], 0.0)

    assert "vip_ignore_penalty_vip-1" in breakdown
    assert breakdown["vip_ignore_penalty_vip-1"] == pytest.approx(-0.3)
    assert adjustment <= -0.3


@pytest.mark.unit
def test_excessive_deferral_penalty():
    """Email deferred more than 2 times → -0.5 penalty."""
    engine = RewardEngine()
    tracker = MemoryTracker()
    email = make_email("email-1")

    tracker.record_email_received("email-1", step=0)
    for step in range(3):
        tracker.record_action("email-1", Action(action_type="defer_email"), step=step)

    adjustment, breakdown = engine.finalize_episode(tracker, [email], 0.0)

    assert "excessive_deferral_email-1" in breakdown
    assert breakdown["excessive_deferral_email-1"] == pytest.approx(-0.5)
    assert adjustment <= -0.5


@pytest.mark.unit
def test_vip_consistency_bonus():
    """All VIP emails classified as important → +0.3 bonus."""
    engine = RewardEngine()
    tracker = MemoryTracker()
    vip_email = make_email("vip-1", sender_type="VIP")

    tracker.record_email_received("vip-1", step=0)
    tracker.record_action(
        "vip-1",
        Action(action_type="classify_email", value="important"),
        step=0,
    )

    adjustment, breakdown = engine.finalize_episode(tracker, [vip_email], 0.0)

    assert "vip_consistency_bonus" in breakdown
    assert breakdown["vip_consistency_bonus"] == pytest.approx(0.3)


@pytest.mark.unit
def test_early_classification_bonus():
    """Email classified on its first step → +0.1 bonus."""
    engine = RewardEngine()
    tracker = MemoryTracker()
    email = make_email("email-1")

    # First seen at step 5, classified at step 5 (same step = first step)
    tracker.record_email_received("email-1", step=5)
    tracker.record_action(
        "email-1",
        Action(action_type="classify_email", value="important"),
        step=5,
    )

    adjustment, breakdown = engine.finalize_episode(tracker, [email], 0.0)

    assert "early_classification_bonus_email-1" in breakdown
    assert breakdown["early_classification_bonus_email-1"] == pytest.approx(0.1)


@pytest.mark.unit
def test_easy_task_reward_components():
    """EASY task only scores classification; other actions contribute 0.0."""
    engine = RewardEngine()
    tracker = MemoryTracker()
    email = make_email()

    # Correct classification → +0.4 (updated from +0.2)
    reward = engine.score_step(
        Action(action_type="classify_email", value="important"),
        email,
        EASY_TASK,
        tracker,
        step_count=0,
    )
    assert "classification" in reward.breakdown
    assert reward.breakdown["classification"] == pytest.approx(0.4)

    # Prioritization not in EASY → no prioritization key
    reward2 = engine.score_step(
        Action(action_type="prioritize_email", value="high"),
        email,
        EASY_TASK,
        tracker,
        step_count=2,
    )
    assert "prioritization" not in reward2.breakdown

    # Routing not in EASY → no routing key
    reward3 = engine.score_step(
        Action(action_type="route_email", value="support"),
        email,
        EASY_TASK,
        tracker,
        step_count=2,
    )
    assert "routing" not in reward3.breakdown


@pytest.mark.unit
def test_medium_task_reward_components():
    """MEDIUM task scores classification, prioritization, and routing."""
    engine = RewardEngine()
    tracker = MemoryTracker()
    email = make_email()

    # Classification
    r1 = engine.score_step(
        Action(action_type="classify_email", value="important"),
        email,
        MEDIUM_TASK,
        tracker,
        step_count=2,
    )
    assert "classification" in r1.breakdown

    # Prioritization
    r2 = engine.score_step(
        Action(action_type="prioritize_email", value="high"),
        email,
        MEDIUM_TASK,
        tracker,
        step_count=2,
    )
    assert "prioritization" in r2.breakdown

    # Routing
    r3 = engine.score_step(
        Action(action_type="route_email", value="support"),
        email,
        MEDIUM_TASK,
        tracker,
        step_count=2,
    )
    assert "routing" in r3.breakdown

    # Reply not in MEDIUM → no reply key
    r4 = engine.score_step(
        Action(action_type="generate_reply", value="Hello, I can help you with that."),
        email,
        MEDIUM_TASK,
        tracker,
        step_count=2,
    )
    assert "reply" not in r4.breakdown


@pytest.mark.unit
def test_hard_task_reward_components():
    """HARD task scores all four components."""
    engine = RewardEngine()
    tracker = MemoryTracker()
    email = make_email()

    for action, key in [
        (Action(action_type="classify_email", value="important"), "classification"),
        (Action(action_type="prioritize_email", value="high"), "prioritization"),
        (Action(action_type="route_email", value="support"), "routing"),
        (
            Action(
                action_type="generate_reply",
                value="Hello, I can help you with that issue.",
            ),
            "reply",
        ),
    ]:
        reward = engine.score_step(action, email, HARD_TASK, tracker, step_count=2)
        assert key in reward.breakdown, f"Expected '{key}' in breakdown for HARD task"


# ---------------------------------------------------------------------------
# Property test: Property 13 — Task-scoped grading
# ---------------------------------------------------------------------------

# Feature: openenv-email-ops, Property 13: Task-scoped grading — inactive components score zero
@pytest.mark.property
@settings(max_examples=100)
@given(
    st.sampled_from([
        ("classify_email", "important", "classification"),
        ("prioritize_email", "high", "prioritization"),
        ("route_email", "support", "routing"),
        ("generate_reply", "Hello, this is a reply to your message.", "reply"),
    ])
)
def test_task_scoped_grading_inactive_zero(action_tuple):
    """
    **Validates: Requirements 6.4**

    For any task configuration, submitting an action whose type is not in the
    task's reward_components SHALL result in a reward breakdown contribution of
    0.0 for that action type (the key should not appear in breakdown at all).
    """
    action_type, value, component = action_tuple

    # Build a task that explicitly excludes this component
    all_components = ["classification", "prioritization", "routing", "reply"]
    active_components = [c for c in all_components if c != component]

    task = make_task("hard", active_components)
    engine = RewardEngine()
    tracker = MemoryTracker()
    email = make_email()

    action = Action(action_type=action_type, value=value)
    reward = engine.score_step(action, email, task, tracker, step_count=2)

    # The inactive component must not appear in the breakdown (contributes 0.0)
    assert component not in reward.breakdown, (
        f"Component '{component}' should not be scored when not in reward_components, "
        f"but found in breakdown: {reward.breakdown}"
    )
