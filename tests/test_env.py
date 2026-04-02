"""Tests for EmailOpsEnv — unit tests and property-based tests."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action, Observation, Reward, TaskConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EASY_TASK = TaskConfig(
    task_id="easy",
    description="Easy task",
    difficulty="easy",
    max_steps=50,
    inbox_size=10,
    reward_components=["classification"],
)

_MEDIUM_TASK = TaskConfig(
    task_id="medium",
    description="Medium task",
    difficulty="medium",
    max_steps=50,
    inbox_size=10,
    reward_components=["classification", "prioritization", "routing"],
)

_HARD_TASK = TaskConfig(
    task_id="hard",
    description="Hard task",
    difficulty="hard",
    max_steps=50,
    inbox_size=10,
    reward_components=["classification", "prioritization", "routing", "reply"],
)

_VALID_ACTION_TYPES = [
    "classify_email",
    "prioritize_email",
    "route_email",
    "generate_reply",
    "defer_email",
]

_ACTION_VALUES = {
    "classify_email": ["spam", "important", "promotion"],
    "prioritize_email": ["low", "medium", "high"],
    "route_email": ["support", "sales", "escalation"],
    "generate_reply": ["Hello, thank you for your email. We will get back to you shortly."],
    "defer_email": [None],
}


def _make_env(inbox_size: int = 4, max_steps: int = 50, seed: int = 42) -> EmailOpsEnv:
    return EmailOpsEnv(
        task_config=_EASY_TASK,
        inbox_size=inbox_size,
        max_steps=max_steps,
        seed=seed,
    )


def _random_action_strategy() -> st.SearchStrategy[Action]:
    """Strategy that generates valid Action objects."""
    return st.sampled_from(_VALID_ACTION_TYPES).flatmap(
        lambda at: st.just(
            Action(
                action_type=at,
                value=_ACTION_VALUES[at][0],
            )
        )
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_step_after_done_raises_runtime_error():
    """Req 1.4: step() after done=True raises RuntimeError."""
    env = _make_env(inbox_size=1, max_steps=50)
    env.reset()
    action = Action(action_type="classify_email", value="spam")
    # Exhaust the single email
    _, _, done, _ = env.step(action)
    assert done is True
    with pytest.raises(RuntimeError, match="Episode has ended"):
        env.step(action)


@pytest.mark.unit
def test_empty_inbox_observation():
    """Req 3.3: current_email is None when inbox is empty."""
    env = _make_env(inbox_size=1, max_steps=50)
    env.reset()
    action = Action(action_type="classify_email", value="spam")
    obs, _, done, _ = env.step(action)
    assert done is True
    assert obs.current_email is None


@pytest.mark.unit
def test_delayed_rewards_in_final_info():
    """Req 7.5: final info dict contains 'delayed_rewards' key."""
    env = _make_env(inbox_size=1, max_steps=50)
    env.reset()
    action = Action(action_type="classify_email", value="spam")
    _, _, done, info = env.step(action)
    assert done is True
    assert "delayed_rewards" in info
    assert isinstance(info["delayed_rewards"], dict)


@pytest.mark.unit
def test_reset_clears_state():
    """Req 8.5: reset() clears action_history and resets step_count to 0."""
    env = _make_env(inbox_size=4, max_steps=50)
    obs = env.reset()
    # Take a few steps
    for _ in range(3):
        env.step(Action(action_type="classify_email", value="spam"))
    # Reset
    obs = env.reset()
    assert obs.step_count == 0
    assert obs.action_history == []


@pytest.mark.unit
def test_episode_terminates_on_max_steps():
    """Req 8.2: episode terminates when step_count reaches max_steps."""
    inbox_size = 10
    max_steps = 3
    env = _make_env(inbox_size=inbox_size, max_steps=max_steps)
    env.reset()
    action = Action(action_type="classify_email", value="spam")
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(action)
        steps += 1
    assert steps == max_steps


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: openenv-email-ops, Property 1: Step return structural invariant
@pytest.mark.property
@settings(max_examples=100)
@given(_random_action_strategy())
def test_step_return_structural_invariant(action: Action):
    """
    **Validates: Requirements 1.2, 3.1, 5.11**

    For any valid action submitted to a non-terminal episode, step() SHALL return
    a 4-tuple where the first element is an Observation, second is a Reward,
    third is a bool, and fourth is a dict.
    """
    env = _make_env(inbox_size=4, max_steps=50)
    env.reset()
    result = env.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 4
    obs, reward, done, info = result
    assert isinstance(obs, Observation)
    assert hasattr(obs, "current_email")
    assert hasattr(obs, "inbox_summary")
    assert hasattr(obs, "action_history")
    assert hasattr(obs, "step_count")
    assert isinstance(reward, Reward)
    assert hasattr(reward, "step_reward")
    assert hasattr(reward, "episode_reward")
    assert hasattr(reward, "breakdown")
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# Feature: openenv-email-ops, Property 2: Reset produces clean initial state
@pytest.mark.property
@settings(max_examples=100)
@given(
    st.integers(min_value=4, max_value=20),
    st.integers(min_value=10, max_value=100),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_reset_produces_clean_initial_state(inbox_size: int, max_steps: int, seed: int):
    """
    **Validates: Requirements 1.1, 8.5**

    For any environment configuration, calling reset() SHALL return an Observation
    with step_count == 0, action_history == [], and a non-None current_email.
    """
    env = EmailOpsEnv(
        task_config=_EASY_TASK,
        inbox_size=inbox_size,
        max_steps=max_steps,
        seed=seed,
    )
    obs = env.reset()
    assert obs.step_count == 0
    assert obs.action_history == []
    assert obs.current_email is not None


# Feature: openenv-email-ops, Property 14: Episode terminates on inbox empty or max_steps
@pytest.mark.property
@settings(max_examples=50)
@given(
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_episode_terminates_on_inbox_empty_or_max_steps(
    inbox_size: int, max_steps: int, seed: int
):
    """
    **Validates: Requirements 8.1, 8.2, 8.3**

    For any episode configuration, done SHALL become True either when all emails
    have been processed or when step_count reaches max_steps.
    """
    env = EmailOpsEnv(
        task_config=_EASY_TASK,
        inbox_size=inbox_size,
        max_steps=max_steps,
        seed=seed,
    )
    env.reset()
    action = Action(action_type="classify_email", value="spam")
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(action)
        steps += 1
        # Safety guard — should never exceed inbox_size + max_steps
        if steps > inbox_size + max_steps + 10:
            raise AssertionError("Episode did not terminate as expected")
    # done must be True at this point
    assert done is True
    # step count must be <= max_steps
    assert env._episode_manager.step_count <= max_steps


# Feature: openenv-email-ops, Property 15: Action history length equals step count
@pytest.mark.property
@settings(max_examples=100)
@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_action_history_length_equals_step_count(num_steps: int, seed: int):
    """
    **Validates: Requirements 7.1**

    For any episode in progress, the length of action_history in the current
    Observation SHALL equal step_count.
    """
    env = _make_env(inbox_size=10, max_steps=50, seed=seed)
    env.reset()
    action = Action(action_type="classify_email", value="spam")
    for i in range(num_steps):
        obs, _, done, _ = env.step(action)
        if done:
            break
        assert len(obs.action_history) == obs.step_count
