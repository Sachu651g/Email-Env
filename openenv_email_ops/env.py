"""EmailOpsEnv: main OpenEnv interface for openenv-email-ops."""

from __future__ import annotations

import logging
from typing import Any

from openenv_email_ops.episode_manager import EpisodeManager
from openenv_email_ops.inbox_generator import InboxGenerator
from openenv_email_ops.memory_tracker import MemoryTracker
from openenv_email_ops.metrics import MetricsTracker
from openenv_email_ops.models import (
    Action,
    InboxSummary,
    Observation,
    Reward,
    TaskConfig,
)
from openenv_email_ops.parser import Parser
from openenv_email_ops.reward_engine import RewardEngine

logger = logging.getLogger("openenv.email_ops")


class EmailOpsEnv:
    """OpenEnv-compliant email operations environment."""

    def __init__(
        self,
        task_config: TaskConfig,
        inbox_size: int = 10,
        max_steps: int = 50,
        seed: int = 42,
        log_level: int = logging.WARNING,
    ) -> None:
        self._task_config = task_config
        self._inbox_size = inbox_size
        self._max_steps = max_steps
        self._seed = seed

        logging.getLogger("openenv.email_ops").setLevel(log_level)

        self._inbox_generator = InboxGenerator()
        self._reward_engine = RewardEngine()
        self._memory_tracker = MemoryTracker()
        self._metrics_tracker = MetricsTracker()

        self._episode_manager: EpisodeManager | None = None
        self._action_history: list[Action] = []
        self._done: bool = False
        self._episode_reward: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Observation:
        """Initialize a new episode and return the initial Observation."""
        if seed is not None:
            self._seed = seed

        # Clear trackers
        self._memory_tracker.reset()
        self._metrics_tracker.reset()
        self._action_history = []
        self._done = False
        self._episode_reward = 0.0

        # Generate inbox
        inbox = self._inbox_generator.generate(self._inbox_size, self._seed, difficulty=self._task_config.difficulty)

        # Create episode manager
        self._episode_manager = EpisodeManager(inbox, self._max_steps)

        # Record each email as received at step 0
        for email in inbox:
            self._memory_tracker.record_email_received(email.id, step=0)

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Advance the episode by one step and return (obs, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

        assert self._episode_manager is not None, "Call reset() before step()"

        current_email = self._episode_manager.current_email()

        # Record action in memory tracker
        step_count = self._episode_manager.step_count
        if current_email is not None:
            self._memory_tracker.record_action(current_email.id, action, step_count, sender_type=current_email.sender_type)

        # Score the step
        reward = self._reward_engine.score_step(
            action=action,
            email=current_email,  # type: ignore[arg-type]
            task_config=self._task_config,
            memory_tracker=self._memory_tracker,
            step_count=step_count,
        ) if current_email is not None else Reward(step_reward=0.0, episode_reward=0.0, breakdown={})

        # Update MetricsTracker based on action type
        if current_email is not None:
            if action.action_type == "classify_email":
                correct = action.value == current_email.ground_truth.correct_classification
                self._metrics_tracker.record_classification(correct)
                # VIP handling: record if VIP email was classified as important
                if current_email.sender_type == "VIP":
                    self._metrics_tracker.record_vip_handled(action.value == "important")
            elif action.action_type == "prioritize_email":
                correct = action.value == current_email.ground_truth.correct_priority
                self._metrics_tracker.record_prioritization(correct)
            elif action.action_type == "route_email":
                correct = action.value == current_email.ground_truth.correct_route
                self._metrics_tracker.record_routing(correct)
            elif action.action_type == "defer_email":
                self._metrics_tracker.record_deferral()

        # Accumulate episode reward
        self._episode_reward += reward.step_reward
        self._episode_reward = max(-1.0, min(1.0, self._episode_reward))

        # Advance or defer
        if action.action_type == "defer_email" and current_email is not None:
            self._episode_manager.defer(current_email)
        else:
            self._episode_manager.advance()

        # Increment step count
        self._episode_manager.increment_step()

        # Record action in history
        self._action_history.append(action)

        # Check done
        done = self._episode_manager.is_done()
        info: dict[str, Any] = {}

        if done:
            self._done = True
            # Finalize episode rewards
            inbox_snapshot = self._episode_manager.remaining_emails()
            # Include all emails (need full inbox for finalize)
            adjustment, delayed_rewards = self._reward_engine.finalize_episode(
                memory_tracker=self._memory_tracker,
                inbox=inbox_snapshot,
                episode_reward=self._episode_reward,
            )
            self._episode_reward += adjustment
            self._episode_reward = max(-1.0, min(1.0, self._episode_reward))
            # Normalize final episode score to [0.0, 1.0] for OpenEnv compliance
            self._episode_reward = max(0.0, min(1.0, self._episode_reward))
            self._metrics_tracker.record_reward(self._episode_reward)
            info["metrics"] = self._metrics_tracker.get_metrics()
            info["delayed_rewards"] = delayed_rewards

        # Build reward with updated episode_reward
        final_reward = Reward(
            step_reward=reward.step_reward,
            episode_reward=self._episode_reward,
            breakdown=reward.breakdown,
        )

        obs = self._build_observation()

        logger.debug(
            "step=%d action_type=%s reward_breakdown=%s done=%s",
            self._episode_manager.step_count,
            action.action_type,
            reward.breakdown,
            done,
        )

        return obs, final_reward, done, info

    @classmethod
    def from_yaml(cls, yaml_path: str, task_id: str, **kwargs) -> "EmailOpsEnv":
        """Create an EmailOpsEnv from an openenv.yaml file and a task_id.

        Loads the YAML, finds the task with matching task_id, constructs a
        TaskConfig, and returns a configured EmailOpsEnv instance.
        """
        parser = Parser()
        data = parser.parse_yaml(yaml_path)

        tasks = data.get("tasks", [])
        task_data = next((t for t in tasks if t["task_id"] == task_id), None)
        if task_data is None:
            available = [t["task_id"] for t in tasks]
            raise ValueError(
                f"task_id '{task_id}' not found in {yaml_path}. "
                f"Available task IDs: {available}"
            )

        task_config = TaskConfig(**task_data)

        # Use task values as defaults, allow kwargs to override
        init_kwargs = {
            "inbox_size": task_config.inbox_size,
            "max_steps": task_config.max_steps,
        }
        init_kwargs.update(kwargs)

        return cls(task_config=task_config, **init_kwargs)

    def state(self) -> dict:
        """Return a serializable snapshot of current internal state."""
        step_count = self._episode_manager.step_count if self._episode_manager else 0
        inbox_remaining = (
            len(self._episode_manager.remaining_emails())
            if self._episode_manager
            else 0
        )
        return {
            "step_count": step_count,
            "done": self._done,
            "episode_reward": self._episode_reward,
            "inbox_size": inbox_remaining,
            "task_id": self._task_config.task_id,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Build an Observation from current episode state."""
        if self._episode_manager is None:
            return Observation(
                current_email=None,
                inbox_summary=InboxSummary(
                    counts_by_sender_type={}, urgency_distribution={}
                ),
                action_history=[],
                step_count=0,
            )

        current_email = self._episode_manager.current_email()
        inbox_summary = self._episode_manager.inbox_summary()
        step_count = self._episode_manager.step_count

        return Observation(
            current_email=current_email,
            inbox_summary=inbox_summary,
            action_history=list(self._action_history),
            step_count=step_count,
        )
