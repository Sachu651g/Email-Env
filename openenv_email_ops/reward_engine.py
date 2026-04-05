"""RewardEngine: aggregates grader outputs and applies bonuses/penalties."""

from __future__ import annotations

import logging

from openenv_email_ops.graders import (
    ClassificationGrader,
    PrioritizationGrader,
    ReplyGrader,
    RoutingGrader,
)
from openenv_email_ops.memory_tracker import MemoryTracker
from openenv_email_ops.models import Action, Email, Reward, TaskConfig

logger = logging.getLogger("openenv.email_ops")

_classification_grader = ClassificationGrader()
_prioritization_grader = PrioritizationGrader()
_routing_grader = RoutingGrader()
_reply_grader = ReplyGrader()

# Valid values per action type — used for invalid-action penalty
_VALID_VALUES: dict[str, set[str]] = {
    "classify_email": {"spam", "important", "promotion"},
    "prioritize_email": {"low", "medium", "high"},
    "route_email": {"support", "sales", "escalation"},
}

# Coherent (classification, route) pairs for hard-task reasoning consistency
_COHERENT_PAIRS: set[tuple[str, str]] = {
    ("important", "escalation"),
    ("important", "sales"),
    ("important", "support"),
    ("spam", "support"),
    ("promotion", "sales"),
}


class RewardEngine:
    """Computes step rewards and episode-level adjustments."""

    def _check_reasoning_consistency(
        self, email_id: str, memory_tracker: MemoryTracker
    ) -> float:
        """Return +0.15 for coherent classify→route pair, -0.15 for contradictory, 0.0 otherwise."""
        actions = {
            a.action_type: (a.value or "").strip().lower()
            for a, _ in memory_tracker._history.get(email_id, [])
        }
        classification = actions.get("classify_email")
        route = actions.get("route_email")
        if not classification or not route:
            return 0.0
        if classification == "spam" and route == "escalation":
            return -0.15
        if (classification, route) in _COHERENT_PAIRS:
            return 0.15
        return 0.0

    def score_step(
        self,
        action: Action,
        email: Email,
        task_config: TaskConfig,
        memory_tracker: MemoryTracker,
        step_count: int,
    ) -> Reward:
        """Score a single step action and return a Reward.

        Only components listed in task_config.reward_components are scored.
        Each grader call is wrapped in try/except; errors default to 0.0.
        """
        components = task_config.reward_components
        breakdown: dict[str, float] = {}
        total = 0.0

        # --- invalid-action penalty (checked before graders) ---
        if action.action_type in _VALID_VALUES:
            val = (action.value or "").strip().lower()
            if val not in _VALID_VALUES[action.action_type]:
                breakdown["invalid_action"] = -0.1
                return Reward(step_reward=-0.1, episode_reward=0.0, breakdown=breakdown)

        # --- repetition penalty ---
        prior_actions = memory_tracker._history.get(email.id, [])
        is_duplicate = any(a.action_type == action.action_type for a, _ in prior_actions)
        if is_duplicate:
            breakdown["repetition_penalty"] = -0.1
            total += -0.1

        # --- classify_email ---
        if action.action_type == "classify_email":
            if "classification" in components:
                score = 0.0
                try:
                    raw = _classification_grader.score(
                        action.value or "",
                        email.ground_truth.correct_classification,
                    )
                    # 1.0 → +0.4, 0.5 (adjacent) → +0.2, 0.0 → -0.2
                    if raw == 1.0:
                        score = 0.4
                    elif raw == 0.0:
                        score = -0.2
                    else:
                        score = raw * 0.4
                except Exception:
                    logger.warning(
                        "ClassificationGrader raised an unexpected error; defaulting to 0.0"
                    )
                breakdown["classification"] = score
                total += score

            # --- consistency penalty (context memory) ---
            history = memory_tracker.get_classification_history(email.sender_type)
            if len(history) >= 2 and history[-1] != history[-2]:
                breakdown["consistency_penalty"] = -0.1
                total += -0.1

        # --- prioritize_email ---
        elif action.action_type == "prioritize_email":
            if "prioritization" in components:
                score = 0.0
                try:
                    raw = _prioritization_grader.score(
                        action.value or "",
                        email.ground_truth.correct_priority,
                    )
                    score = 0.2 if raw == 1.0 else 0.0
                except Exception:
                    logger.warning(
                        "PrioritizationGrader raised an unexpected error; defaulting to 0.0"
                    )
                breakdown["prioritization"] = score
                total += score

        # --- route_email ---
        elif action.action_type == "route_email":
            if "routing" in components:
                score = 0.0
                try:
                    raw = _routing_grader.score(
                        action.value or "",
                        email.ground_truth.correct_route,
                    )
                    score = 0.2 if raw == 1.0 else 0.0
                except Exception:
                    logger.warning(
                        "RoutingGrader raised an unexpected error; defaulting to 0.0"
                    )
                breakdown["routing"] = score
                total += score

        # --- generate_reply ---
        elif action.action_type == "generate_reply":
            if "reply" in components:
                score = 0.0
                try:
                    if task_config.difficulty == "hard":
                        raw = _reply_grader.score_hard(action.value or "", email)
                    else:
                        raw = _reply_grader.score(action.value or "", email)
                    score = raw * 0.2
                except Exception:
                    logger.warning(
                        "ReplyGrader raised an unexpected error; defaulting to 0.0"
                    )
                breakdown["reply"] = score
                total += score

        # --- defer_email: always apply deferral penalty ---
        elif action.action_type == "defer_email":
            breakdown["deferral_penalty"] = -0.05
            total += -0.05

        # --- efficiency bonus: +0.1 if first step on this email ---
        if step_count <= 1:
            breakdown["efficiency_bonus"] = 0.1
            total += 0.1

        # --- hard-task reasoning consistency (after all 4 action types applied) ---
        if task_config.difficulty == "hard":
            email_actions = {a.action_type for a, _ in memory_tracker._history.get(email.id, [])}
            all_four = {"classify_email", "prioritize_email", "route_email", "generate_reply"}
            if all_four.issubset(email_actions):
                consistency = self._check_reasoning_consistency(email.id, memory_tracker)
                if consistency != 0.0:
                    key = "reasoning_consistency_bonus" if consistency > 0 else "reasoning_consistency_penalty"
                    breakdown[key] = consistency
                    total += consistency

        return Reward(step_reward=total, episode_reward=0.0, breakdown=breakdown)

    def finalize_episode(
        self,
        memory_tracker: MemoryTracker,
        inbox: list[Email],
        episode_reward: float,
    ) -> tuple[float, dict]:
        """Compute end-of-episode reward adjustments.

        Returns (total_adjustment, breakdown_dict).
        """
        breakdown: dict[str, float] = {}
        total = 0.0

        vip_emails = [e for e in inbox if e.sender_type == "VIP"]

        # VIP ignore penalty: -0.3 per VIP email not classified as important within 3 steps
        for email in vip_emails:
            try:
                steps = memory_tracker.steps_since_received(email.id, current_step=0)
                classified = memory_tracker.was_classified_important(email.id)
                # Check if classified within 3 steps of receiving
                first_seen = memory_tracker._first_seen.get(email.id)
                classified_in_time = False
                if classified and first_seen is not None:
                    for action, step in memory_tracker._history.get(email.id, []):
                        if (
                            action.action_type == "classify_email"
                            and action.value == "important"
                            and (step - first_seen) <= 3
                        ):
                            classified_in_time = True
                            break
                if not classified_in_time:
                    penalty_key = f"vip_ignore_penalty_{email.id}"
                    breakdown[penalty_key] = -0.3
                    total += -0.3
            except Exception:
                logger.warning(
                    "Error computing VIP ignore penalty for email %s; skipping", email.id
                )

        # Excessive deferral penalty: -0.5 per email deferred > 2 times
        all_email_ids = {e.id for e in inbox}
        # Also include emails tracked in memory that may have been processed
        all_email_ids |= set(memory_tracker._history.keys())
        for email_id in all_email_ids:
            try:
                count = memory_tracker.deferral_count(email_id)
                if count > 2:
                    penalty_key = f"excessive_deferral_{email_id}"
                    breakdown[penalty_key] = -0.5
                    total += -0.5
            except Exception:
                logger.warning(
                    "Error computing excessive deferral penalty for email %s; skipping",
                    email_id,
                )

        # VIP consistency bonus: +0.3 if all VIP emails were handled correctly
        if vip_emails:
            try:
                vip_ids = [e.id for e in vip_emails]
                if memory_tracker.all_vip_handled(vip_ids):
                    breakdown["vip_consistency_bonus"] = 0.3
                    total += 0.3
            except Exception:
                logger.warning("Error computing VIP consistency bonus; skipping")

        # Early classification bonus: +0.1 per email classified on its first step
        try:
            for email_id, history in memory_tracker._history.items():
                first_seen = memory_tracker._first_seen.get(email_id)
                if first_seen is None:
                    continue
                for action, step in history:
                    if action.action_type == "classify_email" and step == first_seen:
                        bonus_key = f"early_classification_bonus_{email_id}"
                        breakdown[bonus_key] = 0.1
                        total += 0.1
                        break
        except Exception:
            logger.warning("Error computing early classification bonus; skipping")

        # Context memory consistency bonus: +0.2 per sender_type where all classifications
        # were correct and consistent throughout the episode
        try:
            # Build ground-truth lookup: sender_type -> correct_classification
            gt_by_sender: dict[str, str] = {}
            for email in inbox:
                gt_by_sender[email.sender_type] = email.ground_truth.correct_classification

            for sender_type, history in memory_tracker.get_all_classification_histories().items():
                if len(history) >= 2 and len(set(history)) == 1:
                    expected = gt_by_sender.get(sender_type)
                    if expected and history[0] == expected:
                        bonus_key = f"consistency_bonus_{sender_type}"
                        breakdown[bonus_key] = 0.2
                        total += 0.2
        except Exception:
            logger.warning("Error computing context memory consistency bonus; skipping")

        return total, breakdown
