"""MemoryTracker: tracks per-email action history within an episode."""

from __future__ import annotations

from openenv_email_ops.models import Action


class MemoryTracker:
    """Tracks per-email decision history and timestamps within an episode."""

    def __init__(self) -> None:
        # Maps email_id -> list of (Action, step) tuples
        self._history: dict[str, list[tuple[Action, int]]] = {}
        # Maps email_id -> step when first seen
        self._first_seen: dict[str, int] = {}

    def record_email_received(self, email_id: str, step: int) -> None:
        """Record when an email first appeared in the inbox."""
        if email_id not in self._first_seen:
            self._first_seen[email_id] = step

    def record_action(self, email_id: str, action: Action, step: int) -> None:
        """Record that an action was taken on an email at a given step."""
        if email_id not in self._history:
            self._history[email_id] = []
        self._history[email_id].append((action, step))

    def deferral_count(self, email_id: str) -> int:
        """Return how many times an email has been deferred."""
        return sum(
            1
            for action, _ in self._history.get(email_id, [])
            if action.action_type == "defer_email"
        )

    def steps_since_received(self, email_id: str, current_step: int) -> int:
        """Return how many steps have passed since the email was first seen."""
        first = self._first_seen.get(email_id)
        if first is None:
            return 0
        return current_step - first

    def all_vip_handled(self, vip_email_ids: list[str]) -> bool:
        """Return True if all VIP emails have been classified as 'important'."""
        return all(self.was_classified_important(eid) for eid in vip_email_ids)

    def was_classified_important(self, email_id: str) -> bool:
        """Return True if the email was classified as 'important'."""
        return any(
            action.action_type == "classify_email" and action.value == "important"
            for action, _ in self._history.get(email_id, [])
        )

    def reset(self) -> None:
        """Clear all tracking state."""
        self._history.clear()
        self._first_seen.clear()
