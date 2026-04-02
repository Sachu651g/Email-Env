"""EpisodeManager: manages inbox state and step progression for an episode."""

from __future__ import annotations

from openenv_email_ops.models import Email, InboxSummary


class EpisodeManager:
    """Manages the inbox queue, step count, and termination conditions for an episode."""

    def __init__(self, inbox: list[Email], max_steps: int) -> None:
        self._inbox: list[Email] = list(inbox)
        self._max_steps = max_steps
        self._step_count = 0

    @property
    def step_count(self) -> int:
        """Current step count."""
        return self._step_count

    def increment_step(self) -> None:
        """Increment step_count by 1."""
        self._step_count += 1

    def current_email(self) -> Email | None:
        """Return the front email in the inbox, or None if empty."""
        return self._inbox[0] if self._inbox else None

    def advance(self) -> None:
        """Remove the front email (mark as processed)."""
        if self._inbox:
            self._inbox.pop(0)

    def defer(self, email: Email) -> None:
        """Move the given email to the end of the inbox."""
        if self._inbox and self._inbox[0].id == email.id:
            self._inbox.pop(0)
        self._inbox.append(email)

    def is_done(self) -> bool:
        """Return True when inbox is empty OR step_count >= max_steps."""
        return len(self._inbox) == 0 or self._step_count >= self._max_steps

    def inbox_summary(self) -> InboxSummary:
        """Return counts by sender_type and urgency distribution for remaining emails."""
        counts_by_sender_type: dict[str, int] = {}
        urgency_distribution: dict[str, int] = {"low": 0, "medium": 0, "high": 0}

        for email in self._inbox:
            counts_by_sender_type[email.sender_type] = (
                counts_by_sender_type.get(email.sender_type, 0) + 1
            )
            score = email.urgency_score
            if score < 0.4:
                urgency_distribution["low"] += 1
            elif score <= 0.7:
                urgency_distribution["medium"] += 1
            else:
                urgency_distribution["high"] += 1

        return InboxSummary(
            counts_by_sender_type=counts_by_sender_type,
            urgency_distribution=urgency_distribution,
        )

    def remaining_emails(self) -> list[Email]:
        """Return a copy of the current inbox list."""
        return list(self._inbox)
