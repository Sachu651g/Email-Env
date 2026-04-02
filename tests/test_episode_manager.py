"""Tests for EpisodeManager."""

from __future__ import annotations

import pytest

from openenv_email_ops.episode_manager import EpisodeManager
from openenv_email_ops.models import Email, GroundTruth


def make_email(
    email_id: str = "e1",
    sender_type: str = "customer",
    urgency_score: float = 0.5,
) -> Email:
    return Email(
        id=email_id,
        subject="Test subject",
        body="Test body",
        sender_type=sender_type,
        urgency_score=urgency_score,
        ground_truth=GroundTruth(
            correct_classification="important",
            correct_priority="medium",
            correct_route="support",
        ),
    )


# --- Unit tests ---

@pytest.mark.unit
class TestCurrentEmail:
    def test_returns_first_email(self):
        e1 = make_email("e1")
        e2 = make_email("e2")
        mgr = EpisodeManager([e1, e2], max_steps=10)
        assert mgr.current_email() is e1

    def test_returns_none_when_empty(self):
        mgr = EpisodeManager([], max_steps=10)
        assert mgr.current_email() is None


@pytest.mark.unit
class TestAdvance:
    def test_removes_front_email(self):
        e1 = make_email("e1")
        e2 = make_email("e2")
        mgr = EpisodeManager([e1, e2], max_steps=10)
        mgr.advance()
        assert mgr.current_email() is e2

    def test_advance_on_empty_is_safe(self):
        mgr = EpisodeManager([], max_steps=10)
        mgr.advance()  # should not raise
        assert mgr.current_email() is None


@pytest.mark.unit
class TestDefer:
    def test_moves_front_email_to_end(self):
        e1 = make_email("e1")
        e2 = make_email("e2")
        e3 = make_email("e3")
        mgr = EpisodeManager([e1, e2, e3], max_steps=10)
        mgr.defer(e1)
        assert mgr.current_email() is e2
        assert mgr.remaining_emails()[-1] is e1

    def test_defer_single_email_stays(self):
        e1 = make_email("e1")
        mgr = EpisodeManager([e1], max_steps=10)
        mgr.defer(e1)
        assert mgr.current_email() is e1
        assert len(mgr.remaining_emails()) == 1


@pytest.mark.unit
class TestIsDone:
    def test_not_done_initially(self):
        mgr = EpisodeManager([make_email()], max_steps=10)
        assert not mgr.is_done()

    def test_done_when_inbox_empty(self):
        mgr = EpisodeManager([], max_steps=10)
        assert mgr.is_done()

    def test_done_when_max_steps_reached(self):
        mgr = EpisodeManager([make_email()], max_steps=3)
        mgr.increment_step()
        mgr.increment_step()
        mgr.increment_step()
        assert mgr.is_done()

    def test_not_done_before_max_steps(self):
        mgr = EpisodeManager([make_email()], max_steps=3)
        mgr.increment_step()
        mgr.increment_step()
        assert not mgr.is_done()


@pytest.mark.unit
class TestInboxSummary:
    def test_counts_by_sender_type(self):
        emails = [
            make_email("e1", sender_type="customer"),
            make_email("e2", sender_type="VIP"),
            make_email("e3", sender_type="customer"),
        ]
        mgr = EpisodeManager(emails, max_steps=10)
        summary = mgr.inbox_summary()
        assert summary.counts_by_sender_type["customer"] == 2
        assert summary.counts_by_sender_type["VIP"] == 1

    def test_urgency_distribution_low(self):
        emails = [make_email("e1", urgency_score=0.1)]
        mgr = EpisodeManager(emails, max_steps=10)
        summary = mgr.inbox_summary()
        assert summary.urgency_distribution["low"] == 1
        assert summary.urgency_distribution["medium"] == 0
        assert summary.urgency_distribution["high"] == 0

    def test_urgency_distribution_medium(self):
        emails = [make_email("e1", urgency_score=0.5)]
        mgr = EpisodeManager(emails, max_steps=10)
        summary = mgr.inbox_summary()
        assert summary.urgency_distribution["medium"] == 1

    def test_urgency_distribution_high(self):
        emails = [make_email("e1", urgency_score=0.9)]
        mgr = EpisodeManager(emails, max_steps=10)
        summary = mgr.inbox_summary()
        assert summary.urgency_distribution["high"] == 1

    def test_urgency_boundary_0_4_is_medium(self):
        emails = [make_email("e1", urgency_score=0.4)]
        mgr = EpisodeManager(emails, max_steps=10)
        summary = mgr.inbox_summary()
        assert summary.urgency_distribution["medium"] == 1

    def test_urgency_boundary_0_7_is_medium(self):
        emails = [make_email("e1", urgency_score=0.7)]
        mgr = EpisodeManager(emails, max_steps=10)
        summary = mgr.inbox_summary()
        assert summary.urgency_distribution["medium"] == 1

    def test_empty_inbox_summary(self):
        mgr = EpisodeManager([], max_steps=10)
        summary = mgr.inbox_summary()
        assert summary.counts_by_sender_type == {}
        assert summary.urgency_distribution == {"low": 0, "medium": 0, "high": 0}


@pytest.mark.unit
class TestRemainingEmails:
    def test_returns_copy(self):
        e1 = make_email("e1")
        mgr = EpisodeManager([e1], max_steps=10)
        remaining = mgr.remaining_emails()
        remaining.clear()
        assert len(mgr.remaining_emails()) == 1

    def test_reflects_current_inbox(self):
        e1 = make_email("e1")
        e2 = make_email("e2")
        mgr = EpisodeManager([e1, e2], max_steps=10)
        mgr.advance()
        assert mgr.remaining_emails() == [e2]


@pytest.mark.unit
class TestStepCount:
    def test_initial_step_count_is_zero(self):
        mgr = EpisodeManager([make_email()], max_steps=10)
        assert mgr.step_count == 0

    def test_increment_step(self):
        mgr = EpisodeManager([make_email()], max_steps=10)
        mgr.increment_step()
        mgr.increment_step()
        assert mgr.step_count == 2
