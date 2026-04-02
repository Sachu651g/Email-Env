"""Per-episode metrics accumulation for openenv-email-ops."""

from __future__ import annotations


class MetricsTracker:
    """Accumulates per-episode metrics for classification, prioritization, routing, VIP handling, and deferral."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._total_reward: float = 0.0
        self._classification_correct: int = 0
        self._classification_total: int = 0
        self._prioritization_correct: int = 0
        self._prioritization_total: int = 0
        self._routing_correct: int = 0
        self._routing_total: int = 0
        self._vip_handled_correct: int = 0
        self._vip_total: int = 0
        self._deferral_count: int = 0

    def record_classification(self, correct: bool) -> None:
        """Record a classification attempt."""
        self._classification_total += 1
        if correct:
            self._classification_correct += 1

    def record_prioritization(self, correct: bool) -> None:
        """Record a prioritization attempt."""
        self._prioritization_total += 1
        if correct:
            self._prioritization_correct += 1

    def record_routing(self, correct: bool) -> None:
        """Record a routing attempt."""
        self._routing_total += 1
        if correct:
            self._routing_correct += 1

    def record_reward(self, reward: float) -> None:
        """Accumulate reward."""
        self._total_reward += reward

    def record_deferral(self) -> None:
        """Record a deferral action."""
        self._deferral_count += 1

    def record_vip_handled(self, handled: bool) -> None:
        """Track VIP email handling."""
        self._vip_total += 1
        if handled:
            self._vip_handled_correct += 1

    def get_metrics(self) -> dict:
        """Return dict with all 6 required metric fields."""
        return {
            "total_reward": self._total_reward,
            "classification_accuracy": (
                self._classification_correct / self._classification_total
                if self._classification_total > 0
                else 0.0
            ),
            "prioritization_accuracy": (
                self._prioritization_correct / self._prioritization_total
                if self._prioritization_total > 0
                else 0.0
            ),
            "routing_accuracy": (
                self._routing_correct / self._routing_total
                if self._routing_total > 0
                else 0.0
            ),
            "vip_handling_rate": (
                self._vip_handled_correct / self._vip_total
                if self._vip_total > 0
                else 0.0
            ),
            "deferral_count": self._deferral_count,
        }
