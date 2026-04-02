"""Pydantic v2 data models for openenv-email-ops."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


class GroundTruth(BaseModel):
    correct_classification: Literal["spam", "important", "promotion"]
    correct_priority: Literal["low", "medium", "high"]
    correct_route: Literal["support", "sales", "escalation"]


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender_type: Literal["customer", "spammer", "VIP", "internal"]
    urgency_score: float
    ground_truth: GroundTruth

    @field_validator("urgency_score")
    @classmethod
    def validate_urgency_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"urgency_score must be in [0.0, 1.0], got {v}")
        return v


class Action(BaseModel):
    action_type: Literal[
        "classify_email",
        "prioritize_email",
        "route_email",
        "generate_reply",
        "defer_email",
    ]
    value: str | None = None


class InboxSummary(BaseModel):
    counts_by_sender_type: dict[str, int]
    urgency_distribution: dict[str, int]


class Observation(BaseModel):
    current_email: Email | None
    inbox_summary: InboxSummary
    action_history: list[Action]
    step_count: int

    model_config = ConfigDict()

    @field_serializer("current_email")
    def serialize_current_email(self, email: Email | None) -> Any:
        if email is None:
            return None
        return email.model_dump(exclude={"ground_truth"})


class Reward(BaseModel):
    step_reward: float
    episode_reward: float
    breakdown: dict[str, float]


class EpisodeInfo(BaseModel):
    total_reward: float
    classification_accuracy: float
    prioritization_accuracy: float
    routing_accuracy: float
    vip_handling_rate: float
    deferral_count: int
    delayed_rewards: dict[str, float]


class TaskConfig(BaseModel):
    task_id: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    inbox_size: int
    reward_components: list[str]
