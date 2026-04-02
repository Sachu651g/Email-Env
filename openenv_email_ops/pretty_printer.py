"""PrettyPrinter: serializes Pydantic models to human-readable text or JSON."""

from __future__ import annotations

import json

from pydantic import BaseModel

from openenv_email_ops.models import Observation


class PrettyPrinter:
    """Serializes environment state for LLM prompts and structured output."""

    def to_text(self, obs: Observation) -> str:
        """Return a human-readable string for LLM prompts.

        Ground truth is NEVER included in the output.
        """
        lines: list[str] = []

        lines.append(f"Step: {obs.step_count}")
        lines.append("")

        # Current email (no ground_truth)
        if obs.current_email is not None:
            email = obs.current_email
            lines.append("=== Current Email ===")
            lines.append(f"Subject: {email.subject}")
            lines.append(f"Body: {email.body}")
            lines.append(f"Sender Type: {email.sender_type}")
            lines.append(f"Urgency Score: {email.urgency_score:.2f}")
        else:
            lines.append("=== Current Email ===")
            lines.append("(inbox empty)")

        lines.append("")

        # Inbox summary
        lines.append("=== Inbox Summary ===")
        counts = obs.inbox_summary.counts_by_sender_type
        if counts:
            for sender_type, count in counts.items():
                lines.append(f"  {sender_type}: {count}")
        else:
            lines.append("  (empty)")

        lines.append("Urgency Distribution:")
        dist = obs.inbox_summary.urgency_distribution
        if dist:
            for bucket, count in dist.items():
                lines.append(f"  {bucket}: {count}")
        else:
            lines.append("  (empty)")

        lines.append("")

        # Action history (last 10)
        lines.append("=== Action History ===")
        history = obs.action_history[-10:] if obs.action_history else []
        if history:
            for i, action in enumerate(history):
                value_str = f" -> {action.value}" if action.value is not None else ""
                lines.append(f"  [{i + 1}] {action.action_type}{value_str}")
        else:
            lines.append("  (no actions yet)")

        return "\n".join(lines)

    def to_json(self, model: BaseModel) -> str:
        """Serialize a Pydantic model to JSON, excluding ground_truth from Email fields."""
        if isinstance(model, Observation):
            # Observation already has a field_serializer that strips ground_truth
            return model.model_dump_json()
        # For all other models, use standard serialization
        return model.model_dump_json()
