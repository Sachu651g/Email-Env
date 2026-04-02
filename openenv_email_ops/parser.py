"""Parser: deserializes LLM output and YAML config into Pydantic models."""

from __future__ import annotations

import json
import re

import yaml

from openenv_email_ops.models import Action


class Parser:
    """Parses LLM output strings and YAML config files."""

    def parse_action(self, raw: str) -> Action:
        """Parse an LLM output string into an Action model.

        Tries JSON parsing first, then falls back to plain text like
        "classify_email: spam" or just "classify_email".
        """
        raw = raw.strip()

        # Try JSON parsing first
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "action_type" in data:
                return Action(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fall back to plain text: "action_type: value" or "action_type"
        # Handle optional surrounding quotes or markdown code fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

        if ":" in raw:
            parts = raw.split(":", 1)
            action_type = parts[0].strip()
            value = parts[1].strip() or None
        else:
            action_type = raw.strip()
            value = None

        return Action(action_type=action_type, value=value)

    def parse_yaml(self, path: str) -> dict:
        """Load and return a YAML file as a dict."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
