"""Tests for inference.py baseline script."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.unit
def test_inference_missing_api_key(monkeypatch):
    """Verify that running inference.py without OPENAI_API_KEY causes sys.exit(1)."""
    # Remove OPENAI_API_KEY from the environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        env={k: v for k, v in __import__("os").environ.items() if k != "OPENAI_API_KEY"},
    )

    assert result.returncode == 1
    assert "OPENAI_API_KEY" in result.stderr
