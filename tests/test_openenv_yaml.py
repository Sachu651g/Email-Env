"""Unit tests for openenv.yaml structure and EmailOpsEnv.from_yaml()."""

from __future__ import annotations

import os

import pytest

from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.parser import Parser

YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "openenv.yaml")


@pytest.fixture(scope="module")
def yaml_data() -> dict:
    return Parser().parse_yaml(YAML_PATH)


@pytest.mark.unit
def test_openenv_yaml_structure(yaml_data: dict) -> None:
    """Verify all required top-level keys are present."""
    # metadata keys
    metadata = yaml_data.get("metadata", {})
    for key in ("name", "version", "description", "author", "license"):
        assert key in metadata, f"Missing metadata key: {key}"

    # top-level structural keys
    for key in ("tasks", "observation_schema", "action_schema"):
        assert key in yaml_data, f"Missing top-level key: {key}"


@pytest.mark.unit
def test_all_three_tasks_defined(yaml_data: dict) -> None:
    """Verify easy, medium, and hard tasks are all present."""
    task_ids = {t["task_id"] for t in yaml_data["tasks"]}
    assert "easy" in task_ids
    assert "medium" in task_ids
    assert "hard" in task_ids


@pytest.mark.unit
def test_task_required_fields(yaml_data: dict) -> None:
    """Verify each task has all required fields."""
    required = {"task_id", "description", "difficulty", "max_steps", "inbox_size", "reward_components"}
    for task in yaml_data["tasks"]:
        missing = required - task.keys()
        assert not missing, f"Task '{task.get('task_id')}' missing fields: {missing}"


@pytest.mark.unit
def test_from_yaml_easy() -> None:
    """Verify from_yaml creates env with correct task_config for 'easy'."""
    env = EmailOpsEnv.from_yaml(YAML_PATH, "easy")
    cfg = env._task_config
    assert cfg.task_id == "easy"
    assert cfg.difficulty == "easy"
    assert cfg.max_steps == 30
    assert cfg.inbox_size == 5
    assert cfg.reward_components == ["classification"]
    assert env._inbox_size == 5
    assert env._max_steps == 30


@pytest.mark.unit
def test_from_yaml_medium() -> None:
    """Verify from_yaml creates env with correct task_config for 'medium'."""
    env = EmailOpsEnv.from_yaml(YAML_PATH, "medium")
    cfg = env._task_config
    assert cfg.task_id == "medium"
    assert cfg.difficulty == "medium"
    assert cfg.max_steps == 50
    assert cfg.inbox_size == 8
    assert set(cfg.reward_components) == {"classification", "prioritization", "routing"}
    assert env._inbox_size == 8
    assert env._max_steps == 50


@pytest.mark.unit
def test_from_yaml_hard() -> None:
    """Verify from_yaml creates env with correct task_config for 'hard'."""
    env = EmailOpsEnv.from_yaml(YAML_PATH, "hard")
    cfg = env._task_config
    assert cfg.task_id == "hard"
    assert cfg.difficulty == "hard"
    assert cfg.max_steps == 80
    assert cfg.inbox_size == 10
    assert set(cfg.reward_components) == {"classification", "prioritization", "routing", "reply"}
    assert env._inbox_size == 10
    assert env._max_steps == 80
