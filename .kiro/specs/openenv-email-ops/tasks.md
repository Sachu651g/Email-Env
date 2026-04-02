# Implementation Plan: openenv-email-ops

## Overview

Implement the openenv-email-ops RL environment incrementally, starting with core data models and building up through the environment interface, graders, reward engine, memory tracking, episode management, serialization, inference script, and deployment artifacts.

## Tasks

- [x] 1. Set up project structure and Pydantic data models
  - Create `openenv_email_ops/` package directory with `__init__.py`
  - Implement all Pydantic v2 models in `models.py`: `GroundTruth`, `Email`, `Action`, `InboxSummary`, `Observation`, `Reward`, `EpisodeInfo`, `TaskConfig`
  - Add field validators: `urgency_score` in [0.0, 1.0], `Literal` constraints on all enum fields
  - Configure `Observation` to exclude `ground_truth` from JSON serialization via `model_config`
  - Create `requirements.txt` with pinned versions (pydantic>=2.0, hypothesis, pytest, openai, pyyaml)
  - _Requirements: 1.5, 2.1, 2.5, 3.1, 4.1, 5.11, 10.1, 10.4, 10.5_

  - [ ]* 1.1 Write property test for Action validation (Property 8)
    - **Property 8: Invalid action type raises ValidationError**
    - **Validates: Requirements 1.6, 4.5, 10.5**

  - [ ]* 1.2 Write property test for out-of-range field validation (Property 16)
    - **Property 16: Out-of-range field values raise ValidationError**
    - **Validates: Requirements 10.4**

- [x] 2. Implement InboxGenerator
  - Create `inbox_generator.py` with `InboxGenerator` class
  - Use `random.Random(seed)` for deterministic generation
  - Ensure at least one email per `sender_type` (customer, spammer, VIP, internal) for inbox_size >= 4
  - Attach `GroundTruth` labels to each generated email
  - Inject noise: typos, informal phrasing, ambiguous subjects
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 2.1 Write property test for seeded determinism (Property 3)
    - **Property 3: Seeded episode determinism**
    - **Validates: Requirements 2.2**

  - [ ]* 2.2 Write property test for sender_type coverage (Property 4)
    - **Property 4: Inbox sender_type coverage invariant**
    - **Validates: Requirements 2.4**

  - [ ]* 2.3 Write property test for ground truth completeness (Property 5)
    - **Property 5: Ground truth completeness invariant**
    - **Validates: Requirements 2.5**

- [x] 3. Implement PrettyPrinter and Parser
  - Create `pretty_printer.py` with `PrettyPrinter.to_text()` and `PrettyPrinter.to_json()`
  - `to_json()` must exclude `ground_truth` from serialized output
  - Create `parser.py` with `Parser.parse_action()` (from LLM output string) and `Parser.parse_yaml()`
  - _Requirements: 3.2, 3.4, 10.2, 10.3_

  - [ ]* 3.1 Write property test for ground truth excluded from serialization (Property 6)
    - **Property 6: Ground truth excluded from observation serialization**
    - **Validates: Requirements 3.2**

  - [ ]* 3.2 Write property test for observation round-trip (Property 7)
    - **Property 7: Observation serialization round-trip**
    - **Validates: Requirements 3.4, 10.2, 10.3**

- [x] 4. Implement Graders
  - Create `graders.py` with `ClassificationGrader`, `PrioritizationGrader`, `RoutingGrader`, `ReplyGrader`
  - `ClassificationGrader.score()`, `PrioritizationGrader.score()`, `RoutingGrader.score()` return 0.0 or 1.0
  - `ReplyGrader.score()` checks: length >= 20 chars, greeting present, subject keyword overlap, no placeholder text; returns float in [0.0, 1.0]
  - All graders are stateless and deterministic
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 4.1 Write property test for grader score range (Property 11)
    - **Property 11: All graders return scores in [0.0, 1.0]**
    - **Validates: Requirements 9.1, 9.2, 9.3**

  - [ ]* 4.2 Write property test for reply grader range (Property 10)
    - **Property 10: Reply grader score is in valid range**
    - **Validates: Requirements 5.4, 9.4**

  - [ ]* 4.3 Write property test for grader determinism (Property 12)
    - **Property 12: Grader determinism**
    - **Validates: Requirements 9.5**

  - [ ]* 4.4 Write unit tests for grader scoring
    - `test_classification_correct_reward` — +0.2 for correct classification (Req 5.1)
    - `test_classification_incorrect_penalty` — -0.2 for incorrect classification (Req 5.6)
    - `test_reply_grader_criteria` — each ReplyGrader criterion independently (Req 9.6)

- [x] 5. Implement MemoryTracker
  - Create `memory_tracker.py` with `MemoryTracker` class
  - Implement `record_action()`, `deferral_count()`, `steps_since_received()`, `all_vip_handled()`
  - Track per-email decision timestamps (step indices) for delayed reward computation
  - _Requirements: 7.1, 7.4_

- [x] 6. Implement RewardEngine
  - Create `reward_engine.py` with `RewardEngine` class
  - `score_step()` aggregates grader outputs and applies per-step bonuses/penalties:
    - +0.2 correct classification, +0.2 correct priority, +0.2 correct route, up to +0.2 reply quality
    - +0.1 efficiency bonus for decision within minimum steps
    - -0.2 incorrect classification, -0.05 deferral penalty
  - `finalize_episode()` computes delayed rewards: -0.3 VIP ignore, -0.5 excessive deferral (>2x same email), +0.3 VIP consistency bonus, +0.1 early classification bonus
  - Wrap each grader call in try/except; default to 0.0 and log WARNING on unexpected error
  - Respect `TaskConfig.reward_components` — inactive components contribute 0.0
  - _Requirements: 5.1–5.11, 6.4, 7.2, 7.3, 7.5_

  - [ ]* 6.1 Write property test for task-scoped grading (Property 13)
    - **Property 13: Task-scoped grading — inactive components score zero**
    - **Validates: Requirements 6.4**

  - [ ]* 6.2 Write unit tests for reward edge cases
    - `test_vip_ignore_penalty` — -0.3 for ignoring VIP email (Req 5.7)
    - `test_excessive_deferral_penalty` — -0.5 for deferring same email 3+ times (Req 5.8)
    - `test_vip_consistency_bonus` — +0.3 for handling all VIPs (Req 5.10)
    - `test_early_classification_bonus` — +0.1 for classifying on first step (Req 7.3)
    - `test_easy_task_reward_components` — EASY task scores only classification (Req 6.1)
    - `test_medium_task_reward_components` — MEDIUM task scores 3 components (Req 6.2)
    - `test_hard_task_reward_components` — HARD task scores 4 components (Req 6.3)

- [x] 7. Implement EpisodeManager
  - Create `episode_manager.py` with `EpisodeManager` class
  - Implement `current_email()`, `advance()`, `defer()`, `is_done()`, `inbox_summary()`
  - `defer()` moves email to end of inbox list
  - `is_done()` returns True when inbox is empty
  - _Requirements: 4.3, 8.1, 8.4_

- [x] 8. Implement MetricsTracker and StructuredLogger
  - Create `metrics.py` with `MetricsTracker` accumulating per-episode metrics: `total_reward`, `classification_accuracy`, `prioritization_accuracy`, `routing_accuracy`, `vip_handling_rate`, `deferral_count`
  - Add structured logging in `env.py` using Python's `logging` module with logger name `"openenv.email_ops"`
  - Emit DEBUG log entries per step with: `step_count`, `action_type`, `reward_breakdown`, `done`
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [ ]* 8.1 Write property test for step log fields (Property 18)
    - **Property 18: Step log entries contain required fields**
    - **Validates: Requirements 14.1**

  - [ ]* 8.2 Write property test for episode metrics completeness (Property 19)
    - **Property 19: Episode metrics completeness**
    - **Validates: Requirements 14.2, 14.3**

- [x] 9. Implement EmailOpsEnv (main environment)
  - Create `env.py` with `EmailOpsEnv` implementing `reset()`, `step()`, `state()`
  - `reset(seed)` clears state, generates inbox via `InboxGenerator`, returns initial `Observation`
  - `step(action)` validates action, advances episode, computes reward via `RewardEngine`, returns 4-tuple
  - Raise `RuntimeError` if `step()` called after `done=True`
  - Raise `ValidationError` for malformed actions (Pydantic handles this automatically)
  - Terminate episode when inbox empty OR `step_count >= max_steps`; set `done=True`
  - Include metrics dict in `info` on final step
  - Accept `log_level` parameter on `__init__`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 3.3, 8.1, 8.2, 8.3, 8.4, 8.5, 14.4_

  - [ ]* 9.1 Write property test for step return structural invariant (Property 1)
    - **Property 1: Step return structural invariant**
    - **Validates: Requirements 1.2, 3.1, 5.11**

  - [ ]* 9.2 Write property test for reset clean state (Property 2)
    - **Property 2: Reset produces clean initial state**
    - **Validates: Requirements 1.1, 8.5**

  - [ ]* 9.3 Write property test for episode termination (Property 14)
    - **Property 14: Episode terminates on inbox empty or max_steps**
    - **Validates: Requirements 8.1, 8.2, 8.3**

  - [ ]* 9.4 Write property test for action history length (Property 15)
    - **Property 15: Action history length equals step count**
    - **Validates: Requirements 7.1**

  - [ ]* 9.5 Write property test for deferral mechanics (Property 9)
    - **Property 9: Deferral moves email to end and applies penalty**
    - **Validates: Requirements 4.3, 5.9**

  - [ ]* 9.6 Write unit tests for environment behavior
    - `test_step_after_done_raises_runtime_error` (Req 1.4)
    - `test_empty_inbox_observation` — current_email=None when inbox empty (Req 3.3)
    - `test_delayed_rewards_in_final_info` (Req 7.5)

- [x] 10. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Create openenv.yaml and TaskConfig wiring
  - Create `openenv.yaml` with: environment name, version, description, author, license
  - Define all three tasks (easy, medium, hard) with task_id, description, difficulty, max_steps, inbox_size, reward_components
  - Include observation schema and action schema sections
  - Wire `TaskConfig` loading via `Parser.parse_yaml()` into `EmailOpsEnv.__init__`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ]* 11.1 Write unit test for openenv.yaml structure
    - `test_openenv_yaml_structure` — verify all required keys present (Req 11.1–11.5)

- [x] 12. Implement inference.py baseline script
  - Create `inference.py` reading `OPENAI_API_KEY` and `MODEL_NAME` from environment variables
  - Exit with non-zero status and descriptive stderr message if `OPENAI_API_KEY` not set
  - Run all three tasks sequentially with a fixed random seed
  - Use OpenAI chat completions API; pass `PrettyPrinter.to_text(obs)` as user message
  - Parse LLM response via `Parser.parse_action()`
  - Print task name, total episode reward, and per-component breakdown after each task
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [ ]* 12.1 Write property test for inference reproducibility (Property 17)
    - **Property 17: Inference script reproducibility**
    - **Validates: Requirements 12.4**

  - [ ]* 12.2 Write unit test for missing API key behavior
    - `test_inference_missing_api_key` — verify exit(1) when OPENAI_API_KEY not set (Req 12.5)

- [x] 13. Create Dockerfile and finalize deployment artifacts
  - Create `Dockerfile`: base image python:3.11-slim, copy source, install from `requirements.txt`, set `OPENAI_API_KEY` and `MODEL_NAME` as ENV/ARG, CMD runs `inference.py`
  - Verify `requirements.txt` has all pinned versions
  - _Requirements: 13.1, 13.2, 13.3, 13.5, 13.6_

- [x] 14. Set up pytest configuration and test infrastructure
  - Create `tests/` directory with `conftest.py` registering Hypothesis profile (`max_examples=100`)
  - Create `pytest.ini` or `pyproject.toml` with `testpaths = ["tests"]` and `property`/`unit` markers
  - Ensure all property tests are tagged `@pytest.mark.property` and unit tests `@pytest.mark.unit`
  - _Requirements: (testing infrastructure for all requirements)_

- [x] 15. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests use `hypothesis` with `@settings(max_examples=100)`
- Each property test must include a comment: `# Feature: openenv-email-ops, Property N: <title>`
- Checkpoints ensure incremental validation before moving to the next phase
