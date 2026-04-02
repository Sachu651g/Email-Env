# Requirements Document

## Introduction

The Advanced AI Email Operations System (openenv-email-ops) is a production-grade OpenEnv reinforcement learning environment that simulates enterprise inbox management. It enables AI agents to learn email triage, classification, prioritization, routing, and reply generation across multi-step episodes with memory-based reward shaping. The environment is fully OpenEnv-compliant, Dockerized, and deployable to Hugging Face Spaces.

## Glossary

- **Environment**: The OpenEnv-compliant simulation engine exposing `step()`, `reset()`, and `state()` interfaces
- **Episode**: A single work session consisting of sequential email processing steps until the inbox is cleared or max steps are reached
- **Email**: A simulated message with subject, body, sender_type, urgency_score, and hidden ground truth labels
- **Sender_Type**: One of: customer, spammer, VIP, internal
- **Agent**: The AI system interacting with the Environment via actions
- **Action**: One of: classify_email, prioritize_email, route_email, generate_reply, defer_email
- **Observation**: The structured view of current state exposed to the Agent, including current email, inbox summary, action history, and step count
- **Grader**: A deterministic scoring component that evaluates Agent actions and returns a float in [0.0, 1.0]
- **Task**: A difficulty-scoped challenge (EASY, MEDIUM, HARD) defining which actions are evaluated
- **Reward**: A scalar signal computed from Grader outputs, bonuses, and penalties
- **Memory**: The Agent's history of previous actions within an episode, used for reward shaping
- **Ground_Truth**: Hidden labels attached to each Email defining correct classification, priority, and routing
- **Inbox**: The ordered collection of Emails available during an Episode
- **Pretty_Printer**: A component that serializes Pydantic models to human-readable or structured text
- **Parser**: A component that deserializes structured input into Pydantic models

---

## Requirements

### Requirement 1: OpenEnv Interface Compliance

**User Story:** As an AI researcher, I want the environment to expose a standard OpenEnv interface, so that I can plug it into any OpenEnv-compatible training framework without custom integration code.

#### Acceptance Criteria

1. THE Environment SHALL expose a `reset()` method that initializes a new Episode and returns an initial Observation
2. THE Environment SHALL expose a `step(action)` method that accepts an Action, advances the Episode by one step, and returns a tuple of (Observation, Reward, done, info)
3. THE Environment SHALL expose a `state()` method that returns the current internal state as a serializable dict
4. WHEN `step()` is called after `done` is True, THE Environment SHALL raise a RuntimeError indicating the Episode has ended
5. THE Environment SHALL use Pydantic models for Observation, Action, and Reward to enforce schema validation
6. WHEN an Action fails Pydantic validation, THE Environment SHALL raise a ValidationError with a descriptive message

---

### Requirement 2: Email Stream Generation

**User Story:** As an AI researcher, I want a realistic dynamic email stream, so that the Agent faces varied, representative inputs across episodes.

#### Acceptance Criteria

1. THE Environment SHALL generate Emails with the following fields: subject (str), body (str), sender_type (one of: customer, spammer, VIP, internal), urgency_score (float in [0.0, 1.0]), and hidden Ground_Truth labels
2. WHEN an Episode is initialized with a fixed random seed, THE Environment SHALL produce the same Email sequence deterministically across runs
3. THE Environment SHALL inject realistic noise into Email text (e.g., typos, informal language, ambiguous phrasing) to simulate partial observability
4. THE Environment SHALL generate Emails such that sender_type distribution across an Episode includes at least one instance of each type: customer, spammer, VIP, internal
5. THE Ground_Truth labels SHALL include: correct_classification (spam/important/promotion), correct_priority (low/medium/high), correct_route (support/sales/escalation)

---

### Requirement 3: Observation Space

**User Story:** As an AI researcher, I want a structured observation space, so that the Agent receives all contextually relevant information needed to make decisions.

#### Acceptance Criteria

1. THE Observation SHALL contain: current_email (Email), inbox_summary (dict with counts by sender_type and urgency distribution), action_history (list of previous Actions taken in the Episode), step_count (int)
2. THE Observation SHALL NOT expose Ground_Truth labels to the Agent
3. WHEN the Inbox is empty, THE Observation SHALL set current_email to None and inbox_summary to empty counts
4. THE Environment SHALL serialize Observations using the Observation Pydantic model with no loss of information

---

### Requirement 4: Action Space

**User Story:** As an AI researcher, I want a well-defined action space, so that the Agent can take meaningful, structured decisions on each email.

#### Acceptance Criteria

1. THE Action model SHALL support the following action types: classify_email (value: spam/important/promotion), prioritize_email (value: low/medium/high), route_email (value: support/sales/escalation), generate_reply (value: free-text string), defer_email (no value required)
2. WHEN an Agent submits a classify_email Action, THE Environment SHALL record the classification against the current Email's Ground_Truth
3. WHEN an Agent submits a defer_email Action, THE Environment SHALL move the current Email to the end of the Inbox and apply a per-step deferral penalty to the Reward
4. WHEN an Agent submits a generate_reply Action, THE Environment SHALL pass the reply text to the Grader for heuristic quality scoring
5. THE Action model SHALL validate that action_type is one of the five defined types and raise a ValidationError for unknown types

---

### Requirement 5: Reward Function

**User Story:** As an AI researcher, I want a well-structured reward function with bonuses and penalties, so that the Agent learns to balance accuracy, efficiency, and long-term consistency.

#### Acceptance Criteria

1. WHEN an Agent correctly classifies an Email, THE Grader SHALL add +0.2 to the step Reward
2. WHEN an Agent correctly prioritizes an Email, THE Grader SHALL add +0.2 to the step Reward
3. WHEN an Agent correctly routes an Email, THE Grader SHALL add +0.2 to the step Reward
4. WHEN an Agent generates a reply that passes heuristic quality checks, THE Grader SHALL add up to +0.2 to the step Reward proportional to quality score
5. WHEN an Agent makes a decision within the minimum required steps, THE Grader SHALL add +0.1 efficiency bonus to the step Reward
6. WHEN an Agent incorrectly classifies an Email, THE Grader SHALL subtract 0.2 from the step Reward
7. WHEN an Agent ignores an Email with sender_type VIP or urgency_score >= 0.8, THE Grader SHALL subtract 0.3 from the Episode Reward
8. WHEN an Agent defers the same Email more than 2 times in an Episode, THE Grader SHALL subtract 0.5 from the Episode Reward as an infinite-loop penalty
9. WHEN an Agent defers an Email, THE Grader SHALL subtract a small per-step delay penalty (0.05) from the step Reward
10. WHEN an Agent correctly handles all VIP Emails in an Episode, THE Grader SHALL add a long-term consistency bonus of +0.3 to the Episode Reward
11. THE Reward model SHALL expose: step_reward (float), episode_reward (float), breakdown (dict mapping reward component names to float values)

---

### Requirement 6: Task Definitions and Difficulty Levels

**User Story:** As an AI researcher, I want three difficulty levels, so that I can progressively train and evaluate agents from simple classification to full pipeline execution.

#### Acceptance Criteria

1. THE EASY Task SHALL evaluate only classify_email actions; THE Grader SHALL score only classification correctness
2. THE MEDIUM Task SHALL evaluate classify_email, prioritize_email, and route_email actions; THE Grader SHALL score all three
3. THE HARD Task SHALL evaluate classify_email, prioritize_email, route_email, and generate_reply actions; THE Grader SHALL score all four components
4. WHEN a Task is initialized, THE Environment SHALL configure the Grader to score only the components relevant to that Task's difficulty level
5. THE openenv.yaml file SHALL define all three Tasks with their difficulty levels, observation schema, action schema, and reward schema

---

### Requirement 7: Memory-Based Reward Shaping

**User Story:** As an AI researcher, I want memory-based reward shaping, so that the Agent learns that past decisions have long-term consequences within an episode.

#### Acceptance Criteria

1. THE Environment SHALL maintain an action_history in the Observation for the full duration of the Episode
2. WHEN an Agent ignores a VIP Email (does not classify it as important within 3 steps of receiving it), THE Environment SHALL apply a -0.3 penalty to a future step Reward within the same Episode
3. WHEN an Agent correctly classifies an Email early (within the first step of receiving it), THE Environment SHALL apply a +0.1 early-classification bonus to the next step Reward
4. THE Environment SHALL track per-email decision timestamps (step indices) to enable delayed reward computation
5. WHEN an Episode ends, THE Environment SHALL compute and include all deferred/delayed reward components in the final info dict

---

### Requirement 8: Multi-Step Episode Management

**User Story:** As an AI researcher, I want multi-step episodes with clear termination conditions, so that the Agent learns to manage a full inbox session efficiently.

#### Acceptance Criteria

1. THE Environment SHALL terminate an Episode when the Inbox is empty (all Emails processed)
2. THE Environment SHALL terminate an Episode when step_count reaches the configured max_steps limit
3. WHEN an Episode terminates, THE Environment SHALL set done=True in the return value of `step()`
4. THE Environment SHALL support configurable Episode parameters: inbox_size (int), max_steps (int), random_seed (int)
5. WHEN `reset()` is called, THE Environment SHALL clear action_history, reset step_count to 0, and generate a new Inbox

---

### Requirement 9: Grader Architecture

**User Story:** As an AI researcher, I want separate, deterministic graders per task, so that scoring is transparent, reproducible, and independently testable.

#### Acceptance Criteria

1. THE Environment SHALL provide a ClassificationGrader that scores classify_email actions against Ground_Truth labels and returns a float in [0.0, 1.0]
2. THE Environment SHALL provide a PrioritizationGrader that scores prioritize_email actions against Ground_Truth labels and returns a float in [0.0, 1.0]
3. THE Environment SHALL provide a RoutingGrader that scores route_email actions against Ground_Truth labels and returns a float in [0.0, 1.0]
4. THE Environment SHALL provide a ReplyGrader that scores generate_reply actions using rule-based and heuristic checks and returns a float in [0.0, 1.0]
5. WHEN given the same inputs, EACH Grader SHALL return the same score deterministically across runs
6. THE ReplyGrader SHALL evaluate replies on: minimum length (>= 20 characters), presence of greeting, relevance to email subject keywords, and absence of placeholder text

---

### Requirement 10: Pydantic Models and Serialization

**User Story:** As an AI researcher, I want all data structures defined as Pydantic models, so that schema validation, serialization, and API compatibility are guaranteed.

#### Acceptance Criteria

1. THE models.py module SHALL define Pydantic models for: Email, Action, Observation, Reward, EpisodeInfo, and TaskConfig
2. WHEN a Pydantic model is serialized to JSON, THE Pretty_Printer SHALL produce valid JSON that can be deserialized back to an equivalent model instance (round-trip property)
3. FOR ALL valid model instances, parsing then printing then parsing SHALL produce an equivalent object
4. WHEN a model field receives an out-of-range value (e.g., urgency_score > 1.0), THE model SHALL raise a ValidationError with a field-specific message
5. THE Action model SHALL use a discriminated union or Literal types to enforce valid action_type values

---

### Requirement 11: openenv.yaml Configuration

**User Story:** As an AI researcher, I want a machine-readable openenv.yaml, so that the environment metadata, tasks, and schemas are discoverable without reading source code.

#### Acceptance Criteria

1. THE openenv.yaml SHALL include: environment name, version, description, author, and license fields
2. THE openenv.yaml SHALL define all three Tasks (easy, medium, hard) with: task_id, description, difficulty, max_steps, inbox_size, and reward_components
3. THE openenv.yaml SHALL include the observation schema listing all Observation fields and their types
4. THE openenv.yaml SHALL include the action schema listing all valid action_type values and their expected value formats
5. WHEN the openenv.yaml is parsed by a YAML parser, THE Parser SHALL produce a valid Python dict with no missing required keys

---

### Requirement 12: Baseline Inference Script

**User Story:** As an AI researcher, I want a baseline inference script, so that I can immediately run the environment with an OpenAI-backed agent and get reproducible benchmark scores.

#### Acceptance Criteria

1. THE inference.py script SHALL read OPENAI_API_KEY and MODEL_NAME from environment variables
2. THE inference.py script SHALL run all three Tasks (easy, medium, hard) sequentially in a single execution
3. WHEN a Task completes, THE inference.py script SHALL print the Task name, total episode reward, and per-component score breakdown
4. THE inference.py script SHALL use a fixed random seed to produce reproducible scores across runs
5. IF OPENAI_API_KEY is not set, THEN THE inference.py script SHALL exit with a non-zero status code and print a descriptive error message
6. THE inference.py script SHALL use the OpenAI chat completions API to generate Agent actions from Observation text

---

### Requirement 13: Docker and Deployment

**User Story:** As a platform engineer, I want a fully working Dockerfile and Hugging Face Spaces compatibility, so that the environment can be deployed and run without local setup.

#### Acceptance Criteria

1. THE Dockerfile SHALL produce an image that passes `docker build` without errors
2. WHEN the Docker container is run with `docker run`, THE container SHALL execute inference.py and exit cleanly
3. THE Dockerfile SHALL install all dependencies from requirements.txt during the build stage
4. THE Environment SHALL start up in under 10 seconds from container launch to first episode step
5. THE requirements.txt SHALL pin all dependency versions to ensure reproducible builds
6. THE Dockerfile SHALL set OPENAI_API_KEY and MODEL_NAME as ARG or ENV instructions to support runtime injection

---

### Requirement 14: Logging and Metrics Tracking

**User Story:** As an AI researcher, I want structured logging and metrics tracking, so that I can debug agent behavior and analyze performance across episodes.

#### Acceptance Criteria

1. THE Environment SHALL emit structured log entries at DEBUG level for each step, including: step_count, action_type, reward_breakdown, and done status
2. THE Environment SHALL track per-episode metrics: total_reward, classification_accuracy, prioritization_accuracy, routing_accuracy, vip_handling_rate, and deferral_count
3. WHEN an Episode ends, THE Environment SHALL include the metrics dict in the info return value of the final `step()` call
4. THE logging system SHALL be configurable via a log_level parameter on Environment initialization
5. THE Environment SHALL use Python's standard logging module with a named logger ("openenv.email_ops")
