---
title: openenv-email-ops
emoji: 📧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# openenv-email-ops

A production-grade [OpenEnv](https://openenv.dev) reinforcement learning environment that simulates enterprise inbox management. AI agents learn to classify, prioritize, route, and reply to emails across multi-step episodes with memory-based reward shaping.

---

## Problem Motivation

Enterprise email inboxes are a high-stakes, high-volume decision environment. A typical knowledge worker processes hundreds of emails per day, making rapid triage decisions that affect customer satisfaction, SLA compliance, and team efficiency. Mistakes — ignoring a VIP escalation, misrouting a support request, or endlessly deferring urgent items — have real business consequences.

This environment models that challenge as a reinforcement learning problem. An agent must learn not just to classify emails correctly, but to do so efficiently, handle VIP senders with priority, avoid infinite deferral loops, and generate contextually appropriate replies — all within a bounded episode. The reward function is designed to reflect real-world incentives: accuracy matters, but so does speed, consistency, and long-term consequence awareness.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent (LLM)                          │
│                  inference.py / custom agent                │
└────────────────────────┬────────────────────────────────────┘
                         │  Action
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     EmailOpsEnv                             │
│                       env.py                                │
│                                                             │
│  ┌─────────────────┐   ┌──────────────────┐                │
│  │  InboxGenerator │   │  EpisodeManager  │                │
│  │  inbox_generator│   │  episode_manager │                │
│  │  .py            │──▶│  .py             │                │
│  └─────────────────┘   └──────────────────┘                │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  RewardEngine                        │   │
│  │                  reward_engine.py                    │   │
│  │                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │Classification│  │Prioritization│                 │   │
│  │  │   Grader     │  │   Grader     │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  │  ┌──────────────┐  ┌──────────────┐                 │   │
│  │  │   Routing    │  │    Reply     │                 │   │
│  │  │   Grader     │  │   Grader     │                 │   │
│  │  └──────────────┘  └──────────────┘                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────┐   ┌──────────────────┐                │
│  │  MemoryTracker  │   │  MetricsTracker  │                │
│  │  memory_tracker │   │  metrics.py      │                │
│  │  .py            │   └──────────────────┘                │
│  └─────────────────┘                                       │
└────────────────────────┬────────────────────────────────────┘
                         │  (Observation, Reward, done, info)
                         ▼
                      Agent

Supporting modules:
  models.py        — Pydantic models: Email, Action, Observation, Reward
  pretty_printer.py — Serializes observations to human-readable text / JSON
  parser.py        — Parses LLM output into Action models
  openenv.yaml     — Machine-readable environment and task metadata
```

---

## Action Space

The agent submits one action per step. All actions are validated by Pydantic before processing.

| Action Type | Value | Description |
|---|---|---|
| `classify_email` | `"spam"` \| `"important"` \| `"promotion"` | Classify the current email into one of three categories |
| `prioritize_email` | `"low"` \| `"medium"` \| `"high"` | Assign a priority level to the current email |
| `route_email` | `"support"` \| `"sales"` \| `"escalation"` | Route the email to the appropriate team |
| `generate_reply` | free text (min 20 chars) | Generate a reply to the current email |
| `defer_email` | _(no value)_ | Move the current email to the end of the inbox for later processing |

---

## Observation Space

Each step returns an `Observation` with four fields:

| Field | Type | Description |
|---|---|---|
| `current_email` | `Email \| None` | The email to act on; `None` when inbox is empty |
| `inbox_summary` | `InboxSummary` | Aggregate stats: counts by sender type and urgency distribution |
| `action_history` | `list[Action]` | All actions taken so far in the current episode |
| `step_count` | `int` | Number of steps taken in the current episode |

Ground truth labels (correct classification, priority, route) are attached to each email internally but are **never exposed** in the observation — the agent must infer the correct action from email content alone.

---

## Reward Design

The reward function is designed to reflect real-world email operations incentives:

| Component | Value | Reasoning |
|---|---|---|
| Correct classification | +0.2 | Core task accuracy; the most fundamental signal |
| Correct prioritization | +0.2 | Ensures urgent items are surfaced appropriately |
| Correct routing | +0.2 | Prevents emails from landing in the wrong team's queue |
| Reply quality | up to +0.2 | Proportional to heuristic quality score (length, greeting, relevance) |
| Efficiency bonus | +0.1 | Rewards fast, decisive action within minimum required steps |
| Incorrect classification | -0.2 | Symmetric penalty to classification reward; wrong answers have cost |
| Deferral penalty | -0.05/step | Small per-step cost to discourage unnecessary deferral |
| VIP ignore penalty | -0.3 | Episode-level penalty for ignoring high-value senders; reflects real SLA risk |
| Excessive deferral penalty | -0.5 | Prevents infinite deferral loops on the same email (triggered after 2 deferrals) |
| VIP consistency bonus | +0.3 | Episode-level bonus for handling all VIP emails correctly; rewards long-term planning |

Delayed rewards (VIP penalties/bonuses, early classification bonuses) are computed at episode end and included in the final `info` dict under `metrics.delayed_rewards`.

---

## Task Difficulty Levels

| Task | Inbox Size | Max Steps | Evaluated Components |
|---|---|---|---|
| `easy` | 5 | 30 | classification only |
| `medium` | 8 | 50 | classification, prioritization, routing |
| `hard` | 10 | 80 | classification, prioritization, routing, reply |

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"   # optional, defaults to gpt-4o-mini
```

---

## Running Inference

### Local

```bash
python inference.py
```

### Docker

```bash
# Build
docker build -t openenv-email-ops .

# Run (inject API key at runtime)
docker run --rm -e OPENAI_API_KEY="sk-..." openenv-email-ops
```

---

## Example Output

```
=== Task: easy ===
Episode complete.
Total reward: 0.80
Score breakdown:
  classification: 0.80
  deferral_penalty: -0.05

=== Task: medium ===
Episode complete.
Total reward: 1.45
Score breakdown:
  classification: 0.60
  prioritization: 0.40
  routing: 0.60
  deferral_penalty: -0.10
  vip_consistency_bonus: 0.30

=== Task: hard ===
Episode complete.
Total reward: 2.10
Score breakdown:
  classification: 0.60
  prioritization: 0.40
  routing: 0.40
  reply: 0.60
  efficiency_bonus: 0.10
  vip_consistency_bonus: 0.30
  deferral_penalty: -0.05
```

---

## Project Structure

```
openenv_email_ops/
├── env.py              # EmailOpsEnv: main OpenEnv interface
├── models.py           # Pydantic models
├── inbox_generator.py  # Seeded email stream generation
├── graders.py          # Per-component scoring graders
├── reward_engine.py    # Reward aggregation and bonus/penalty logic
├── memory_tracker.py   # Per-email decision history tracking
├── episode_manager.py  # Inbox state and step management
├── pretty_printer.py   # Human-readable serialization
├── parser.py           # LLM output parsing
└── metrics.py          # Per-episode metrics accumulation

inference.py            # Baseline OpenAI-backed agent
openenv.yaml            # Machine-readable environment metadata
Dockerfile
requirements.txt
```

---

## License

MIT
