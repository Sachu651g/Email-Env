---
title: openenv-email-ops
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - email
  - rl-environment
---

# openenv-email-ops

An OpenEnv-compatible RL environment for enterprise email triage. Agents learn to classify, prioritize, route, and reply to emails — with memory-based reward shaping and partial observability.

Built for the OpenEnv × Scaler hackathon.

---

## Why email?

Email triage is something every company does, and it's genuinely hard to automate well. A spam filter isn't enough — you need to know *who* sent it (VIP vs. random customer), *how urgent* it is, *where* it should go (support vs. sales vs. escalation), and sometimes *what to say back*. Miss a VIP email and there are real consequences.

This environment captures that complexity. The agent sees realistic email content with noise injected (typos, informal phrasing, ambiguous subjects) and has to make decisions without seeing the ground truth labels. Past decisions affect future rewards — ignore a VIP email early and you'll pay for it later in the episode.

---

## Quick start

```bash
pip install -r requirements.txt

# dry run (no API key needed)
python inference.py --dry-run

# live run
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

Docker:
```bash
docker build -t openenv-email-ops .
docker run --rm -e OPENAI_API_KEY="sk-..." openenv-email-ops
```

---

## Environment interface

```python
from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action, TaskConfig

task = TaskConfig(
    task_id="hard", description="Full pipeline", difficulty="hard",
    max_steps=80, inbox_size=10,
    reward_components=["classification", "prioritization", "routing", "reply"]
)
env = EmailOpsEnv(task_config=task, seed=42)

obs = env.reset()
obs, reward, done, info = env.step(Action(action_type="classify_email", value="important"))
state = env.state()
```

---

## Action space

| Action | Values | Notes |
|---|---|---|
| `classify_email` | `spam` / `important` / `promotion` | Core triage decision |
| `prioritize_email` | `low` / `medium` / `high` | Urgency assignment |
| `route_email` | `support` / `sales` / `escalation` | Team routing |
| `generate_reply` | free text (≥20 chars) | Scored on length, greeting, relevance |
| `defer_email` | — | Moves email to end of inbox; costs -0.05/step |

---

## Observation space

```python
{
  "current_email": {
    "id": "uuid",
    "subject": "...",
    "body": "...",
    "sender_type": "customer | spammer | VIP | internal",
    "urgency_score": 0.0–1.0
    # ground_truth is NEVER exposed to the agent
  },
  "inbox_summary": {
    "counts_by_sender_type": {"customer": 2, "VIP": 1, ...},
    "urgency_distribution": {"low": 1, "medium": 2, "high": 1}
  },
  "action_history": [...],
  "step_count": 3
}
```

---

## Reward function

| Signal | Value | When |
|---|---|---|
| Correct classification | +0.2 | action matches ground truth |
| Correct prioritization | +0.2 | action matches ground truth |
| Correct routing | +0.2 | action matches ground truth |
| Reply quality | 0–0.2 | heuristic: length + greeting + keyword match |
| Efficiency bonus | +0.1 | decision made within first step |
| Wrong classification | -0.2 | misclassified email |
| Deferral penalty | -0.05 | per defer action |
| VIP ignore penalty | -0.3 | VIP not classified as important within 3 steps |
| Excessive deferral | -0.5 | same email deferred 3+ times |
| VIP consistency bonus | +0.3 | all VIP emails handled correctly in episode |

The reward is shaped across the full trajectory — not just at the end. This forces the agent to learn long-term consequences, not just greedy per-step decisions.

---

## Tasks

| Task | Inbox | Max steps | What's graded |
|---|---|---|---|
| `easy` | 5 emails | 30 | classification only |
| `medium` | 8 emails | 50 | classify + prioritize + route |
| `hard` | 10 emails | 80 | full pipeline including reply |

---

## Baseline scores (dry-run, seed=42)

```
easy:   total_reward=0.70  (classification: 0.40, efficiency_bonus: 0.20)
medium: total_reward=0.60  (prioritization: 0.20, routing: 0.20, efficiency_bonus: 0.20)
hard:   total_reward=1.10  (routing: 0.40, reply: 0.30, prioritization: 0.20, efficiency_bonus: 0.20)
```

Structured log format (one JSON per line):
```
{"event": "START", "task_id": "easy", ...}
{"event": "STEP", "task_id": "easy", "step": 0, "action_type": "classify_email", ...}
{"event": "END", "task_id": "easy", "total_reward": 0.7, ...}
```

---

## Novel features

- **Memory-based reward shaping** — ignoring a VIP email at step 2 triggers a penalty at episode end
- **Partial observability** — ground truth labels are hidden; agent must infer from noisy email content
- **Realistic noise injection** — typos, informal phrasing, ambiguous subject prefixes
- **Defer action with loop detection** — deferring the same email 3+ times triggers a -0.5 penalty
- **Configurable difficulty** — swap task configs to change inbox size, max steps, and graded components

---

## Project layout

```
openenv_email_ops/
  env.py              # EmailOpsEnv — step/reset/state
  models.py           # Pydantic models (Email, Action, Observation, Reward)
  inbox_generator.py  # seeded email generation with noise
  graders.py          # ClassificationGrader, PrioritizationGrader, RoutingGrader, ReplyGrader
  reward_engine.py    # shaped reward + delayed bonuses/penalties
  memory_tracker.py   # per-email action history
  episode_manager.py  # inbox queue + step count
  metrics.py          # per-episode accuracy tracking
  pretty_printer.py   # observation → LLM prompt text
  parser.py           # LLM output → Action

inference.py          # baseline agent (OpenAI API)
openenv.yaml          # task definitions + schemas
Dockerfile
requirements.txt
tests/                # 84 tests (property-based + unit)
```

---

## HF Space API

The deployed space exposes:

- `GET /` — landing page
- `POST /reset` — reset episode, returns initial observation
- `POST /step?action_type=...&value=...` — take action
- `GET /state` — current env state
- `GET /demo` — dry-run output for all 3 tasks
- `GET /docs` — Swagger UI

Space URL: https://huggingface.co/spaces/sachingunagi66/openenv-email-ops
