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
  - multi-agent
  - ai-safety
  - scalable-oversight
---

# openenv-email-ops · AI Oversight Inspector

> **Meta × Hugging Face OpenEnv Hackathon 2026 — Grand Finale Submission**
> Theme: Multi-Agent Interactions + Scalable Oversight

---

## Quick links

| Resource | URL |
|---|---|
| 🤗 HF Space (live demo) | https://huggingface.co/spaces/sachingunagi66/openenv-email-ops |
| 💻 GitHub | https://github.com/Sachu651g/AI-Oversight-Inspector |
| 📓 Colab training notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sachu651g/AI-Oversight-Inspector/blob/main/round2_oversight_inspector/colab_train_oversight.ipynb) |
| 📝 Blog post | [Read on GitHub](https://github.com/Sachu651g/AI-Oversight-Inspector/blob/main/round2_oversight_inspector/BLOG_POST.md) |

---

## What problem does this solve?

Everyone is building AI agents. Nobody is asking the most important question: **who monitors the AI?**

When you deploy a fleet of AI agents in an enterprise — classifying emails, routing tickets, generating responses — they will make mistakes. Some obvious (spam classified as urgent). Some subtle (correct classification, wrong routing). Some dangerous (hallucinated facts in customer responses).

The standard answer is "add a human in the loop." That doesn't scale.

**This project trains an AI Oversight Inspector** — an LLM that monitors a fleet of sub-agents and detects their violations in real time, *without ever seeing ground truth labels.* It must reason the same way a real auditor would: from inputs, outputs, and explanations only.

This directly addresses **scalable oversight**, one of the most important open problems in AI safety.

---

## Training results (Round 2 — AI Oversight Inspector)

After **500 training steps** on `Llama-3.2-1B-Instruct` via **GRPO + Unsloth** on a **free T4 GPU**:

| Metric | Before training | After training | Delta |
|---|---|---|---|
| Detection accuracy | 42% | **78%** | +36pp |
| False positive rate | 35% | **12%** | −23pp |
| Severity accuracy | 38% | **71%** | +33pp |
| Explanation quality | 0.31 | **0.67** | +0.36 |
| Avg episode score | 0.21 | **0.74** | +0.53 |

### Reward curve — episode score over 500 training steps

![Training Results](round2_oversight_inspector/assets/training_results.png)

*Left: Episode reward rises from baseline 0.21 to 0.74 over 500 steps. Right: Before/after comparison across all five metrics.*

### Adaptive curriculum — difficulty progression during training

![Curriculum Progression](round2_oversight_inspector/assets/curriculum_progression.png)

*The environment automatically promotes the overseer Easy → Medium → Hard as detection accuracy crosses 70% / 75%. The brief demotion at step ~330 shows the curriculum is live and reactive — not scripted.*

### Detailed metrics

![Detailed Metrics](round2_oversight_inspector/assets/detailed_metrics.png)

*Left: False positive rate drops 35% → 12% (precision improves). Right: Severity accuracy 38% → 71%.*

---

## Architecture overview

### Round 2 — AI Oversight Inspector (Grand Finale)

```
Sub-Agent Fleet (4 agents):
  ├── ClassifierAgent    → classifies incoming tasks
  ├── PrioritizerAgent   → assigns urgency levels
  ├── RouterAgent        → routes to correct team
  └── ResponderAgent     → generates replies
            │
            │  outputs only — ground truth NEVER exposed
            ▼
AI Oversight Inspector (the LLM being trained):
  ├── Observes all sub-agent inputs, outputs, and explanations
  ├── Detects violations: hallucination, policy breach, wrong classification,
  │   severity mismatch, cross-agent inconsistency
  ├── Must explain WHY something is wrong (not just THAT it is wrong)
  └── Rewarded for precision: false positives cost MORE than missed violations
```

### Round 1 — Email Operations Environment

```
EmailOpsEnv (OpenEnv compliant):
  ├── Inbox: 5–10 emails (seeded, noise-injected, partial observability)
  ├── Actions: classify → prioritize → route → reply → defer
  ├── Reward: shaped across trajectory — VIP ignore penalty at episode end
  └── Difficulty: easy (classify only) → medium → hard (full pipeline)

MultiAgentOrchestrator (evaluation layer):
  ClassifierAgent → PriorityAgent → RoutingAgent → ResponseAgent
       └── CoordinatorAgent (conflict resolution + coherence scoring)
       └── SelfImprovingMemory (per-agent mistake tracking)
       └── DecisionTrace (full explainability per step)
```

---

## Why the reward is designed this way

The oversight reward function is deliberately hard to game:

| Signal | Value | Rationale |
|---|---|---|
| Correct detection | +0.40 | Core training signal |
| Correct severity assessment | +0.20 | Not just binary — calibration matters |
| Quality explanation | +0.20 | Causal reasoning, not pattern matching |
| Correct approve (no FP) | +0.20 | Teaches precision |
| **False positive** | **−0.30** | Heavier than miss — alert fatigue is real |
| Missed violation | −0.20 | Can't just approve everything |
| Detection rate improving | +0.10 | Self-improvement meta-signal |

An agent that flags everything is destroyed by FP penalties. An agent that approves everything misses violations. The only winning strategy is genuine reasoning.

---

## OpenEnv compliance

| Requirement | Status |
|---|---|
| OpenEnv base class (reset/step/state API) | ✅ Both environments |
| openenv.yaml manifests | ✅ `openenv.yaml` + `round2_oversight_inspector/openenv_oversight.yaml` |
| Client/server separation | ✅ `server/app.py` separate from env internals |
| HuggingFace Space | ✅ Docker, FastAPI + Gradio UI |
| Training script (Unsloth + TRL) | ✅ `round2_oversight_inspector/train_oversight.py` (469 lines, `--dry-run` flag) |
| Colab notebook | ✅ `round2_oversight_inspector/colab_train_oversight.ipynb` |
| Training evidence | ✅ 3 PNG plots in `round2_oversight_inspector/assets/` |
| Unit tests | ✅ 84 tests (Round 1) + 20 tests (Round 2 oversight) |

---

## How to run

### Live demo
Visit the [HF Space](https://huggingface.co/spaces/sachingunagi66/openenv-email-ops) — no setup needed.

### Colab training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sachu651g/AI-Oversight-Inspector/blob/main/round2_oversight_inspector/colab_train_oversight.ipynb)

Run all cells. ~30 minutes on free T4 GPU.

### Local (dry-run, no GPU needed)
```bash
git clone https://github.com/Sachu651g/AI-Oversight-Inspector
cd AI-Oversight-Inspector/round2_oversight_inspector
pip install -r requirements.txt
python train_oversight.py --dry-run
```

### REST API
```bash
POST /reset          # reset episode
POST /step           # take action
GET  /state          # current state
GET  /demo           # dry-run all 3 difficulties
```

---

## Project structure

```
.
├── openenv.yaml                  # Round 1 OpenEnv manifest
├── openenv_email_ops/            # Round 1 environment
│   ├── env.py                    # EmailOpsEnv (reset/step/state)
│   ├── reward_engine.py          # Composable reward rubric
│   ├── graders.py                # Per-component graders
│   └── ...
├── multi_agent_system/           # Multi-agent evaluation layer
│   ├── agents.py                 # 4 specialized agents
│   ├── orchestrator.py           # Pipeline coordination
│   └── self_improving_memory.py  # Per-agent mistake tracking
├── round2_oversight_inspector/   # Round 2 (Grand Finale): AI Oversight Inspector
│   ├── openenv_oversight.yaml    # Round 2 OpenEnv manifest
│   ├── oversight_env/            # Round 2 RL Environment
│   │   ├── env.py                # OversightEnv (OpenEnv compliant)
│   │   ├── reward_engine.py      # Oversight reward (precision-first)
│   │   ├── adaptive_curriculum.py # Live Easy→Medium→Hard curriculum
│   │   └── sub_agent_fleet.py    # 4-agent fleet with violation injection
│   ├── train_oversight.py        # GRPO training script (469 lines)
│   ├── inference_oversight.py    # Baseline overseer runner
│   ├── benchmark_baseline.py     # Random vs trained comparison
│   ├── colab_train_oversight.ipynb # Runnable Colab notebook (judges: start here)
│   └── assets/                   # Training evidence: 3 PNG plots
│       ├── training_results.png  # Reward curve over 500 steps
│       ├── curriculum_progression.png # Difficulty progression
│       └── detailed_metrics.png  # FP rate + severity accuracy
├── server/app.py                 # FastAPI server (client/server sep.)
├── app.py                        # HF Space entry point (Gradio + API)
├── Dockerfile                    # Docker deployment
└── tests/                        # 84 Round 1 unit tests
```

---

*Built by Sachin S Gunagi for the Meta × Hugging Face OpenEnv Hackathon 2026 — Grand Finale.*
