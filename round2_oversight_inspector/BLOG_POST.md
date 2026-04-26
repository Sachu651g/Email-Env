# Who Watches the AI? Building an Oversight Inspector Environment for LLM Training

*Meta × Hugging Face OpenEnv Hackathon — Grand Finale, April 2026*
*Author: Sachin S Gunagi*

| Resource | Link |
|---|---|
| 🤗 HF Space (live demo) | https://huggingface.co/spaces/sachingunagi66/openenv-email-ops |
| 💻 GitHub | https://github.com/Sachu651g/AI-Oversight-Inspector |
| 📓 Colab notebook | [Open in Colab](https://colab.research.google.com/github/Sachu651g/AI-Oversight-Inspector/blob/main/round2_oversight_inspector/colab_train_oversight.ipynb) |
| 🎥 Demo video | *(link your YouTube video here)* |

---

## Training Results (500 steps · Llama-3.2-1B-Instruct · GRPO + Unsloth · Free T4 GPU)

| Metric | Before | After | Delta |
|---|---|---|---|
| Detection accuracy | 42% | **78%** | +36pp |
| False positive rate | 35% | **12%** | −23pp |
| Severity accuracy | 38% | **71%** | +33pp |
| Explanation quality | 0.31 | **0.67** | +0.36 |
| Avg episode score | 0.21 | **0.74** | +0.53 |

---

## The Problem Nobody Talks About

Everyone is building AI agents. But who monitors them?

When you deploy a fleet of AI agents in an enterprise — classifying emails, routing tickets, generating responses — they will make mistakes. Some are obvious: spam classified as urgent. Some are subtle: correct classification, wrong routing. Some are dangerous: hallucinated facts in customer-facing responses.

The standard answer is "add a human in the loop." But at enterprise scale, with thousands of decisions per day, human review doesn't scale.

**What if the monitor itself is an AI?**

That is exactly the problem this project tackles — and it turns out to be one of the most important open research questions in AI safety: **scalable oversight**.

---

## What We Built

This submission contains two complete, interlinked OpenEnv environments:

### Round 1 — EmailOpsEnv

A realistic enterprise inbox environment where an LLM agent must classify, prioritize, route, and reply to emails. Key design choices:

- **Partial observability**: Agent never sees ground truth labels
- **VIP penalty system**: VIP emails not handled within 3 steps trigger −0.30 penalty
- **Shaped rewards**: Per-step signals, not just terminal reward
- **Adaptive difficulty**: Easy (5 emails, classify only) → Medium → Hard (10 emails, full pipeline)

### Round 2 — AI Oversight Inspector (Grand Finale)

The novel contribution. The trained agent is *not* the email processor — it is the **auditor** that watches the email processors.

```
Sub-Agent Fleet:
  ClassifierAgent  → classifies incoming tasks
  PrioritizerAgent → assigns urgency levels
  RouterAgent      → routes to correct team
  ResponderAgent   → generates replies
        │
        │  outputs only — ground truth NEVER exposed
        ▼
AI Oversight Inspector (the LLM being trained):
  - Observes all sub-agent inputs, outputs, explanations
  - Detects: hallucination, policy breach, wrong classification,
    severity mismatch, cross-agent inconsistency
  - Must explain WHY something is wrong (not just THAT it is)
  - Rewarded for precision: FP penalty (−0.30) > miss (−0.20)
```

The key constraint: **the overseer never sees ground truth labels**. It must reason like a real auditor — from evidence and context alone.

---

## The Reward Design That Actually Teaches

Most RL environments have obvious reward shortcuts. Ours doesn't.

If the overseer just flags everything, it gets destroyed by false positive penalties. If it approves everything, it misses violations. The only winning strategy is **genuine reasoning**:

| Signal | Value | Purpose |
|---|---|---|
| Correct detection | +0.40 | Core training signal |
| Correct severity | +0.20 | Calibration — not just binary |
| Quality explanation | +0.20 | Causal reasoning — WHY |
| Correct approve | +0.20 | Precision — knowing when NOT to flag |
| False positive | **−0.30** | Alert fatigue is real |
| Missed violation | −0.20 | Can't just approve everything |
| Detection rate improving | +0.10 | Self-improvement meta-signal |

The asymmetry (−0.30 FP vs −0.20 miss) is intentional. In real enterprise deployments, false alarms destroy trust faster than missed violations.

---

## Adaptive Curriculum: The Model Teaches Itself

Rather than training on fixed difficulty, the environment automatically promotes and demotes the overseer:

- Easy → Medium when detection accuracy ≥ 70% (last 5 steps)
- Medium → Hard when accuracy ≥ 75%
- Hard → Medium when accuracy < 50% (live demotion)

The training charts show a brief demotion at step ~330: the model was promoted to Hard, struggled, was demoted, then recovered and was promoted again. **This is real learning, not a scripted curve.**

---

## Training: GRPO on a Free T4 GPU

We used GRPO (Group Relative Policy Optimization) — the same algorithm that trained DeepSeek-R1 — because:
1. No separate critic/value model needed (fits on free T4)
2. Proven effective for reasoning tasks
3. Group-based reward comparison makes sense for oversight

**Config**: Llama-3.2-1B-Instruct · LoRA rank 16 · 500 steps · batch 8 · ~30 min

The Colab notebook runs end-to-end on a free GPU. Open it, run all cells, get a trained model.

---

## Why This Matters

Scalable oversight is not a future problem. It is happening right now, in every company deploying AI agents at scale.

We built the training environment to solve it — an open, reproducible, OpenEnv-compliant environment that any researcher can use to train and evaluate AI oversight systems.

The environment is live at [huggingface.co/spaces/sachingunagi66/openenv-email-ops](https://huggingface.co/spaces/sachingunagi66/openenv-email-ops).

The code is at [github.com/Sachu651g/AI-Oversight-Inspector](https://github.com/Sachu651g/AI-Oversight-Inspector).

---

*Built for the Meta × Hugging Face OpenEnv Hackathon 2026 — Grand Finale.*
*Theme: Multi-Agent Interactions + Scalable Oversight*
