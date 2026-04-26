# 3-Minute Pitch Script — AI Oversight Inspector
## Meta × Hugging Face OpenEnv Hackathon Grand Finale

---

## MINUTE 1: The Problem (0:00 – 1:00)

**Say this:**

"Everyone is building AI agents. But nobody is asking the most important question —
**who monitors the AI?**

When you deploy a fleet of AI agents in an enterprise — classifying emails,
routing tickets, generating responses — they WILL make mistakes.

Some mistakes are obvious: spam classified as urgent.
Some are subtle: correct classification, wrong routing.
Some are dangerous: hallucinated facts in customer responses.

The standard answer is 'add a human in the loop.' But that doesn't scale.

**What if the monitor itself is an AI?**

That's what I built."

**[Show terminal running: python round2/inference_oversight.py --dry-run]**

---

## MINUTE 2: The Solution (1:00 – 2:00)

**Say this:**

"This is the **AI Oversight Inspector** — an OpenEnv environment that trains
an LLM to act as an AI auditor.

Here's how it works:

A fleet of 4 sub-agents processes enterprise tasks — classifier, prioritizer,
router, and responder. These agents deliberately inject violations at controlled
rates — hallucinations, policy violations, wrong classifications.

The Oversight Inspector observes all sub-agent outputs and must:
- Detect which agent made a mistake
- Identify the violation type
- Assess the severity
- Explain WHY it's wrong

**Critically — it never sees ground truth labels.** It must reason from the
agent's inputs, outputs, and explanations. Just like a real auditor.

The reward structure penalizes false positives more than missed violations —
teaching the model precision over recall."

**[Point to the [STEP] output showing rewards]**

---

## MINUTE 3: The Innovation (2:00 – 3:00)

**Say this:**

"What makes this stand out is the **adaptive curriculum**.

The environment doesn't stay at fixed difficulty. It starts easy — 20% violation
rate, obvious errors. As the overseer's detection accuracy crosses 70%, it
automatically escalates to medium difficulty. Cross 75% and it goes to hard —
60% violation rate with adversarial hallucinations.

If the overseer degrades, it gets demoted back. This drives **recursive skill
amplification** — the agent must keep improving to maintain its level.

Training uses GRPO via Unsloth on Llama-3.2-1B — fits on a T4 GPU.
After 500 steps, detection accuracy improves from 42% to 78%.
False positive rate drops from 35% to 12%.

This environment teaches theory-of-mind reasoning, causal reasoning,
and calibrated confidence — exactly the capabilities needed for
**scalable AI oversight**, one of the most important open problems in AI safety.

The code is at github.com/Sachu651g/AI-Oversight-Inspector in the round2 folder.
Thank you."

---

## Q&A Preparation (2 minutes)

**Q: Why is this better than just using a rule-based checker?**
A: "Rule-based checkers can only catch known violations. This trains the model
to generalize — it learns the *principles* behind what makes a decision wrong,
not just a fixed list of rules. It can catch novel violations it's never seen."

**Q: How do you prevent the overseer from just flagging everything?**
A: "The reward structure. False positives cost -0.3, which is higher than the
-0.2 for missed violations. The model learns that crying wolf is worse than
missing an occasional error. This teaches calibrated confidence."

**Q: What's the real-world application?**
A: "Any enterprise deploying AI agents needs this. Customer support, legal
document review, medical triage — anywhere AI makes decisions that affect
people, you need an oversight layer. This environment trains that layer."

**Q: How does it handle the hard task?**
A: "Hard task has 60% violation rate with adversarial hallucinations — agents
produce confident-sounding wrong answers. The overseer must detect subtle
inconsistencies between the input and the agent's claims. Even frontier models
struggle with this."

---

## Demo Commands (have these ready)

```bash
# Show environment running
python round2/inference_oversight.py --dry-run

# Show test suite passing
python -m pytest round2/tests/ -v

# Show training script
python round2/train_oversight.py --dry-run
```

---

## Key Numbers to Remember

- 4 tasks: easy (20% violations), medium (40%), hard (60%), adaptive
- Reward: +0.4 detection, -0.3 false positive
- Training: 42% → 78% detection accuracy after 500 steps
- 20 tests passing
- Adaptive curriculum: promotes at 70%/75% accuracy, demotes at 50%
