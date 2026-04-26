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

**Meta × Hugging Face OpenEnv Hackathon 2026 — Grand Finale Submission**

> This is the Hugging Face Space deployment of the openenv-email-ops project.
> For full documentation, training results, and project details, see the [main README](../README.md) or [GitHub](https://github.com/Sachu651g/AI-Oversight-Inspector).

---

## What this Space does

This Space exposes a live REST API for the **AI Oversight Inspector** RL environment.
An LLM is trained here to monitor a fleet of AI sub-agents and detect violations — without ever seeing ground truth labels.

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Landing page |
| `/reset` | POST | Reset episode |
| `/step` | POST | Take action: `?action_type=...&value=...` |
| `/state` | GET | Current state |
| `/demo` | GET | Dry-run output for all 3 tasks |
| `/docs` | GET | Swagger UI |

---

## Quick start

```python
from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action, TaskConfig

task = TaskConfig(task_id="hard", description="Full pipeline", difficulty="hard",
                  max_steps=80, inbox_size=10,
                  reward_components=["classification","prioritization","routing","reply"])
env = EmailOpsEnv(task_config=task, seed=42)
obs = env.reset()
obs, reward, done, info = env.step(Action(action_type="classify_email", value="important"))
```

```bash
pip install -r requirements.txt
python inference.py --dry-run
```

---

## Links

| Resource | URL |
|---|---|
| GitHub | https://github.com/Sachu651g/AI-Oversight-Inspector |
| Colab training notebook | `round2/colab_train_oversight.ipynb` |
| Blog post | `round2/BLOG_POST.md` |

*Built by Sachin S Gunagi for the Meta × Hugging Face OpenEnv Hackathon 2026.*
