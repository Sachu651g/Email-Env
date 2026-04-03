"""
OpenEnv Email Ops — HF Space API Server
Exposes reset(), step(), state() as HTTP endpoints + Gradio UI
"""
from __future__ import annotations

import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import gradio as gr

from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action, TaskConfig
from openenv_email_ops.pretty_printer import PrettyPrinter

# ---------------------------------------------------------------------------
# Global env instance (stateful for demo)
# ---------------------------------------------------------------------------
_DEFAULT_TASK = TaskConfig(
    task_id="easy",
    description="Classify emails correctly",
    difficulty="easy",
    max_steps=30,
    inbox_size=5,
    reward_components=["classification"],
)
_env = EmailOpsEnv(task_config=_DEFAULT_TASK, seed=42)
_printer = PrettyPrinter()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
api = FastAPI(title="openenv-email-ops", version="1.0.0")


@api.get("/")
def root():
    return {"name": "openenv-email-ops", "version": "1.0.0", "status": "running"}


@api.get("/health")
def health():
    return {"status": "healthy"}


@api.post("/reset")
def reset(seed: int = 42):
    obs = _env.reset(seed=seed)
    return JSONResponse({"observation": obs.model_dump(exclude={"current_email": {"ground_truth"}}), "status": "ok"})


@api.post("/step")
def step(action_type: str = "classify_email", value: str = "spam"):
    action = Action(action_type=action_type, value=value)
    obs, reward, done, info = _env.step(action)
    return JSONResponse({
        "observation": obs.model_dump(exclude={"current_email": {"ground_truth"}}),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    })


@api.get("/state")
def state():
    return JSONResponse(_env.state())


# ---------------------------------------------------------------------------
# Gradio UI (dry-run demo)
# ---------------------------------------------------------------------------
_MOCK = [
    Action(action_type="classify_email", value="important"),
    Action(action_type="prioritize_email", value="high"),
    Action(action_type="route_email", value="support"),
    Action(action_type="generate_reply",
           value="Hello, thank you for reaching out. We will respond shortly."),
]


def run_demo():
    lines = ["openenv-email-ops — Dry Run Demo", "=" * 40, ""]
    step_idx = 0
    for task_id in ["easy", "medium", "hard"]:
        env = EmailOpsEnv.from_yaml("openenv.yaml", task_id, seed=42)
        obs = env.reset(seed=42)
        done = False
        total = 0.0
        breakdown: dict[str, float] = {}
        while not done:
            action = _MOCK[step_idx % len(_MOCK)]
            step_idx += 1
            obs, reward, done, _ = env.step(action)
            total = reward.episode_reward
            for k, v in reward.breakdown.items():
                breakdown[k] = breakdown.get(k, 0.0) + v
        lines.append(f"=== Task: {task_id.upper()} ===")
        lines.append(f"Total reward: {total:.4f}")
        for k, v in breakdown.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")
    return "\n".join(lines)


with gr.Blocks(title="openenv-email-ops") as demo:
    gr.Markdown("# 📧 openenv-email-ops\n**Enterprise Inbox Simulation — OpenEnv RL Environment**")
    gr.Markdown("API endpoints: `POST /reset` · `POST /step` · `GET /state`")
    btn = gr.Button("▶ Run Dry-Run Inference (All 3 Tasks)", variant="primary")
    out = gr.Textbox(label="Output", lines=25, interactive=False)
    btn.click(fn=run_demo, outputs=out)

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(api, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
