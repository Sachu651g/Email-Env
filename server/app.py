"""
server/app.py — OpenEnv server entry point for openenv-email-ops.
Exposes reset(), step(), state() as HTTP endpoints.
"""
from __future__ import annotations
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action, TaskConfig

_TASK = TaskConfig(
    task_id="easy", description="Classify emails", difficulty="easy",
    max_steps=30, inbox_size=5, reward_components=["classification"],
)
_env = EmailOpsEnv(task_config=_TASK, seed=42)

app = FastAPI(title="openenv-email-ops", version="1.0.0")


@app.get("/")
def root():
    return {"name": "openenv-email-ops", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(seed: int = 42):
    obs = _env.reset(seed=seed)
    return {"observation": obs.model_dump(exclude={"current_email": {"ground_truth"}}), "status": "ok"}


@app.post("/step")
def step(action_type: str = "classify_email", value: str = "spam"):
    try:
        action = Action(action_type=action_type, value=value)
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(exclude={"current_email": {"ground_truth"}}),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/state")
def state():
    return _env.state()


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
