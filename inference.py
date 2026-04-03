"""
inference.py — openenv-email-ops baseline inference script

Reads environment variables:
  API_BASE_URL  — OpenAI-compatible API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME    — model identifier (default: gpt-4o-mini)
  OPENAI_API_KEY — API key
  HF_TOKEN      — Hugging Face token (optional, for HF-hosted models)

Emits structured stdout logs in [START] / [STEP] / [END] format for automated scoring.

Usage:
  python inference.py              # live mode (requires API key)
  python inference.py --dry-run   # mock mode, no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action
from openenv_email_ops.parser import Parser
from openenv_email_ops.pretty_printer import PrettyPrinter

SEED = 42
TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert AI email operations agent for an enterprise inbox management system.

Your job is to process one email at a time and respond with EXACTLY ONE JSON action.

## RESPONSE FORMAT (strict)
Respond with only this JSON — no explanation, no markdown, no extra text:
{"action_type": "<type>", "value": "<value>"}

## CLASSIFICATION RULES (classify_email)
- "spam"      → sender is a spammer, email contains prizes/offers/scams/phishing
- "important" → sender is VIP or internal, OR email is about billing/account/refund/outage/contract
- "promotion" → marketing, newsletters, product feedback, announcements that are NOT urgent

## PRIORITIZATION RULES (prioritize_email)
- "high"   → urgency_score > 0.7, OR sender is VIP, OR subject contains urgent/critical/outage
- "medium" → urgency_score 0.4–0.7, OR billing/account/support issues
- "low"    → urgency_score < 0.4, OR spam/promotion/feedback

## ROUTING RULES (route_email)
- "escalation" → VIP sender, OR subject contains critical/outage/executive/board/contract
- "sales"      → partnership/proposal/pricing/renewal/revenue topics
- "support"    → everything else

## REPLY RULES (generate_reply)
Write a professional reply of at least 30 characters. Start with Hello/Hi/Dear.

Respond with ONLY the JSON. No other text."""


# ---------------------------------------------------------------------------
# Structured logging helpers — [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------

def log_start(task_id: str, task_config: dict) -> None:
    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "difficulty": task_config.get("difficulty", task_id),
        "inbox_size": task_config.get("inbox_size", 0),
        "max_steps": task_config.get("max_steps", 0),
        "reward_components": task_config.get("reward_components", []),
        "seed": SEED,
        "timestamp": time.time(),
    }), flush=True)


def log_step(task_id: str, step: int, action_type: str, value: str | None,
             step_reward: float, episode_reward: float, breakdown: dict, done: bool) -> None:
    print(json.dumps({
        "event": "STEP",
        "task_id": task_id,
        "step": step,
        "action_type": action_type,
        "value": value,
        "step_reward": round(step_reward, 4),
        "episode_reward": round(episode_reward, 4),
        "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        "done": done,
    }), flush=True)


def log_end(task_id: str, total_reward: float, score_breakdown: dict,
            metrics: dict | None = None) -> None:
    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "total_reward": round(total_reward, 4),
        "score_breakdown": {k: round(v, 4) for k, v in score_breakdown.items()},
        "metrics": metrics or {},
        "seed": SEED,
        "timestamp": time.time(),
    }), flush=True)


# ---------------------------------------------------------------------------
# Smart rule-based mock agent for --dry-run mode
# Reads observation and picks correct action based on sender_type + urgency
# ---------------------------------------------------------------------------

def _smart_action(obs, task_id: str) -> Action:
    """Rule-based agent that reads the observation and picks the best action."""
    email = obs.current_email
    if email is None:
        return Action(action_type="classify_email", value="spam")

    sender = email.sender_type
    urgency = email.urgency_score
    subject = email.subject.lower()
    body = email.body.lower()

    # Determine correct classification, priority, route from content
    if sender == "spammer":
        classification, priority, route = "spam", "low", "support"
    elif sender == "VIP":
        classification = "important"
        priority = "high"
        route = "escalation" if any(
            w in subject for w in ["critical", "outage", "executive", "board", "contract", "investment"]
        ) else "sales"
    elif sender == "internal":
        if urgency > 0.4 or any(w in subject for w in ["maintenance", "urgent", "policy", "planning", "it "]):
            classification = "important"
            priority = "high" if urgency > 0.7 else "medium"
        else:
            classification = "promotion"
            priority = "low"
        route = "support"
    else:  # customer
        if any(w in subject + body for w in ["refund", "billing", "account", "access", "damaged", "charged", "cannot", "problem", "issue", "order"]):
            classification = "important"
            priority = "high" if urgency > 0.7 else "medium"
            route = "support"
        else:
            classification = "promotion"
            priority = "low"
            route = "sales"

    step = obs.step_count
    if task_id == "easy":
        return Action(action_type="classify_email", value=classification)
    elif task_id == "medium":
        cycle = step % 3
        if cycle == 0:
            return Action(action_type="classify_email", value=classification)
        elif cycle == 1:
            return Action(action_type="prioritize_email", value=priority)
        else:
            return Action(action_type="route_email", value=route)
    else:  # hard
        cycle = step % 4
        if cycle == 0:
            return Action(action_type="classify_email", value=classification)
        elif cycle == 1:
            return Action(action_type="prioritize_email", value=priority)
        elif cycle == 2:
            return Action(action_type="route_email", value=route)
        else:
            reply = (
                f"Hello, thank you for your email regarding '{email.subject}'. "
                "We have received your message and our team will respond within 24 hours."
            )
            return Action(action_type="generate_reply", value=reply)


class MockClient:
    def __init__(self, task_id: str = "easy") -> None:
        self._task_id = task_id

    def get_action(self, obs) -> Action:
        return _smart_action(obs, self._task_id)


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_task_live(client, model_name: str, task_id: str) -> None:
    printer = PrettyPrinter()
    parser = Parser()
    env = EmailOpsEnv.from_yaml("openenv.yaml", task_id, seed=SEED)
    task_cfg = {
        "difficulty": env._task_config.difficulty,
        "inbox_size": env._inbox_size,
        "max_steps": env._max_steps,
        "reward_components": env._task_config.reward_components,
    }

    log_start(task_id, task_cfg)
    obs = env.reset(seed=SEED)
    done = False
    total_reward = 0.0
    score_breakdown: dict[str, float] = {}
    step_num = 0
    final_info: dict = {}

    while not done:
        user_message = printer.to_text(obs)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            response_text = response.choices[0].message.content or ""
            action = parser.parse_action(response_text)
        except Exception:
            action = Action(action_type="classify_email", value="spam")

        obs, reward, done, info = env.step(action)
        total_reward = reward.episode_reward
        for k, v in reward.breakdown.items():
            score_breakdown[k] = score_breakdown.get(k, 0.0) + v
        if done:
            final_info = info

        log_step(task_id, step_num, action.action_type, action.value,
                 reward.step_reward, reward.episode_reward, reward.breakdown, done)
        step_num += 1

    log_end(task_id, total_reward, score_breakdown, final_info.get("metrics"))


def run_task_dry(mock: MockClient, task_id: str) -> None:
    env = EmailOpsEnv.from_yaml("openenv.yaml", task_id, seed=SEED)
    task_cfg = {
        "difficulty": env._task_config.difficulty,
        "inbox_size": env._inbox_size,
        "max_steps": env._max_steps,
        "reward_components": env._task_config.reward_components,
    }

    log_start(task_id, task_cfg)
    obs = env.reset(seed=SEED)
    done = False
    total_reward = 0.0
    score_breakdown: dict[str, float] = {}
    step_num = 0
    final_info: dict = {}

    while not done:
        action = mock.get_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward = reward.episode_reward
        for k, v in reward.breakdown.items():
            score_breakdown[k] = score_breakdown.get(k, 0.0) + v
        if done:
            final_info = info

        log_step(task_id, step_num, action.action_type, action.value,
                 reward.step_reward, reward.episode_reward, reward.breakdown, done)
        step_num += 1

    log_end(task_id, total_reward, score_breakdown, final_info.get("metrics"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="openenv-email-ops inference")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock agent mode — no API calls needed")
    args = parser.parse_args()

    if args.dry_run:
        for task_id in TASKS:
            mock = MockClient(task_id=task_id)
            run_task_dry(mock, task_id)
        return

    # Live mode
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        print("Error: OPENAI_API_KEY or HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    client = OpenAI(api_key=api_key, base_url=api_base)

    for task_id in TASKS:
        run_task_live(client, model_name, task_id)


if __name__ == "__main__":
    main()
