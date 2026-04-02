"""Baseline inference script for openenv-email-ops.

Runs all three tasks (easy, medium, hard) sequentially using an OpenAI-backed agent.

Usage:
    python inference.py                  # uses OPENAI_API_KEY env var
    python inference.py --dry-run        # mock mode, no API calls needed
"""

from __future__ import annotations

import argparse
import os
import sys

from openenv_email_ops.env import EmailOpsEnv
from openenv_email_ops.models import Action
from openenv_email_ops.parser import Parser
from openenv_email_ops.pretty_printer import PrettyPrinter

SYSTEM_PROMPT = """You are an expert AI email operations agent for an enterprise inbox management system.

Your job is to process one email at a time and respond with EXACTLY ONE JSON action.

## RESPONSE FORMAT (strict)
Respond with only this JSON — no explanation, no markdown, no extra text:
{"action_type": "<type>", "value": "<value>"}

## CLASSIFICATION RULES (classify_email)
Use these rules to pick the correct value:

- "spam"      → sender is a spammer, email contains prizes/offers/scams/phishing/weight-loss/get-rich schemes
- "important" → sender is VIP or internal staff, OR email is about billing/account/refund/outage/contract/escalation/urgent issue
- "promotion" → marketing, newsletters, product feedback, announcements that are NOT urgent

## PRIORITIZATION RULES (prioritize_email)
- "high"   → urgency_score > 0.7, OR sender is VIP, OR subject contains urgent/critical/outage/escalation
- "medium" → urgency_score 0.4–0.7, OR billing/account/support issues
- "low"    → urgency_score < 0.4, OR spam/promotion/feedback

## ROUTING RULES (route_email)
- "escalation" → VIP sender, OR subject contains critical/outage/executive/board/contract/investment
- "sales"      → partnership/proposal/pricing/renewal/revenue topics
- "support"    → everything else (billing, account, technical issues, general inquiries, spam)

## REPLY RULES (generate_reply)
Write a professional reply of at least 30 characters. Start with a greeting (Hello/Hi/Dear).
Reference the email subject. Do NOT use placeholders like [NAME] or TODO.

## DECISION GUIDE
Look at the current email's sender_type and subject carefully:
- sender_type "spammer" → always classify as "spam", priority "low", route "support"
- sender_type "VIP"     → always classify as "important", priority "high", route "escalation" or "sales"
- sender_type "internal"→ classify as "important" or "promotion" based on urgency
- sender_type "customer"→ classify as "important" for issues/billing, "promotion" for feedback

Respond with ONLY the JSON. No other text."""

SEED = 42
TASKS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Mock agent for --dry-run mode (no API calls)
# ---------------------------------------------------------------------------

_MOCK_SEQUENCE = [
    Action(action_type="classify_email", value="important"),
    Action(action_type="prioritize_email", value="high"),
    Action(action_type="route_email", value="support"),
    Action(
        action_type="generate_reply",
        value="Hello, thank you for reaching out. We have received your message and will respond shortly.",
    ),
]


class MockClient:
    """Cycles through a fixed action sequence without calling any API."""

    def __init__(self) -> None:
        self._step = 0

    def get_action(self, obs) -> Action:  # noqa: ARG002
        action = _MOCK_SEQUENCE[self._step % len(_MOCK_SEQUENCE)]
        self._step += 1
        return action


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def run_task_live(client, model_name: str, task_id: str) -> None:
    """Run a task episode using the OpenAI API."""
    from openai import OpenAI  # imported here so dry-run doesn't need openai installed

    printer = PrettyPrinter()
    parser = Parser()
    env = EmailOpsEnv.from_yaml("openenv.yaml", task_id, seed=SEED)

    obs = env.reset(seed=SEED)
    done = False
    total_reward = 0.0
    score_breakdown: dict[str, float] = {}

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

        obs, reward, done, _ = env.step(action)
        total_reward = reward.episode_reward
        for k, v in reward.breakdown.items():
            score_breakdown[k] = score_breakdown.get(k, 0.0) + v

    _print_results(task_id, total_reward, score_breakdown)


def run_task_dry(mock: MockClient, task_id: str) -> None:
    """Run a task episode using the mock agent (no API calls)."""
    env = EmailOpsEnv.from_yaml("openenv.yaml", task_id, seed=SEED)

    obs = env.reset(seed=SEED)
    done = False
    total_reward = 0.0
    score_breakdown: dict[str, float] = {}

    while not done:
        action = mock.get_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward = reward.episode_reward
        for k, v in reward.breakdown.items():
            score_breakdown[k] = score_breakdown.get(k, 0.0) + v

    _print_results(task_id, total_reward, score_breakdown)


def _print_results(task_id: str, total_reward: float, breakdown: dict[str, float]) -> None:
    print(f"\n=== Task: {task_id} ===")
    print("Episode complete.")
    print(f"Total reward: {total_reward:.4f}")
    print("Score breakdown:")
    for component, value in breakdown.items():
        print(f"  {component}: {value:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="openenv-email-ops inference")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with a mock agent (no OpenAI API calls). Useful for local testing.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Using mock agent — no API calls will be made.")
        mock = MockClient()
        for task_id in TASKS:
            run_task_dry(mock, task_id)
        return

    # Live mode — requires OPENAI_API_KEY
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Tip: run with --dry-run to test without an API key.", file=sys.stderr)
        sys.exit(1)

    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    for task_id in TASKS:
        run_task_live(client, model_name, task_id)


if __name__ == "__main__":
    main()
