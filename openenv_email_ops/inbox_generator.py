"""Seeded email stream generation with noise injection."""

from __future__ import annotations

import random
import uuid
from typing import Literal

from openenv_email_ops.models import Email, GroundTruth

# ---------------------------------------------------------------------------
# Email templates per sender_type
# ---------------------------------------------------------------------------

_CUSTOMER_TEMPLATES = [
    {
        "subject": "Issue with my recent order",
        "body": "Hi, I placed an order last week and it still hasn't arrived. Can you help me track it?",
        "urgency": 0.5,
        "classification": "important",
        "priority": "medium",
        "route": "support",
    },
    {
        "subject": "Billing question",
        "body": "Hello, I was charged twice for my subscription this month. Please look into this.",
        "urgency": 0.6,
        "classification": "important",
        "priority": "medium",
        "route": "support",
    },
    {
        "subject": "Product feedback",
        "body": "Just wanted to share some feedback on your latest product update. Overall pretty good!",
        "urgency": 0.2,
        "classification": "promotion",
        "priority": "low",
        "route": "sales",
    },
    {
        "subject": "Account access problem",
        "body": "I cannot log into my account. I tried resetting my password but the email never arrived.",
        "urgency": 0.7,
        "classification": "important",
        "priority": "high",
        "route": "support",
    },
    {
        "subject": "Refund request",
        "body": "I would like to request a refund for my purchase. The item was damaged on arrival.",
        "urgency": 0.65,
        "classification": "important",
        "priority": "medium",
        "route": "support",
    },
]

_SPAMMER_TEMPLATES = [
    {
        "subject": "You WON a FREE iPhone!!!",
        "body": "Congratulations! Click here to claim your prize. Limited time offer. Act now!!!",
        "urgency": 0.1,
        "classification": "spam",
        "priority": "low",
        "route": "support",
    },
    {
        "subject": "Make $5000 a day from home",
        "body": "Our proven system lets you earn thousands daily. No experience needed. Sign up today!",
        "urgency": 0.05,
        "classification": "spam",
        "priority": "low",
        "route": "support",
    },
    {
        "subject": "Exclusive deal just for you",
        "body": "Buy now and get 90% off everything. This offer expires in 1 hour. Don't miss out!",
        "urgency": 0.1,
        "classification": "spam",
        "priority": "low",
        "route": "sales",
    },
    {
        "subject": "Your account has been compromised",
        "body": "Click this link immediately to verify your identity and secure your account.",
        "urgency": 0.15,
        "classification": "spam",
        "priority": "low",
        "route": "support",
    },
    {
        "subject": "Lose 30 pounds in 30 days",
        "body": "Doctors hate this one weird trick. Order our miracle supplement today risk-free!",
        "urgency": 0.05,
        "classification": "spam",
        "priority": "low",
        "route": "support",
    },
]

_VIP_TEMPLATES = [
    {
        "subject": "Urgent: Contract renewal discussion",
        "body": "We need to discuss the terms of our contract renewal before end of quarter. Please schedule a call.",
        "urgency": 0.9,
        "classification": "important",
        "priority": "high",
        "route": "sales",
    },
    {
        "subject": "Executive briefing required",
        "body": "I need a full briefing on the Q3 performance metrics before the board meeting on Friday.",
        "urgency": 0.85,
        "classification": "important",
        "priority": "high",
        "route": "escalation",
    },
    {
        "subject": "Partnership opportunity",
        "body": "We are interested in exploring a strategic partnership. Can we set up a meeting this week?",
        "urgency": 0.75,
        "classification": "important",
        "priority": "high",
        "route": "sales",
    },
    {
        "subject": "Critical issue escalation",
        "body": "The production outage is affecting our operations. This needs immediate attention from your team.",
        "urgency": 0.95,
        "classification": "important",
        "priority": "high",
        "route": "escalation",
    },
    {
        "subject": "Investment proposal review",
        "body": "Please review the attached investment proposal and provide your feedback by tomorrow morning.",
        "urgency": 0.8,
        "classification": "important",
        "priority": "high",
        "route": "sales",
    },
]

_INTERNAL_TEMPLATES = [
    {
        "subject": "Team standup notes",
        "body": "Here are the notes from today's standup. Please review and add any missing items.",
        "urgency": 0.3,
        "classification": "promotion",
        "priority": "low",
        "route": "support",
    },
    {
        "subject": "Q4 planning session",
        "body": "We are scheduling the Q4 planning session for next Tuesday. Please confirm your availability.",
        "urgency": 0.4,
        "classification": "important",
        "priority": "medium",
        "route": "support",
    },
    {
        "subject": "IT maintenance window tonight",
        "body": "Reminder: scheduled maintenance will occur from 11pm to 2am. Systems may be unavailable.",
        "urgency": 0.5,
        "classification": "important",
        "priority": "medium",
        "route": "support",
    },
    {
        "subject": "New policy update",
        "body": "Please review the updated expense reimbursement policy attached to this email.",
        "urgency": 0.35,
        "classification": "promotion",
        "priority": "low",
        "route": "support",
    },
    {
        "subject": "All-hands meeting agenda",
        "body": "The agenda for next week's all-hands meeting is ready. Topics include roadmap and OKR review.",
        "urgency": 0.45,
        "classification": "important",
        "priority": "medium",
        "route": "support",
    },
]

_TEMPLATES: dict[str, list[dict]] = {
    "customer": _CUSTOMER_TEMPLATES,
    "spammer": _SPAMMER_TEMPLATES,
    "VIP": _VIP_TEMPLATES,
    "internal": _INTERNAL_TEMPLATES,
}

# ---------------------------------------------------------------------------
# Noise injection helpers
# ---------------------------------------------------------------------------

_INFORMAL_PHRASES = [
    "hey, ",
    "hi there, ",
    "fyi - ",
    "asap pls, ",
    "just a heads up - ",
    "btw, ",
    "quick note: ",
]

_TYPO_SWAPS = {
    "the": "teh",
    "and": "adn",
    "with": "wiht",
    "have": "ahve",
    "your": "yuor",
    "that": "taht",
    "this": "tihs",
    "from": "form",
    "they": "tehy",
    "been": "bene",
}

_AMBIGUOUS_SUBJECT_PREFIXES = [
    "Re: ",
    "Fwd: ",
    "Follow up - ",
    "Quick question about ",
    "Regarding ",
]


def _inject_typo(text: str, rng: random.Random) -> str:
    """Randomly swap one known word with a typo variant."""
    words = text.split()
    for i, word in enumerate(words):
        lower = word.lower().rstrip(".,!?")
        if lower in _TYPO_SWAPS and rng.random() < 0.3:
            words[i] = word.replace(lower, _TYPO_SWAPS[lower])
            break
    return " ".join(words)


def _inject_informal(text: str, rng: random.Random) -> str:
    """Prepend an informal phrase to the text."""
    phrase = rng.choice(_INFORMAL_PHRASES)
    return phrase + text[0].lower() + text[1:]


def _inject_ambiguous_subject(subject: str, rng: random.Random) -> str:
    """Prepend an ambiguous prefix to the subject."""
    prefix = rng.choice(_AMBIGUOUS_SUBJECT_PREFIXES)
    return prefix + subject


def _apply_noise(subject: str, body: str, rng: random.Random) -> tuple[str, str]:
    """Apply random noise to subject and body."""
    noise_roll = rng.random()

    if noise_roll < 0.25:
        body = _inject_typo(body, rng)
    elif noise_roll < 0.50:
        body = _inject_informal(body, rng)
    elif noise_roll < 0.70:
        subject = _inject_ambiguous_subject(subject, rng)
    # else: no noise (30% chance of clean email)

    return subject, body


# ---------------------------------------------------------------------------
# UUID generation from seeded RNG
# ---------------------------------------------------------------------------

def _make_uuid(rng: random.Random) -> str:
    """Generate a UUID4-like string deterministically from the seeded RNG."""
    rand_bytes = bytes(rng.getrandbits(8) for _ in range(16))
    # Set version 4 bits
    rand_bytes = bytearray(rand_bytes)
    rand_bytes[6] = (rand_bytes[6] & 0x0F) | 0x40
    rand_bytes[8] = (rand_bytes[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(rand_bytes)))


# ---------------------------------------------------------------------------
# InboxGenerator
# ---------------------------------------------------------------------------

_SENDER_TYPES: list[Literal["customer", "spammer", "VIP", "internal"]] = [
    "customer",
    "spammer",
    "VIP",
    "internal",
]


class InboxGenerator:
    """Generates a seeded, reproducible list of Email objects with noise."""

    def generate(self, size: int, seed: int) -> list[Email]:
        """Generate *size* emails deterministically from *seed*.

        When size >= 4, guarantees at least one email of each sender_type.
        """
        rng = random.Random(seed)
        emails: list[Email] = []

        if size >= 4:
            # Build guaranteed set (one per sender_type), then shuffle it
            guaranteed = list(_SENDER_TYPES)
            rng.shuffle(guaranteed)
            for sender_type in guaranteed:
                emails.append(self._make_email(sender_type, rng))

            # Fill remaining slots randomly
            for _ in range(size - 4):
                sender_type = rng.choice(_SENDER_TYPES)
                emails.append(self._make_email(sender_type, rng))
        else:
            for _ in range(size):
                sender_type = rng.choice(_SENDER_TYPES)
                emails.append(self._make_email(sender_type, rng))

        return emails

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_email(
        self,
        sender_type: Literal["customer", "spammer", "VIP", "internal"],
        rng: random.Random,
    ) -> Email:
        template = rng.choice(_TEMPLATES[sender_type])

        subject, body = _apply_noise(template["subject"], template["body"], rng)

        # Urgency: use template base with small random jitter
        base_urgency: float = template["urgency"]
        jitter = rng.uniform(-0.05, 0.05)
        urgency_score = max(0.0, min(1.0, base_urgency + jitter))

        ground_truth = GroundTruth(
            correct_classification=template["classification"],
            correct_priority=template["priority"],
            correct_route=template["route"],
        )

        return Email(
            id=_make_uuid(rng),
            subject=subject,
            body=body,
            sender_type=sender_type,
            urgency_score=round(urgency_score, 4),
            ground_truth=ground_truth,
        )
