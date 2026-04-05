"""Seeded email stream generation with noise injection."""

from __future__ import annotations

import math
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
# Hard-task email pool: ambiguous / multi-intent emails with conflicting signals
# Each template has urgency >= 0.7 but priority "medium" or "low" (conflicting),
# a multi-intent subject, and a dominant_intent field for reply grading.
# ---------------------------------------------------------------------------

HARD_EMAIL_POOL: list[dict] = [
    {
        "subject": "Refund request and product question",
        "body": (
            "URGENT: I need an immediate refund for my last order — the item arrived broken. "
            "Also, while I have you, can you tell me if the new model supports Bluetooth? "
            "I'm considering a replacement purchase."
        ),
        "urgency": 0.8,
        "classification": "important",
        "priority": "medium",
        "route": "support",
        "dominant_intent": "refund",
    },
    {
        "subject": "Critical feedback and partnership inquiry",
        "body": (
            "I must say, your latest update has caused serious disruptions to our workflow — "
            "this is completely unacceptable and needs to be fixed immediately. "
            "That said, we are also exploring a potential partnership opportunity and would "
            "love to schedule a call with your business development team."
        ),
        "urgency": 0.75,
        "classification": "important",
        "priority": "medium",
        "route": "support",
        "dominant_intent": "feedback",
    },
    {
        "subject": "Account suspension notice and upgrade request",
        "body": (
            "I received a notice that my account has been suspended, which is extremely urgent "
            "and needs to be resolved right away. Additionally, I wanted to ask about upgrading "
            "to the premium plan — can you send me the pricing details?"
        ),
        "urgency": 0.85,
        "classification": "important",
        "priority": "low",
        "route": "support",
        "dominant_intent": "account suspension",
    },
    {
        "subject": "Re: Invoice dispute — also interested in new features",
        "body": (
            "Following up on my previous email about the incorrect invoice amount. "
            "This is time-sensitive as our accounting team needs to close the books by Friday. "
            "On a separate note, I saw your announcement about the new analytics dashboard — "
            "we would be very interested in a demo."
        ),
        "urgency": 0.78,
        "classification": "important",
        "priority": "medium",
        "route": "sales",
        "dominant_intent": "invoice dispute",
    },
    {
        "subject": "Urgent: Service outage report and feature suggestion",
        "body": (
            "Our entire team has been unable to access the platform for the past two hours — "
            "this is causing significant business impact and must be escalated immediately. "
            "While I have your attention, I also wanted to suggest adding a bulk export feature "
            "to the reporting module, which would save us a lot of time."
        ),
        "urgency": 0.92,
        "classification": "important",
        "priority": "low",
        "route": "escalation",
        "dominant_intent": "service outage",
    },
    {
        "subject": "Fwd: Complaint about delivery AND loyalty rewards question",
        "body": (
            "I am very disappointed — my order was supposed to arrive three days ago and "
            "I still haven't received it. This is completely unacceptable. "
            "By the way, I also noticed my loyalty points haven't been updated in two months. "
            "Can you look into both issues?"
        ),
        "urgency": 0.72,
        "classification": "important",
        "priority": "medium",
        "route": "support",
        "dominant_intent": "delivery complaint",
    },
]

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

    def generate(self, size: int, seed: int, difficulty: str = "easy") -> list[Email]:
        """Generate *size* emails deterministically from *seed*.

        When size >= 4, guarantees at least one email of each sender_type.
        When difficulty == "hard", at least 40% of emails are drawn from
        HARD_EMAIL_POOL (with dominant_intent populated).
        """
        rng = random.Random(seed)
        emails: list[Email] = []

        if difficulty == "hard":
            hard_count = max(1, math.ceil(size * 0.4))
            normal_count = size - hard_count

            # Draw hard-pool emails
            for _ in range(hard_count):
                template = rng.choice(HARD_EMAIL_POOL)
                emails.append(self._make_hard_email(template, rng))

            # Fill remaining from normal pool
            if normal_count >= 4:
                guaranteed = list(_SENDER_TYPES)
                rng.shuffle(guaranteed)
                for sender_type in guaranteed:
                    emails.append(self._make_email(sender_type, rng))
                for _ in range(normal_count - 4):
                    sender_type = rng.choice(_SENDER_TYPES)
                    emails.append(self._make_email(sender_type, rng))
            else:
                for _ in range(normal_count):
                    sender_type = rng.choice(_SENDER_TYPES)
                    emails.append(self._make_email(sender_type, rng))

            rng.shuffle(emails)
        elif size >= 4:
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

    def _make_hard_email(self, template: dict, rng: random.Random) -> Email:
        """Construct an Email from a HARD_EMAIL_POOL template with dominant_intent set."""
        subject, body = _apply_noise(template["subject"], template["body"], rng)

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
            sender_type="customer",  # hard emails are always customer-type
            urgency_score=round(urgency_score, 4),
            ground_truth=ground_truth,
            dominant_intent=template.get("dominant_intent"),
        )
