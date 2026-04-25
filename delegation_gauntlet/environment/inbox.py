from __future__ import annotations

import random
import uuid
from typing import List, Optional, Tuple

from delegation_gauntlet.models import Channel, Message, Priority, ScenarioType


class InboxGenerator:
    def __init__(self, rng: random.Random):
        self._rng = rng

    def generate_stream(self, scenario: ScenarioType, max_turns: int) -> List[Message]:
        # 50-80 total messages across episode
        n = self._rng.randint(50, 80)
        stream: List[Message] = []

        # A mid-episode family emergency
        emergency_turn = int(max_turns * 0.55)
        stream.append(
            self._msg(
                sender="Family",
                channel=Channel.email,
                priority=Priority.critical,
                subject="Emergency: please call ASAP",
                body="Something happened at home. Please call when you see this.",
                requires_action=True,
                deadline_turn=emergency_turn + 2,
                created_turn=emergency_turn,
            )
        )

        for _ in range(n - 1):
            created_turn = self._rng.randint(0, max_turns - 1)
            channel = self._rng.choice([Channel.email, Channel.slack])

            kind = self._rng.random()
            if kind < 0.15:
                pr = Priority.critical
                requires_action = True
            elif kind < 0.35:
                pr = Priority.high
                requires_action = True
            elif kind < 0.65:
                pr = Priority.medium
                requires_action = self._rng.random() < 0.6
            elif kind < 0.85:
                pr = Priority.low
                requires_action = self._rng.random() < 0.2
            else:
                pr = Priority.distractor
                requires_action = False

            deadline = None
            if requires_action and pr in (Priority.critical, Priority.high, Priority.medium):
                slack = {Priority.critical: 6, Priority.high: 14, Priority.medium: 28}[pr]
                deadline = min(max_turns - 1, created_turn + self._rng.randint(2, slack))

            subject, body, sender = self._template_message(scenario, pr, channel)
            stream.append(
                self._msg(
                    sender=sender,
                    channel=channel,
                    priority=pr,
                    subject=subject,
                    body=body,
                    requires_action=requires_action,
                    deadline_turn=deadline,
                    created_turn=created_turn,
                )
            )

        stream.sort(key=lambda m: (m.created_turn, m.priority.value))
        return stream

    def _template_message(self, scenario: ScenarioType, pr: Priority, channel: Channel) -> Tuple[str, str, str]:
        sender_pool = {
            ScenarioType.CONFERENCE_PLANNING: ["VenueOps", "TravelDesk", "Finance", "Ops"],
            ScenarioType.PRODUCT_LAUNCH: ["Marketing", "EngLead", "PM", "Comms"],
            ScenarioType.EXECUTIVE_TRAVEL: ["TravelDesk", "Assistant2", "HotelConcierge", "VisaAgent"],
            ScenarioType.CRISIS_MANAGEMENT: ["SRE", "Legal", "PR", "EngOncall"],
            ScenarioType.VENDOR_NEGOTIATION: ["VendorA", "VendorB", "Procurement", "Finance"],
        }[scenario]
        sender = self._rng.choice(sender_pool)

        if pr == Priority.distractor:
            return (
                "Newsletter: 10 productivity tips",
                "Thought you might like these tips. No action needed.",
                "Newsletters",
            )

        if channel == Channel.slack:
            subject = "Slack ping"
            body = {
                Priority.critical: "Need an answer now — what's the decision?",
                Priority.high: "Quick check: status update?",
                Priority.medium: "FYI, please review when you can.",
                Priority.low: "No rush: can you take a look later?",
            }[pr]
            return (subject, body, sender)

        # email
        if pr == Priority.critical:
            return ("URGENT: approval needed", "Please approve immediately; deadline is today.", sender)
        if pr == Priority.high:
            return ("Action required", "Could you handle this today? Thanks.", sender)
        if pr == Priority.medium:
            return ("Request", "Please review and respond when possible.", sender)
        return ("FYI", "Sharing this for visibility.", sender)

    def _msg(
        self,
        sender: str,
        channel: Channel,
        priority: Priority,
        subject: str,
        body: str,
        requires_action: bool,
        deadline_turn: Optional[int],
        created_turn: int,
    ) -> Message:
        return Message(
            id=str(uuid.uuid4())[:10],
            sender=sender,
            channel=channel,
            priority=priority,
            subject=subject,
            body=body,
            requires_action=requires_action,
            deadline_turn=deadline_turn,
            is_adversarial=False,
            injected_by=None,
            created_turn=created_turn,
            read=False,
        )

