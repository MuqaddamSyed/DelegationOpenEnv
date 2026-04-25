from __future__ import annotations

import random
import uuid
from typing import Dict, List, Optional

from delegation_gauntlet.models import PendingDecision, Priority, ScenarioType


class Scenario:
    def __init__(self, scenario_type: ScenarioType, seed: int):
        self.scenario_type = scenario_type
        self.seed = seed
        self.essential_facts: Dict[str, str] = {}
        self.initial_pending: List[PendingDecision] = []

    @staticmethod
    def sample(rng: random.Random, scenario: Optional[ScenarioType], seed: int) -> "Scenario":
        st = scenario or rng.choice(list(ScenarioType))
        sc = Scenario(st, seed=seed)
        sc._populate(rng)
        return sc

    def _populate(self, rng: random.Random) -> None:
        # Shared "exec assistant" facts
        self.essential_facts["boss_diet"] = rng.choice(["vegetarian", "vegan", "no preference"])
        self.essential_facts["preferred_airline"] = rng.choice(["IndiGo", "Vistara", "Air India"])
        self.essential_facts["budget_sensitivity"] = rng.choice(["tight", "normal"])
        self.essential_facts["timezone_sensitive"] = rng.choice(["yes", "no"])

        if self.scenario_type == ScenarioType.CONFERENCE_PLANNING:
            self.essential_facts["conference_city"] = rng.choice(["Bengaluru", "Mumbai", "Delhi"])
            self.essential_facts["attendees_count"] = "8"
            self.initial_pending = [
                self._pd("Confirm conference dates and venue shortlist", Priority.critical, deadline=18),
                self._pd("Draft attendee travel preference form", Priority.high, deadline=30),
                self._pd("Collect bids from 3 venues within budget", Priority.critical, deadline=40),
                self._pd("Book flights + hotels for attendees", Priority.critical, deadline=80, requires_boss=True),
            ]

        elif self.scenario_type == ScenarioType.PRODUCT_LAUNCH:
            self.essential_facts["launch_date"] = rng.choice(["Week 2", "Week 3"])
            self.initial_pending = [
                self._pd("Schedule launch sync across timezones", Priority.critical, deadline=12),
                self._pd("Coordinate announcement calendar with marketing", Priority.high, deadline=35),
                self._pd("Prepare stakeholder update email draft", Priority.medium, deadline=45),
            ]

        elif self.scenario_type == ScenarioType.EXECUTIVE_TRAVEL:
            self.essential_facts["trip_cities"] = ", ".join(rng.sample(["Singapore", "Dubai", "London", "Tokyo"], k=2))
            self.essential_facts["hotel_pref"] = rng.choice(["quiet", "near office", "suite if possible"])
            self.initial_pending = [
                self._pd("Draft travel itinerary options", Priority.critical, deadline=10),
                self._pd("Book multi-city travel", Priority.critical, deadline=30, requires_boss=True),
                self._pd("Schedule pre-brief meeting", Priority.high, deadline=20),
            ]

        elif self.scenario_type == ScenarioType.CRISIS_MANAGEMENT:
            self.essential_facts["crisis_type"] = rng.choice(["PR incident", "outage", "legal issue"])
            self.initial_pending = [
                self._pd("Assemble crisis response meeting", Priority.critical, deadline=6),
                self._pd("Draft internal status update", Priority.critical, deadline=10),
                self._pd("Reschedule non-essential meetings", Priority.high, deadline=16),
            ]

        elif self.scenario_type == ScenarioType.VENDOR_NEGOTIATION:
            self.essential_facts["vendor_category"] = rng.choice(["cloud", "security", "contractor"])
            self.initial_pending = [
                self._pd("Request final bids from top 2 vendors", Priority.critical, deadline=15),
                self._pd("Compare bids and summarize tradeoffs", Priority.high, deadline=30),
                self._pd("Negotiate price within budget", Priority.critical, deadline=40, requires_boss=True),
            ]

    def _pd(self, desc: str, pr: Priority, deadline: int, requires_boss: bool = False) -> PendingDecision:
        return PendingDecision(
            id=str(uuid.uuid4())[:8],
            description=desc,
            priority=pr,
            created_turn=0,
            deadline_turn=deadline,
            requires_boss_approval=requires_boss,
            resolved=False,
        )

