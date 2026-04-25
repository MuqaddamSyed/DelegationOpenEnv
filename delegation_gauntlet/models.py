from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Channel(str, Enum):
    email = "email"
    slack = "slack"


class Priority(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    distractor = "distractor"


class BossPersonality(str, Enum):
    MICROMANAGER = "MICROMANAGER"
    HANDS_OFF = "HANDS_OFF"
    PASSIVE_AGGRESSIVE = "PASSIVE_AGGRESSIVE"


class ScenarioType(str, Enum):
    CONFERENCE_PLANNING = "CONFERENCE_PLANNING"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    EXECUTIVE_TRAVEL = "EXECUTIVE_TRAVEL"
    CRISIS_MANAGEMENT = "CRISIS_MANAGEMENT"
    VENDOR_NEGOTIATION = "VENDOR_NEGOTIATION"


class CurveballType(str, Enum):
    CONTEXT_POLLUTION = "CONTEXT_POLLUTION"
    AUTHORITY_SPOOFING = "AUTHORITY_SPOOFING"
    PRIORITY_INVERSION = "PRIORITY_INVERSION"
    BUDGET_TRAP = "BUDGET_TRAP"
    DEADLINE_COMPRESSION = "DEADLINE_COMPRESSION"
    DELEGATION_CONFUSION = "DELEGATION_CONFUSION"
    PERMISSION_AMBIGUITY = "PERMISSION_AMBIGUITY"


class Message(BaseModel):
    id: str
    sender: str
    channel: Channel
    priority: Priority
    subject: str
    body: str
    requires_action: bool = False
    deadline_turn: Optional[int] = None
    is_adversarial: bool = False
    injected_by: Optional[CurveballType] = None
    created_turn: int = 0
    read: bool = False


class Event(BaseModel):
    id: str
    title: str
    start_turn: int
    end_turn: int
    attendees: List[str] = Field(default_factory=list)
    location: Optional[str] = None


class Booking(BaseModel):
    id: str
    traveler: str
    destination: str
    depart_turn: int
    return_turn: int
    amount: float
    confirmation: str
    approved_by_boss: bool = False
    created_turn: int = 0


class Email(BaseModel):
    to: str
    subject: str
    body: str
    reply_to: Optional[str] = None
    created_turn: int = 0


class DraftDocument(BaseModel):
    id: str
    title: str
    content: str
    recipients: List[str] = Field(default_factory=list)
    created_turn: int = 0


ActionType = Literal[
    "send_email",
    "book_travel",
    "transfer_funds",
    "create_event",
    "ask_boss",
    "purchase_item",
    "draft_document",
    "delegate",
    "do_nothing",
]


class Action(BaseModel):
    action_type: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    success: bool
    tool_name: str
    irreversible: bool = False
    message: str
    consequences: Dict[str, Any] = Field(default_factory=dict)


class PendingDecision(BaseModel):
    id: str
    description: str
    priority: Priority = Priority.medium
    created_turn: int = 0
    deadline_turn: Optional[int] = None
    requires_boss_approval: bool = False
    resolved: bool = False


class WorldConfig(BaseModel):
    seed: int = 0
    max_episode_steps: int = 200
    turns_per_week: int = 70  # ~3 weeks at 2h/turn (70*3=210)
    budget_min: float = 2000.0
    budget_max: float = 10000.0
    scenario: Optional[ScenarioType] = None
    boss_personality: Optional[BossPersonality] = None
    adversarial_mode: bool = True


class WorldState(BaseModel):
    episode_id: str
    seed: int
    scenario: ScenarioType
    boss_personality: BossPersonality

    current_turn: int = 0
    current_week: int = 1

    budget_spent: float = 0.0
    budget_limit: float = 0.0

    calendar: List[Event] = Field(default_factory=list)
    inbox: List[Message] = Field(default_factory=list)
    sent_emails: List[Email] = Field(default_factory=list)
    bookings: List[Booking] = Field(default_factory=list)
    drafts: List[DraftDocument] = Field(default_factory=list)

    boss_interventions: int = 0
    autonomous_mistakes: int = 0

    tasks_completed: List[str] = Field(default_factory=list)
    tasks_failed: List[str] = Field(default_factory=list)
    pending_items: List[PendingDecision] = Field(default_factory=list)
    initial_tasks_by_priority: Dict[str, int] = Field(default_factory=dict)

    # Tracking for reward rubrics / adversary dynamics
    decisions_total: int = 0
    do_nothing_actions: int = 0
    irreversible_actions: int = 0
    irreversible_without_approval: int = 0
    messages_read_ids: List[str] = Field(default_factory=list)
    message_read_counts: Dict[str, int] = Field(default_factory=dict)
    essential_facts: Dict[str, str] = Field(default_factory=dict)
    approvals: Dict[str, bool] = Field(default_factory=dict)  # keyed by approval_type

    adversary_weights: Dict[CurveballType, float] = Field(default_factory=dict)
    last_curveball: Optional[CurveballType] = None
    last_curveball_caused_failure: bool = False

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_limit - self.budget_spent)

