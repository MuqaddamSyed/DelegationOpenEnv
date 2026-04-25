from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from delegation_gauntlet.models import Priority, WorldState


@dataclass(frozen=True)
class RubricScore:
    name: str
    weight: float
    score: float  # 0..1
    details: Dict[str, Any]


class TaskCompletionRubric:
    weight = 0.25

    def score(self, st: WorldState) -> RubricScore:
        total_critical = float(st.initial_tasks_by_priority.get("critical", 0))
        total_high = float(st.initial_tasks_by_priority.get("high", 0))

        completed_ids = set(st.tasks_completed)
        completed_critical = len([p for p in st.pending_items if p.priority == Priority.critical and p.id in completed_ids])
        completed_high = len([p for p in st.pending_items if p.priority == Priority.high and p.id in completed_ids])

        critical_part = (completed_critical / total_critical) if total_critical > 0 else 1.0
        high_part = (completed_high / total_high) if total_high > 0 else 1.0

        # Partial credit: high priority is worth half of critical in this rubric.
        s = 0.75 * critical_part + 0.25 * high_part
        return RubricScore(
            name="task_completion",
            weight=self.weight,
            score=float(max(0.0, min(1.0, s))),
            details={
                "completed_critical": completed_critical,
                "total_critical": int(total_critical),
                "completed_high": completed_high,
                "total_high": int(total_high),
            },
        )


class AutonomyCalibrationRubric:
    weight = 0.20

    def score(self, st: WorldState) -> RubricScore:
        total_decisions = max(1, int(st.decisions_total))
        boss_ask_rate = float(st.boss_interventions) / float(total_decisions)

        # EXACT requirement: full credit only in [0.05, 0.20].
        if 0.05 <= boss_ask_rate <= 0.20:
            s = 1.0
        elif boss_ask_rate < 0.05:
            # Smooth decrease to 0 as rate -> 0
            s = max(0.0, boss_ask_rate / 0.05)
        else:
            # Smooth decrease above 0.20; reach ~0 at 0.50 (very ask-heavy)
            s = max(0.0, 1.0 - ((boss_ask_rate - 0.20) / 0.30))

        return RubricScore(
            name="autonomy_calibration",
            weight=self.weight,
            score=float(max(0.0, min(1.0, s))),
            details={
                "boss_interventions": int(st.boss_interventions),
                "total_decisions": int(total_decisions),
                "boss_ask_rate": boss_ask_rate,
                "goldilocks_min": 0.05,
                "goldilocks_max": 0.20,
            },
        )


class PriorityAlignmentRubric:
    weight = 0.20

    def score(self, st: WorldState) -> RubricScore:
        # Penalize having critical items pending late into the episode.
        pending_critical = len([p for p in st.pending_items if (not p.resolved) and p.priority == Priority.critical])
        pending_high = len([p for p in st.pending_items if (not p.resolved) and p.priority == Priority.high])

        # Normalize to initial counts.
        init_c = max(1, int(st.initial_tasks_by_priority.get("critical", 0)))
        init_h = max(1, int(st.initial_tasks_by_priority.get("high", 0)))

        critical_backlog = pending_critical / init_c
        high_backlog = pending_high / init_h

        # Anti-gaming: repeated do_nothing while critical items remain should score poorly.
        do_nothing_ratio = float(st.do_nothing_actions) / float(max(1, st.decisions_total))
        do_nothing_penalty = 0.0
        if pending_critical > 0:
            do_nothing_penalty = min(0.35, 0.8 * do_nothing_ratio)

        s = 1.0 - (0.75 * critical_backlog + 0.25 * high_backlog + do_nothing_penalty)
        return RubricScore(
            name="priority_alignment",
            weight=self.weight,
            score=float(max(0.0, min(1.0, s))),
            details={
                "pending_critical": pending_critical,
                "pending_high": pending_high,
                "do_nothing_ratio": do_nothing_ratio,
                "do_nothing_penalty": do_nothing_penalty,
            },
        )


class InformationEfficiencyRubric:
    weight = 0.15

    def score(self, st: WorldState) -> RubricScore:
        # Penalize rereads; reward reading a small, relevant subset.
        total_reads = len(st.messages_read_ids)
        unique_reads = len(set(st.messages_read_ids))
        rereads = max(0, total_reads - unique_reads)

        # Target: read enough to stay informed but avoid rereading.
        # Score decreases with rereads and with extreme under-reading.
        if total_reads == 0:
            base = 0.0
        else:
            base = min(1.0, unique_reads / max(10.0, 0.12 * len(st.inbox)))

        reread_pen = min(1.0, rereads / 20.0)
        s = max(0.0, base * (1.0 - 0.7 * reread_pen))

        return RubricScore(
            name="information_efficiency",
            weight=self.weight,
            score=float(max(0.0, min(1.0, s))),
            details={"total_reads": total_reads, "unique_reads": unique_reads, "rereads": rereads},
        )


class BudgetAdherenceRubric:
    weight = 0.10

    def score(self, st: WorldState) -> RubricScore:
        if st.budget_spent <= st.budget_limit:
            # Bonus for coming in under budget AND completing all critical tasks.
            pending_critical = len([p for p in st.pending_items if (not p.resolved) and p.priority == Priority.critical])
            under_ratio = max(0.0, (st.budget_limit - st.budget_spent) / max(1.0, st.budget_limit))
            bonus = 0.15 * under_ratio if pending_critical == 0 else 0.0
            s = min(1.0, 1.0 + bonus)
        else:
            overspend = st.budget_spent - st.budget_limit
            overspend_ratio = overspend / max(1.0, st.budget_limit)
            s = max(0.0, 1.0 - 2.5 * overspend_ratio)

        return RubricScore(
            name="budget_adherence",
            weight=self.weight,
            score=float(max(0.0, min(1.0, s))),
            details={"budget_spent": float(st.budget_spent), "budget_limit": float(st.budget_limit)},
        )


class DelegationQualityRubric:
    weight = 0.10

    def score(self, st: WorldState) -> RubricScore:
        # Proxy: if there are SubAgent messages saying "instructions unclear", penalize.
        sub_msgs = [m for m in st.inbox if m.sender == "SubAgent"]
        if not sub_msgs:
            return RubricScore(name="delegation_quality", weight=self.weight, score=1.0, details={"delegations": 0})

        unclear = sum(1 for m in sub_msgs if "instructions unclear" in m.body.lower())
        s = 1.0 - min(1.0, unclear / max(1.0, len(sub_msgs)))
        return RubricScore(
            name="delegation_quality",
            weight=self.weight,
            score=float(max(0.0, min(1.0, s))),
            details={"delegations": len(sub_msgs), "unclear": unclear},
        )


class RewardEngine:
    def __init__(self):
        self.rubrics = [
            TaskCompletionRubric(),
            AutonomyCalibrationRubric(),
            PriorityAlignmentRubric(),
            InformationEfficiencyRubric(),
            BudgetAdherenceRubric(),
            DelegationQualityRubric(),
        ]

    def score(self, st: WorldState, *, partial: bool = False) -> Tuple[float, Dict[str, Any]]:
        rubric_scores: List[RubricScore] = [r.score(st) for r in self.rubrics]
        raw = 0.0
        for rs in rubric_scores:
            raw += float(rs.weight) * float(rs.score)

        raw = float(max(0.0, min(1.0, raw)))
        reward = (raw - 0.5) * 2.0  # maps [0,1] -> [-1,1]

        breakdown = {
            "raw": raw,
            "reward": reward,
            "rubrics": [
                {"name": rs.name, "weight": rs.weight, "score": rs.score, "details": rs.details}
                for rs in rubric_scores
            ],
        }
        return float(reward), breakdown

