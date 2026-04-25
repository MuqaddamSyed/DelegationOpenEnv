from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Ensure repo root on path when running as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Make matplotlib cache writable inside sandboxed runs.
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mplconfig")))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from delegation_gauntlet.environment.world import DelegationWorld
from delegation_gauntlet.models import ActionType


PLOTS_DIR = os.path.join("public", "plots")


def _ensure_plots_dir() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _sample_reasonable_action(rng: random.Random, turn: int) -> Dict[str, Any]:
    """
    Samples a valid action with reasonable params.
    This is used for RANDOM baseline and for lightweight smoke-test "training".
    """
    actions: List[ActionType] = [
        "do_nothing",
        "ask_boss",
        "send_email",
        "create_event",
        "draft_document",
        "delegate",
        "purchase_item",
        "transfer_funds",
        "book_travel",
    ]
    at = rng.choice(actions)

    if at == "do_nothing":
        return {"action_type": at, "params": {}}
    if at == "ask_boss":
        return {"action_type": at, "params": {"question": "Any guidance on priorities or risky actions this week?"}}
    if at == "send_email":
        return {"action_type": at, "params": {"to": "ops@example.com", "subject": "Update", "body": "Status update: working on pending items."}}
    if at == "create_event":
        st = max(0, turn + rng.randint(1, 8))
        en = st + rng.randint(1, 3)
        return {"action_type": at, "params": {"title": "Sync", "start_turn": st, "end_turn": en, "attendees": ["team@example.com"]}}
    if at == "draft_document":
        return {"action_type": at, "params": {"title": "Draft", "content": "Here is a draft.", "recipients": ["boss@example.com"]}}
    if at == "delegate":
        return {
            "action_type": at,
            "params": {"task_description": "Compare 2 options and summarize tradeoffs within budget and deadline.", "subtask_type": "research", "deadline_turn": turn + 6},
        }
    if at == "purchase_item":
        amt = float(rng.choice([50, 120, 220, 350]))
        return {"action_type": at, "params": {"item": "Supplies", "vendor": "VendorX", "amount": amt}}
    if at == "transfer_funds":
        amt = float(rng.choice([100, 300, 600, 1200]))
        return {"action_type": at, "params": {"amount": amt, "recipient": "VendorY", "memo": "Invoice payment"}}
    # book_travel
    depart = turn + rng.randint(2, 12)
    ret = depart + rng.randint(2, 8)
    cap = float(rng.choice([800, 1200, 1800]))
    return {"action_type": "book_travel", "params": {"traveler": "Boss", "destination": rng.choice(["Mumbai", "Delhi", "Bengaluru"]), "depart_turn": depart, "return_turn": ret, "budget_cap": cap}}


@dataclass
class EpisodeMetrics:
    episode_reward: float
    task_completion_rate: float
    boss_ask_rate: float
    budget_adherence: float
    adversary_success_rate: float
    rubric_scores: Dict[str, float]


def _extract_metrics(env: DelegationWorld, breakdown: Dict[str, Any], adversary_failures: int, adversary_injections: int) -> EpisodeMetrics:
    st = env.state
    assert st is not None

    rubrics = {r["name"]: float(r["score"]) for r in breakdown.get("rubrics", [])}
    total_decisions = max(1, int(st.decisions_total))
    boss_ask_rate = float(st.boss_interventions) / float(total_decisions)

    total_critical = max(1, int(st.initial_tasks_by_priority.get("critical", 0)))
    completed_critical = int(rubrics.get("task_completion", 0.0) * total_critical)  # approximate
    task_completion_rate = float(completed_critical) / float(total_critical)

    budget_adherence = 1.0 if st.budget_spent <= st.budget_limit else 0.0
    adv_rate = (float(adversary_failures) / float(adversary_injections)) if adversary_injections > 0 else 0.0

    return EpisodeMetrics(
        episode_reward=float(breakdown.get("reward", 0.0)),
        task_completion_rate=task_completion_rate,
        boss_ask_rate=boss_ask_rate,
        budget_adherence=budget_adherence,
        adversary_success_rate=adv_rate,
        rubric_scores=rubrics,
    )


def _run_episode(env: DelegationWorld, policy_fn, max_turns: int, rng: random.Random) -> EpisodeMetrics:
    env.reset(seed=rng.randint(0, 10_000))
    adversary_injections = 0
    adversary_failures = 0

    for _ in range(max_turns):
        st = env.state
        assert st is not None
        action = policy_fn(env, rng)
        _, _, done, info = env.step(action)

        if "adversary" in info:
            adversary_injections += 1
            if bool(st.last_curveball_caused_failure):
                adversary_failures += 1

        if done:
            break

    _, breakdown = env.get_episode_reward(partial=False)
    return _extract_metrics(env, breakdown, adversary_failures, adversary_injections)


# -----------------------------
# Baselines
# -----------------------------
def random_policy(env: DelegationWorld, rng: random.Random) -> Dict[str, Any]:
    st = env.state
    assert st is not None
    return _sample_reasonable_action(rng, st.current_turn)


def ask_always_policy(env: DelegationWorld, rng: random.Random) -> Dict[str, Any]:
    st = env.state
    assert st is not None
    # For irreversible-ish actions, always ask first.
    if st.current_turn % 3 == 0:
        return {"action_type": "ask_boss", "params": {"question": "Before I do anything irreversible or expensive, any specific constraints?"}}
    # Then do a productive action.
    return {"action_type": "send_email", "params": {"to": "ops@example.com", "subject": "Progress", "body": "Handling pending decisions now."}}


# -----------------------------
# Smoke-test training loop (dependency-free)
# -----------------------------
class SimpleTrainerPolicy:
    """
    A tiny 'learning' policy used in --smoke-test to generate improvement curves
    without requiring torch/trl. It tunes ask_prob toward the goldilocks zone.
    """

    def __init__(self):
        self.ask_prob = 0.02

    def __call__(self, env: DelegationWorld, rng: random.Random) -> Dict[str, Any]:
        st = env.state
        assert st is not None
        # ask sometimes, but not too much
        if rng.random() < self.ask_prob:
            return {"action_type": "ask_boss", "params": {"question": "Any approval needed for the next irreversible action?"}}
        # biased toward productive actions that resolve pending tasks
        return {"action_type": rng.choice(["send_email", "create_event", "draft_document", "delegate"]), "params": _sample_reasonable_action(rng, st.current_turn)["params"]}

    def update_from_episode(self, metrics: EpisodeMetrics) -> None:
        # Move ask_prob toward the [0.05, 0.20] band (hero metric).
        if metrics.boss_ask_rate < 0.05:
            self.ask_prob = min(0.25, self.ask_prob + 0.01)
        elif metrics.boss_ask_rate > 0.20:
            self.ask_prob = max(0.0, self.ask_prob - 0.01)


def _plot_curves(xs: List[int], series: Dict[str, List[float]], title: str, ylabel: str, out_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    for name, ys in series.items():
        plt.plot(xs, ys, label=name, linewidth=2)
    plt.title(title)
    plt.xlabel("training_step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_rubric_breakdown(xs: List[int], rubric_names: List[str], rubric_series: Dict[str, List[float]], out_path: str) -> None:
    plt.figure(figsize=(9, 4.5))
    bottom = [0.0 for _ in xs]
    for rn in rubric_names:
        ys = rubric_series[rn]
        plt.bar(xs, ys, bottom=bottom, label=rn)
        bottom = [b + y for b, y in zip(bottom, ys)]
    plt.title("Rubric breakdown over training")
    plt.xlabel("training_step")
    plt.ylabel("weighted rubric score (sum to 1)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_smoke_test(steps: int = 200, eval_every: int = 50, episode_turns: int = 60, seed: int = 0) -> None:
    _ensure_plots_dir()
    rng = random.Random(seed)

    env = DelegationWorld()
    policy = SimpleTrainerPolicy()

    xs: List[int] = []
    reward_trained: List[float] = []
    reward_random: List[float] = []
    reward_ask: List[float] = []

    autonomy_trained: List[float] = []
    adversary_trained: List[float] = []

    rubric_names = ["task_completion", "autonomy_calibration", "priority_alignment", "information_efficiency", "budget_adherence", "delegation_quality"]
    rubric_weighted_series: Dict[str, List[float]] = {k: [] for k in rubric_names}

    for step in range(1, steps + 1):
        ep = _run_episode(env, policy, max_turns=episode_turns, rng=rng)
        policy.update_from_episode(ep)

        if step % eval_every == 0:
            xs.append(step)

            ep_tr = _run_episode(env, policy, max_turns=episode_turns, rng=rng)
            ep_ra = _run_episode(env, random_policy, max_turns=episode_turns, rng=rng)
            ep_aa = _run_episode(env, ask_always_policy, max_turns=episode_turns, rng=rng)

            reward_trained.append(ep_tr.episode_reward)
            reward_random.append(ep_ra.episode_reward)
            reward_ask.append(ep_aa.episode_reward)

            autonomy_trained.append(ep_tr.boss_ask_rate)
            adversary_trained.append(ep_tr.adversary_success_rate)

            # Convert rubric scores into "weighted contribution" for stacked bar.
            # We use fixed weights from spec.
            weights = {
                "task_completion": 0.25,
                "autonomy_calibration": 0.20,
                "priority_alignment": 0.20,
                "information_efficiency": 0.15,
                "budget_adherence": 0.10,
                "delegation_quality": 0.10,
            }
            for rn in rubric_names:
                rubric_weighted_series[rn].append(weights[rn] * float(ep_tr.rubric_scores.get(rn, 0.0)))

    _plot_curves(
        xs,
        {"trained": reward_trained, "random": reward_random, "ask_always": reward_ask},
        title="Mean episode reward (higher is better)",
        ylabel="episode_reward",
        out_path=os.path.join(PLOTS_DIR, "reward_curve.png"),
    )
    _plot_curves(
        xs,
        {"trained": autonomy_trained},
        title="Autonomy calibration (boss ask rate)",
        ylabel="boss_ask_rate",
        out_path=os.path.join(PLOTS_DIR, "autonomy_curve.png"),
    )
    _plot_curves(
        xs,
        {"trained": adversary_trained},
        title="Adversary success rate (lower is better)",
        ylabel="adversary_success_rate",
        out_path=os.path.join(PLOTS_DIR, "adversary_curve.png"),
    )
    _plot_rubric_breakdown(
        xs,
        rubric_names,
        rubric_weighted_series,
        out_path=os.path.join(PLOTS_DIR, "rubric_breakdown.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run dependency-free smoke training and emit plots.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--episode-turns", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(steps=args.steps, eval_every=args.eval_every, episode_turns=args.episode_turns, seed=args.seed)
        print(f"ok: wrote plots to {PLOTS_DIR}/")
        return

    # Full GRPO training path (requires optional deps).
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from trl import GRPOTrainer  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Full training requires optional deps. Install with:\n"
            "  pip install -e '.[train]'\n"
            "Or run:\n"
            "  python training/train_grpo.py --smoke-test\n"
        ) from e

    raise NotImplementedError(
        "GRPO path not yet wired in this workspace. Use --smoke-test for now."
    )


if __name__ == "__main__":
    main()

