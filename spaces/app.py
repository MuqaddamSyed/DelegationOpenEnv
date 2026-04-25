from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, Generator, List, Optional, Tuple

# Ensure repo root on path when running as a script:
#   python spaces/app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from delegation_gauntlet.environment.world import DelegationWorld
from delegation_gauntlet.models import BossPersonality, ScenarioType


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "plots")


def _format_rubrics(breakdown: Dict[str, Any]) -> Dict[str, float]:
    rubrics = breakdown.get("rubrics", [])
    out: Dict[str, float] = {}
    for r in rubrics:
        out[str(r.get("name"))] = float(r.get("score", 0.0))
    return out


def _policy_for_demo(env: DelegationWorld, rng: random.Random) -> Dict[str, Any]:
    """
    A deterministic-ish demo policy:
    - Ask boss occasionally to stay near goldilocks band
    - Prefer productive actions that resolve pending tasks
    """
    st = env.state
    assert st is not None

    boss_ask_rate = (st.boss_interventions / max(1, st.decisions_total)) if st.decisions_total > 0 else 0.0
    ask_prob = 0.10 if boss_ask_rate < 0.15 else 0.04

    if rng.random() < ask_prob:
        return {"action_type": "ask_boss", "params": {"question": "Any approvals or constraints for irreversible actions?"}}

    action_type = rng.choice(["send_email", "create_event", "draft_document", "delegate", "do_nothing"])
    if action_type == "send_email":
        return {"action_type": "send_email", "params": {"to": "ops@example.com", "subject": "Update", "body": "Progress update: working through pending decisions."}}
    if action_type == "create_event":
        start_turn = int(st.current_turn + rng.randint(1, 6))
        end_turn = int(start_turn + rng.randint(1, 3))
        return {"action_type": "create_event", "params": {"title": "Coordination sync", "start_turn": start_turn, "end_turn": end_turn, "attendees": ["team@example.com"], "location": "Zoom"}}
    if action_type == "draft_document":
        return {"action_type": "draft_document", "params": {"title": "Draft", "content": "Draft content for review.", "recipients": ["boss@example.com"]}}
    if action_type == "delegate":
        return {
            "action_type": "delegate",
            "params": {
                "task_description": "Research options and summarize tradeoffs within budget and deadline. Include 2 alternatives.",
                "subtask_type": "research",
                "deadline_turn": int(st.current_turn + 6),
            },
        }
    return {"action_type": "do_nothing", "params": {}}


def run_episode(
    scenario: str,
    boss_personality: str,
    adversarial_mode: bool,
    max_turns: int,
    seed: int,
) -> Generator[Tuple[str, float, float, float, float, float, float], None, None]:
    """
    Yields incremental UI updates:
      (log_text, task_completion, autonomy_calibration, priority_alignment,
       information_efficiency, budget_adherence, delegation_quality)
    """
    env = DelegationWorld()
    rng = random.Random(int(seed))

    sc = ScenarioType[scenario]
    bp = BossPersonality[boss_personality]
    env.reset(seed=int(seed), scenario=sc, boss=bp, adversarial_mode=bool(adversarial_mode))

    log_lines: List[str] = []
    log_lines.append(f"Scenario={scenario} | Boss={boss_personality} | Adversarial={'ON' if adversarial_mode else 'OFF'} | Seed={seed}")
    log_lines.append("")

    # Initialize rubric bars
    _, breakdown0 = env.get_episode_reward(partial=True)
    rub0 = _format_rubrics(breakdown0)
    yield ("\n".join(log_lines), rub0.get("task_completion", 0.0), rub0.get("autonomy_calibration", 0.0), rub0.get("priority_alignment", 0.0), rub0.get("information_efficiency", 0.0), rub0.get("budget_adherence", 0.0), rub0.get("delegation_quality", 0.0))

    for _ in range(int(max_turns)):
        st = env.state
        assert st is not None
        action = _policy_for_demo(env, rng)
        _, _, done, info = env.step(action)

        turn = info.get("turn", st.current_turn)
        act_desc = f"{action['action_type']}({', '.join([f'{k}={v}' for k,v in action.get('params', {}).items() if k != 'read_message_ids'])})"
        res = info.get("result", {})
        ok = "✓" if res.get("success") else "✗"
        log_lines.append(f"Turn {turn:03d} | Action: {act_desc} | Result: {ok} {res.get('message','')}")

        if "adversary" in info:
            adv = info["adversary"]
            log_lines.append(f"Turn {turn:03d} | ADVERSARY: Injected {adv.get('injected')}")

        # Update rubric bars from partial breakdown
        _, breakdown = env.get_episode_reward(partial=True)
        rub = _format_rubrics(breakdown)

        yield (
            "\n".join(log_lines[-250:]),
            rub.get("task_completion", 0.0),
            rub.get("autonomy_calibration", 0.0),
            rub.get("priority_alignment", 0.0),
            rub.get("information_efficiency", 0.0),
            rub.get("budget_adherence", 0.0),
            rub.get("delegation_quality", 0.0),
        )

        if done:
            break

    # Final rubric snapshot
    _, breakdown_f = env.get_episode_reward(partial=False)
    rubf = _format_rubrics(breakdown_f)
    log_lines.append("")
    log_lines.append(f"Final episode reward: {breakdown_f.get('reward', 0.0):.3f} (raw={breakdown_f.get('raw', 0.0):.3f})")
    yield (
        "\n".join(log_lines[-400:]),
        rubf.get("task_completion", 0.0),
        rubf.get("autonomy_calibration", 0.0),
        rubf.get("priority_alignment", 0.0),
        rubf.get("information_efficiency", 0.0),
        rubf.get("budget_adherence", 0.0),
        rubf.get("delegation_quality", 0.0),
    )


def _plot_path(filename: str) -> str:
    return os.path.abspath(os.path.join(PLOTS_DIR, filename))


def build_demo():
    # Import gradio lazily so non-Spaces environments with mismatched deps
    # can still import this module (HF Spaces will have correct deps).
    import gradio as gr

    with gr.Blocks(title="Delegation Gauntlet") as demo:
        gr.Markdown(
            """
## Delegation Gauntlet

Production-grade agent hardening infrastructure: a 3-week executive-assistant simulation with **budget authority**, **simulated tools**, and a **deterministic adversary** that injects curveballs to provoke failures.
"""
        )

        with gr.Tab("Live Episode"):
            with gr.Row():
                scenario = gr.Dropdown(choices=[s.name for s in ScenarioType], value=ScenarioType.CONFERENCE_PLANNING.name, label="Scenario")
                boss = gr.Dropdown(choices=[b.name for b in BossPersonality], value=BossPersonality.MICROMANAGER.name, label="Boss personality")
                adversarial = gr.Checkbox(value=True, label="Adversarial mode")
            with gr.Row():
                max_turns = gr.Slider(10, 200, value=60, step=1, label="Max turns to run")
                seed = gr.Number(value=0, precision=0, label="Seed")
                run_btn = gr.Button("Run Episode", variant="primary")

            log = gr.Textbox(label="Live log", lines=18, interactive=False)

            gr.Markdown("### Final rubric breakdown (0 to 1)")
            task_completion = gr.Slider(0, 1, value=0, step=0.01, label="TaskCompletionRubric", interactive=False)
            autonomy = gr.Slider(0, 1, value=0, step=0.01, label="AutonomyCalibrationRubric (Goldilocks zone)", interactive=False)
            priority = gr.Slider(0, 1, value=0, step=0.01, label="PriorityAlignmentRubric", interactive=False)
            info_eff = gr.Slider(0, 1, value=0, step=0.01, label="InformationEfficiencyRubric", interactive=False)
            budget = gr.Slider(0, 1, value=0, step=0.01, label="BudgetAdherenceRubric", interactive=False)
            delegation = gr.Slider(0, 1, value=0, step=0.01, label="DelegationQualityRubric", interactive=False)

            run_btn.click(
                fn=run_episode,
                inputs=[scenario, boss, adversarial, max_turns, seed],
                outputs=[log, task_completion, autonomy, priority, info_eff, budget, delegation],
            )

        with gr.Tab("Training Results"):
            gr.Markdown("### Plots generated by `python training/train_grpo.py --smoke-test`")
            with gr.Row():
                gr.Image(_plot_path("reward_curve.png"), label="reward_curve.png", show_label=True)
                gr.Image(_plot_path("autonomy_curve.png"), label="autonomy_curve.png (hero)", show_label=True)
            with gr.Row():
                gr.Image(_plot_path("adversary_curve.png"), label="adversary_curve.png", show_label=True)
                gr.Image(_plot_path("rubric_breakdown.png"), label="rubric_breakdown.png", show_label=True)

            gr.Markdown(
                """
**What judges should notice**

- **Autonomy curve**: the agent learns a stable **boss_ask_rate** inside the \([0.05, 0.20]\) goldilocks band.
- **Adversary success**: should trend down as the policy becomes robust to curveballs.
"""
            )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()

