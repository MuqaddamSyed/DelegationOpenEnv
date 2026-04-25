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
    judge_mode: bool,
    max_turns: int,
    seed: int,
) -> Generator[Tuple[str, float, float, float, float, float, float, float, float, float, float], None, None]:
    """
    Yields incremental UI updates:
      (log_text, task_completion, autonomy_calibration, priority_alignment,
       information_efficiency, budget_adherence, delegation_quality)
    """
    env = DelegationWorld()
    rng = random.Random(int(seed))

    if judge_mode:
        # Deterministic stress test for reviewers.
        scenario = ScenarioType.CRISIS_MANAGEMENT.name
        boss_personality = BossPersonality.PASSIVE_AGGRESSIVE.name
        adversarial_mode = True
        seed = 4242
        max_turns = min(int(max_turns), 70)

    sc = ScenarioType[scenario]
    bp = BossPersonality[boss_personality]
    env.reset(seed=int(seed), scenario=sc, boss=bp, adversarial_mode=bool(adversarial_mode))

    log_lines: List[str] = []
    log_lines.append(f"Scenario={scenario} | Boss={boss_personality} | Adversarial={'ON' if adversarial_mode else 'OFF'} | Seed={seed}")
    log_lines.append("")

    # Initialize rubric bars
    _, breakdown0 = env.get_episode_reward(partial=True)
    rub0 = _format_rubrics(breakdown0)
    st0 = env.state
    assert st0 is not None
    yield (
        "\n".join(log_lines),
        rub0.get("task_completion", 0.0),
        rub0.get("autonomy_calibration", 0.0),
        rub0.get("priority_alignment", 0.0),
        rub0.get("information_efficiency", 0.0),
        rub0.get("budget_adherence", 0.0),
        rub0.get("delegation_quality", 0.0),
        float(breakdown0.get("reward", 0.0)),
        float(st0.boss_interventions) / float(max(1, st0.decisions_total)),
        float(st0.budget_spent) / float(max(1.0, st0.budget_limit)),
        float(st0.irreversible_without_approval),
    )

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
            float(breakdown.get("reward", 0.0)),
            float(st.boss_interventions) / float(max(1, st.decisions_total)),
            float(st.budget_spent) / float(max(1.0, st.budget_limit)),
            float(st.irreversible_without_approval),
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
        float(breakdown_f.get("reward", 0.0)),
        float(env.state.boss_interventions) / float(max(1, env.state.decisions_total)) if env.state else 0.0,
        float(env.state.budget_spent) / float(max(1.0, env.state.budget_limit)) if env.state else 0.0,
        float(env.state.irreversible_without_approval) if env.state else 0.0,
    )


def _plot_path(filename: str) -> str:
    return os.path.abspath(os.path.join(PLOTS_DIR, filename))


def build_demo():
    # Import gradio lazily so non-Spaces environments with mismatched deps
    # can still import this module (HF Spaces will have correct deps).
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="pink",
        neutral_hue="slate",
        radius_size="md",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    css = """
    .hero {
        border: 1px solid rgba(99, 102, 241, 0.22);
        background: linear-gradient(135deg, rgba(79,70,229,0.13), rgba(236,72,153,0.10));
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .muted {
        opacity: 0.9;
        font-size: 0.95rem;
    }
    """

    with gr.Blocks(title="Delegation Gauntlet", theme=theme, css=css) as demo:
        gr.Markdown(
            """
<div class="hero">
  <h2 style="margin:0 0 8px 0;">Delegation Gauntlet</h2>
  <div class="muted">
    OpenEnv hardening environment for tool-using agents: dynamic inbox, budget constraints, deterministic adversary, and explicit autonomy calibration.
  </div>
</div>
"""
        )

        with gr.Tab("Live Episode"):
            gr.Markdown("Configure a run, then stream a full episode with adversarial interventions and rubric progress.")
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    scenario = gr.Dropdown(
                        choices=[s.name for s in ScenarioType],
                        value=ScenarioType.CONFERENCE_PLANNING.name,
                        label="Scenario",
                    )
                    boss = gr.Dropdown(
                        choices=[b.name for b in BossPersonality],
                        value=BossPersonality.MICROMANAGER.name,
                        label="Boss personality",
                    )
                    adversarial = gr.Checkbox(value=True, label="Adversarial mode")
                    judge_mode = gr.Checkbox(value=False, label="Judge mode (deterministic stress test)")
                    max_turns = gr.Slider(10, 200, value=60, step=1, label="Max turns")
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    with gr.Row():
                        run_btn = gr.Button("Run Episode", variant="primary")
                        clear_btn = gr.Button("Clear Log")
                    with gr.Row():
                        preset_quick = gr.Button("Preset: Quick Demo")
                        preset_judge = gr.Button("Preset: Judge Run")
                        preset_stress = gr.Button("Preset: Stress Test")
                    with gr.Accordion("What Judge Mode does", open=False):
                        gr.Markdown(
                            """
- Forces `CRISIS_MANAGEMENT`
- Uses `PASSIVE_AGGRESSIVE` boss
- Enables adversarial injections
- Fixes seed for reproducible judging
"""
                        )

                with gr.Column(scale=2):
                    log = gr.Textbox(label="Live Log", lines=22, interactive=False, autoscroll=True)

            gr.Markdown("### Rubric Progress (0 to 1)")
            with gr.Row():
                with gr.Column():
                    task_completion = gr.Slider(0, 1, value=0, step=0.01, label="Task completion", interactive=False)
                    autonomy = gr.Slider(0, 1, value=0, step=0.01, label="Autonomy calibration", interactive=False)
                    priority = gr.Slider(0, 1, value=0, step=0.01, label="Priority alignment", interactive=False)
                with gr.Column():
                    info_eff = gr.Slider(0, 1, value=0, step=0.01, label="Information efficiency", interactive=False)
                    budget = gr.Slider(0, 1, value=0, step=0.01, label="Budget adherence", interactive=False)
                    delegation = gr.Slider(0, 1, value=0, step=0.01, label="Delegation quality", interactive=False)
            with gr.Row():
                kpi_reward = gr.Number(value=0.0, label="Episode reward", interactive=False, precision=4)
                kpi_ask_rate = gr.Number(value=0.0, label="Boss ask rate", interactive=False, precision=4)
                kpi_budget = gr.Number(value=0.0, label="Budget used ratio", interactive=False, precision=4)
                kpi_unapproved = gr.Number(value=0.0, label="Unapproved irreversible actions", interactive=False, precision=0)

            run_btn.click(
                fn=run_episode,
                inputs=[scenario, boss, adversarial, judge_mode, max_turns, seed],
                outputs=[log, task_completion, autonomy, priority, info_eff, budget, delegation, kpi_reward, kpi_ask_rate, kpi_budget, kpi_unapproved],
            )
            clear_btn.click(
                fn=lambda: ("", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                inputs=[],
                outputs=[log, task_completion, autonomy, priority, info_eff, budget, delegation, kpi_reward, kpi_ask_rate, kpi_budget, kpi_unapproved],
            )
            preset_quick.click(
                fn=lambda: (ScenarioType.CONFERENCE_PLANNING.name, BossPersonality.MICROMANAGER.name, True, False, 40, 7),
                inputs=[],
                outputs=[scenario, boss, adversarial, judge_mode, max_turns, seed],
            )
            preset_judge.click(
                fn=lambda: (ScenarioType.CRISIS_MANAGEMENT.name, BossPersonality.PASSIVE_AGGRESSIVE.name, True, True, 70, 4242),
                inputs=[],
                outputs=[scenario, boss, adversarial, judge_mode, max_turns, seed],
            )
            preset_stress.click(
                fn=lambda: (ScenarioType.BOARD_REVIEW.name, BossPersonality.HANDS_OFF.name, True, False, 120, 101),
                inputs=[],
                outputs=[scenario, boss, adversarial, judge_mode, max_turns, seed],
            )

        with gr.Tab("Training Results"):
            gr.Markdown("Plots generated by `training/train_grpo.py`")
            with gr.Row():
                gr.Image(_plot_path("autonomy_curve.png"), label="Autonomy curve (hero)", show_label=True)
                gr.Image(_plot_path("reward_curve.png"), label="Reward curve", show_label=True)
            with gr.Row():
                gr.Image(_plot_path("adversary_curve.png"), label="Adversary success curve", show_label=True)
                gr.Image(_plot_path("rubric_breakdown.png"), label="Rubric before vs after", show_label=True)
            with gr.Row():
                gr.Image(_plot_path("adversary_weights.png"), label="Adversary strategy weights", show_label=True)
                gr.Image(_plot_path("before_after_summary.png"), label="Overall before/after summary", show_label=True)

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
    # Hugging Face Spaces provides PORT. In Docker/Spaces we must bind 0.0.0.0.
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    demo.launch(server_name=host, server_port=port)

