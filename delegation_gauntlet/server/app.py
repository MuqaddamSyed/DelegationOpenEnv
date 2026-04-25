from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from delegation_gauntlet.environment.world import DelegationWorld
from delegation_gauntlet.models import Action, BossPersonality, ScenarioType


app = FastAPI(title="Delegation Gauntlet (OpenEnv)")

# Single-process in-memory world (sufficient for hackathon demo).
_WORLD = DelegationWorld()


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    scenario: Optional[ScenarioType] = None
    boss_personality: Optional[BossPersonality] = None
    adversarial_mode: Optional[bool] = None


class ResetResponse(BaseModel):
    observation: str


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    obs = _WORLD.reset(
        seed=req.seed,
        scenario=req.scenario,
        boss=req.boss_personality,
        adversarial_mode=req.adversarial_mode,
    )
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    obs, reward, done, info = _WORLD.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state() -> Dict[str, Any]:
    return _WORLD.get_state()

