from __future__ import annotations

from typing import Any, Dict

from openenv.env.env import Env

from delegation_gauntlet.environment.world import DelegationWorld
from delegation_gauntlet.models import Action


class DelegationOpenEnv(Env):
    """
    Thin OpenEnv-compatible wrapper around DelegationWorld.
    Keeps a clear separation between simulation logic and OpenEnv interface.
    """

    def __init__(self):
        super().__init__(name="delegation-gauntlet", state_space=None, action_space=None, episode_max_length=200)
        self.world = DelegationWorld()

    def reset(self, **kwargs: Any) -> str:
        return self.world.reset(
            seed=kwargs.get("seed"),
            scenario=kwargs.get("scenario"),
            boss=kwargs.get("boss"),
            adversarial_mode=kwargs.get("adversarial_mode"),
        )

    def step(self, action: Dict[str, Any]) -> tuple[str, float, bool, Dict[str, Any]]:
        validated = Action.model_validate(action)
        return self.world.step(validated)

    def state(self) -> Dict[str, Any]:
        return self.world.get_state()

