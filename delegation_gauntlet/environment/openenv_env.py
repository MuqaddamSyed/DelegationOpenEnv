from __future__ import annotations

from typing import Any, Dict, Tuple

from delegation_gauntlet.environment.world import DelegationWorld
from delegation_gauntlet.models import Action


def _resolve_openenv_base():
    """
    Locate the OpenEnv base class across known package layouts.

    The OpenEnv ecosystem ships with multiple package names / module paths:
      - `openenv.env.env.Env` (older local installs)
      - `openenv.environment.Environment` (some core builds)
      - `openenv_core.environment.Environment` (newer)

    If none are importable, we fall back to `object` so the Space still boots.
    """
    candidates = (
        ("openenv.env.env", "Env"),
        ("openenv.environment", "Environment"),
        ("openenv_core.environment", "Environment"),
        ("openenv.core.environment", "Environment"),
    )
    for module_path, attr in candidates:
        try:
            mod = __import__(module_path, fromlist=[attr])
            base = getattr(mod, attr, None)
            if base is not None:
                return base
        except Exception:
            continue
    return object


_BaseEnv = _resolve_openenv_base()


class DelegationOpenEnv(_BaseEnv):  # type: ignore[misc,valid-type]
    """
    Thin OpenEnv-compatible wrapper around `DelegationWorld`.
    Keeps simulation logic separate from the OpenEnv interface and tolerates
    different `openenv` package layouts at import time.
    """

    def __init__(self) -> None:
        try:
            super().__init__(  # type: ignore[misc]
                name="delegation-gauntlet",
                state_space=None,
                action_space=None,
                episode_max_length=200,
            )
        except TypeError:
            try:
                super().__init__()  # type: ignore[misc]
            except Exception:
                pass
        self.world = DelegationWorld()

    def reset(self, **kwargs: Any) -> str:
        return self.world.reset(
            seed=kwargs.get("seed"),
            scenario=kwargs.get("scenario"),
            boss=kwargs.get("boss"),
            adversarial_mode=kwargs.get("adversarial_mode"),
        )

    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        validated = Action.model_validate(action)
        return self.world.step(validated)

    def state(self) -> Dict[str, Any]:
        return self.world.get_state()
