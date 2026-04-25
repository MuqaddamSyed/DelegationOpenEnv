from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import httpx

from delegation_gauntlet.models import Action


class DelegationGauntletClient:
    """
    Typed client for the OpenEnv-style HTTP API:
      POST /reset -> observation (str)
      POST /step  -> {observation, reward, done, info}
      GET  /state -> json
      GET  /health -> {"status":"ok"}
    """

    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def health(self) -> Dict[str, Any]:
        return self._client.get("/health").json()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        scenario: Optional[str] = None,
        boss_personality: Optional[str] = None,
        adversarial_mode: Optional[bool] = None,
    ) -> str:
        payload: Dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        if scenario is not None:
            payload["scenario"] = scenario
        if boss_personality is not None:
            payload["boss_personality"] = boss_personality
        if adversarial_mode is not None:
            payload["adversarial_mode"] = adversarial_mode
        r = self._client.post("/reset", json=payload)
        r.raise_for_status()
        return r.json()["observation"]

    def step(self, action: Action | Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        a = Action.model_validate(action).model_dump()
        r = self._client.post("/step", json={"action": a})
        r.raise_for_status()
        data = r.json()
        return data["observation"], float(data["reward"]), bool(data["done"]), dict(data["info"])

    def state(self) -> Dict[str, Any]:
        r = self._client.get("/state")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()

