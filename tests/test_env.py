import os
import sys

# Ensure repo root on path when running as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from delegation_gauntlet.environment.world import DelegationWorld


def test_env_reset_and_step_smoke():
    env = DelegationWorld()
    obs = env.reset(seed=123)
    assert "DELEGATION GAUNTLET" in obs

    for _ in range(5):
        obs, reward, done, info = env.step({"action_type": "do_nothing", "params": {}})
        assert isinstance(obs, str) and len(obs) > 10
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


if __name__ == "__main__":
    # Allows running without pytest installed:
    #   python tests/test_env.py
    test_env_reset_and_step_smoke()
    print("ok")

