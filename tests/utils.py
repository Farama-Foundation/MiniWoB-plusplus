"""Test utilities."""
from collections.abc import Iterable
from typing import Any

import gymnasium
from gymnasium.utils import RecordConstructorArgs


NONDETERMINISTIC_INFO_KEYS = ("elapsed",)


class StripNondeterministicInfo(gymnasium.Wrapper, RecordConstructorArgs):
    """Wrapper that removes inherently non-deterministic keys from info dicts.

    Wall-clock fields like ``elapsed`` break gymnasium's determinism checks
    because they vary across runs even when the environment is seeded
    identically.  This wrapper is intended for **test use only**.
    """

    def __init__(self, env: gymnasium.Env):
        RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        for key in NONDETERMINISTIC_INFO_KEYS:
            info.pop(key, None)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in NONDETERMINISTIC_INFO_KEYS:
            info.pop(key, None)
        return obs, reward, terminated, truncated, info


def get_all_registered_miniwob_envs() -> Iterable[str]:
    """Return the name of all registered MiniWoB environments."""
    envs = []
    for env_id, env_spec in gymnasium.registry.items():
        if env_spec.namespace == "miniwob":
            envs.append(env_id)
    return sorted(envs)
