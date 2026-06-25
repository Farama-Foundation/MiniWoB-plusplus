"""Test utilities."""
from collections.abc import Iterable
from typing import Any

import gymnasium
import numpy as np
from gymnasium.utils import RecordConstructorArgs


NONDETERMINISTIC_INFO_KEYS = ("elapsed",)

POSITION_DECIMALS = 1


def _normalize_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Normalize non-deterministic fields in a MiniWoB observation.

    - Re-indexes ``ref`` / ``parent`` codes so they always start from 0.
    - Rounds position floats (``left``, ``top``, ``width``, ``height``)
      to avoid sub-pixel rendering drift.
    """
    dom_elements = obs.get("dom_elements")
    if not dom_elements:
        return obs

    ref_map: dict[int, int] = {0: 0}
    normalized = []
    for elem in dom_elements:
        elem = dict(elem)
        old_ref = elem["ref"]
        if old_ref not in ref_map:
            ref_map[old_ref] = len(ref_map)
        elem["ref"] = ref_map[old_ref]

        old_parent = elem["parent"]
        if old_parent not in ref_map:
            ref_map[old_parent] = len(ref_map)
        elem["parent"] = ref_map[old_parent]

        for key in ("left", "top", "width", "height"):
            if key in elem:
                elem[key] = np.round(elem[key], decimals=POSITION_DECIMALS)

        normalized.append(elem)

    obs = dict(obs)
    obs["dom_elements"] = tuple(normalized)
    return obs


class StripNondeterministicInfo(gymnasium.Wrapper, RecordConstructorArgs):
    """Wrapper that normalizes non-deterministic observation and info fields.

    Handles two framework-level sources of non-determinism:

    1. Wall-clock fields like ``elapsed`` in the info dict.
    2. Monotonically-increasing ``ref`` codes and sub-pixel position floats
       from ``getBoundingClientRect()`` in observations.

    This wrapper is intended for **test use only**.
    """

    def __init__(self, env: gymnasium.Env):
        RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = _normalize_obs(obs)
        for key in NONDETERMINISTIC_INFO_KEYS:
            info.pop(key, None)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = _normalize_obs(obs)
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
