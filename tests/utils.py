"""Test utilities."""
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium.utils import RecordConstructorArgs


NONDETERMINISTIC_INFO_KEYS = ("elapsed",)
POSITION_DECIMALS = 1
TAMPERED_FLAG_INDEX = 1
SCREENSHOT_QUANTIZE_FACTOR = 8.0


def get_all_registered_miniwob_envs() -> Iterable[str]:
    """Return the name of all registered MiniWoB environments."""
    envs = []
    for env_id, env_spec in gymnasium.registry.items():
        if env_spec.namespace == "miniwob":
            envs.append(env_id)
    return sorted(envs)


class StripNondeterministicInfo(gymnasium.Wrapper, RecordConstructorArgs):
    """Wrapper that normalizes non-deterministic observation and info fields.

    Handles framework-level sources of non-determinism:

    1. Wall-clock fields like ``elapsed`` in the info dict.
    2. Monotonically-increasing ``ref`` codes in DOM elements.
    3. Sub-pixel position floats from ``getBoundingClientRect()``.
    4. Timing-dependent ``tampered`` flag in DOM element flags.
    5. Anti-aliasing / sub-pixel rendering jitter in screenshots.

    This wrapper is intended for **test use only**.
    """

    def __init__(self, env: gymnasium.Env, data_collection: bool = False):
        RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)
        self.data_collection = data_collection
        self.observations: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = _normalize_obs(obs)
        if self.data_collection:
            self.observations.append(
                {
                    "method": "reset",
                    "seed": seed,
                    "obs": obs,
                    "info": {
                        k: v
                        for k, v in info.items()
                        if k not in NONDETERMINISTIC_INFO_KEYS
                    },
                }
            )
        for key in NONDETERMINISTIC_INFO_KEYS:
            info.pop(key, None)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = _normalize_obs(obs)
        if self.data_collection:
            self.observations.append(
                {
                    "method": "step",
                    "action": action,
                    "obs": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": {
                        k: v
                        for k, v in info.items()
                        if k not in NONDETERMINISTIC_INFO_KEYS
                    },
                }
            )
        for key in NONDETERMINISTIC_INFO_KEYS:
            info.pop(key, None)
        return obs, reward, terminated, truncated, info

    def dump(self, path: str, **extra):
        """Dump recorded observations to a pickle file.

        Args:
            path: File path to write the pickle to.
            **extra: Additional metadata to include in the dump.
        """

        data = {
            "env_id": self.spec.id if self.spec else str(self.env),
            "observations": self.observations,
            **extra,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=5)


def _normalize_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Normalize non-deterministic fields in a MiniWoB observation.

    - Re-indexes ``ref`` / ``parent`` codes so they always start from 0.
    - Rounds position floats (``left``, ``top``, ``width``, ``height``)
      to avoid sub-pixel rendering drift.
    - Zeros the ``tampered`` flag which is timing-dependent (chrome's issue).
    - Rounds screenshot pixels to the nearest ``SCREENSHOT_QUANTIZE_FACTOR``
      to absorb anti-aliasing jitter.
    """
    obs = dict(obs)

    dom_elements = obs.get("dom_elements")
    if dom_elements:
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

            if "flags" in elem:
                flags = elem["flags"].copy()
                flags[TAMPERED_FLAG_INDEX] = 0
                elem["flags"] = flags

            normalized.append(elem)

        obs["dom_elements"] = tuple(normalized)

    screenshot = obs.get("screenshot")
    if screenshot is not None and isinstance(screenshot, np.ndarray):
        q_factor = SCREENSHOT_QUANTIZE_FACTOR
        quantized = np.floor(screenshot.astype(np.float32) / q_factor + 0.5) * q_factor
        obs["screenshot"] = np.clip(quantized, 0, 255).astype(np.uint8)

    return obs
