"""Test integration with Gymnasium API."""
import gymnasium
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import FlattenObservation
from selenium.common.exceptions import (
    InvalidSessionIdException,
    JavascriptException,
    MoveTargetOutOfBoundsException,
    SessionNotCreatedException,
)
from urllib3.exceptions import ReadTimeoutError
from urllib3.exceptions import TimeoutError as TimeoutError_urllib3

from miniwob.environment import MiniWoBEnvironment
from tests.utils import StripNondeterministicInfo, get_all_registered_miniwob_envs


RETRY_EXCEPTION = (
    # Inevitable with random movement
    MoveTargetOutOfBoundsException,
    # Some environment have bugs
    JavascriptException,
    InvalidSessionIdException,
    # Selenium / browser issue
    SessionNotCreatedException,
    ReadTimeoutError,
    TimeoutError_urllib3,
)


class TestGymAPI:
    """Test integration with Gymnasium API."""

    @pytest.fixture(params=get_all_registered_miniwob_envs())
    def env(self, request):
        """Yield an environment for the task."""
        env = None
        for i in range(1, 4):
            try:
                env = gymnasium.make(
                    request.param, wait_ms=150
                )  # allow browser to finish certain works
                break
            except RETRY_EXCEPTION as e:
                print(f"\033[31mMake {request.param} attempt {i} failed:\033[0m {e}")
        assert env
        yield env
        env.close()

    def test_gym_api(self, env):
        """Check that the environment follows Gym API."""
        # Run check_env to check space containment, determinism, etc.
        for i in range(1, 4):
            try:
                # We use wrapper to normalize DOM & screenshot obs, avoiding benign sources of nondeterminism.
                check_env(
                    StripNondeterministicInfo(env.unwrapped), skip_render_check=True
                )
                break
            except RETRY_EXCEPTION as e:
                print(f"\033[31m{env.unwrapped.spec.id} attempt {i} failed:\033[0m {e}")
            except AssertionError:
                # Increment wait_ms, attempt to save determinism
                unwrapped_env: MiniWoBEnvironment = env.unwrapped
                unwrapped_env.instance_kwargs["wait_ms"] += 100.0
                unwrapped_env.instance.wait_ms += 100.0
                print(
                    f"\033[31m{getattr(unwrapped_env.spec, 'id')} wait_ms + 100\033[0m"
                )
        # Check the spaces and flattened spaces.
        assert isinstance(env.observation_space, spaces.Dict)
        assert set(env.observation_space) == {
            "utterance",
            "dom_elements",
            "screenshot",
            "fields",
        }
        # dom_elements is a Sequence space and cannot be flattened.
        # But each element in the Sequence can be flattened.
        env = FlattenObservation(env)
        assert isinstance(env.observation_space, spaces.Dict)
        assert isinstance(env.observation_space["utterance"], spaces.Box)
        assert isinstance(env.observation_space["dom_elements"], spaces.Sequence)
        assert isinstance(env.observation_space["screenshot"], spaces.Box)
        assert isinstance(env.observation_space["fields"], spaces.Sequence)
