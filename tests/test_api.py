"""Test integration with Gymnasium API."""
import gymnasium
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import FlattenObservation

from tests.utils import StripNondeterministicInfo, get_all_registered_miniwob_envs


class TestGymAPI:
    """Test integration with Gymnasium API."""

    @pytest.fixture(params=get_all_registered_miniwob_envs())
    def env(self, request):
        """Yield an environment for the task."""
        env = gymnasium.make(request.param)
        yield env
        env.close()

    def test_gym_api(self, env):
        """Check that the environment follows Gym API."""
        # Run check_env to check space containment, determinism, etc.
        # We use wrapper to strip wall-clock info key "elapsed" which broke determinism checks.
        check_env(StripNondeterministicInfo(env.unwrapped), skip_render_check=True)
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
