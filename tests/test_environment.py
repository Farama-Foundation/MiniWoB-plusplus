"""Test environment methods."""
import functools
import time

import gymnasium
import numpy as np
import pytest

from miniwob.action import ActionTypes
from miniwob.fields import field_lookup
from miniwob.reward import (
    get_binary_reward,
    get_original_reward,
    get_raw_reward,
    get_thresholded_reward,
)


class MiniWoBTester:
    """Base class for testing on a single task."""

    # Subclasses should set this field
    ENV_NAME = ""

    @pytest.fixture
    def env(self):
        """Yield an environment for the task."""
        env = gymnasium.make(self.ENV_NAME)
        yield env
        env.close()

    ################################
    # Helpers

    def create_click_element_action(self, env, element):
        """Create an action that clicks in the specified element."""
        return env.unwrapped.create_action(
            ActionTypes.CLICK_ELEMENT, ref=element["ref"]
        )

    def create_click_button_action(self, env, obs, button_text):
        """Create an action that clicks on the button with the specified text."""
        for element in obs["dom_elements"]:
            if element["tag"] == "button" and element["text"] == button_text:
                return self.create_click_element_action(env, element)
        assert False, f"{button_text} button not found"


class TestMiniWoBEnvironment(MiniWoBTester):
    """Tests for basic environment functions."""

    ENV_NAME = "miniwob/click-test-v1"

    ################################
    # Tests

    def test_do_nothing(self, env):
        """Test the ability to start an instance for the click-test task."""
        obs, info = env.reset()
        assert obs["utterance"] == "Click the button."
        assert any(element["tag"] == "button" for element in obs["dom_elements"])

    def test_run(self, env):
        """Test reset() and step()."""
        obs, info = env.reset()
        assert obs["utterance"] == "Click the button."
        # Test empty action
        obs, reward, terminated, truncated, info = env.step(None)
        assert obs["utterance"] == "Click the button."
        assert reward == 0
        assert terminated is False
        assert truncated is False
        # Test clicking
        action = self.create_click_button_action(env, obs, "Click Me!")
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward > 0
        assert terminated is True
        assert truncated is False
        # Test reset
        obs, info = env.reset()
        assert obs["utterance"] == "Click the button."
        # Test clicking again
        action = self.create_click_button_action(env, obs, "Click Me!")
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward > 0
        assert terminated is True
        assert truncated is False

    def test_timeout(self, env):
        """Test environment timeout."""
        obs, info = env.reset()
        assert obs["utterance"] == "Click the button."
        # Wait for timeout
        time.sleep(12)
        obs, reward, terminated, truncated, info = env.step(None)
        assert reward < 0
        assert terminated is True
        assert truncated is False
        # Start again
        obs, info = env.reset()
        assert obs["utterance"] == "Click the button."
        obs, reward, terminated, truncated, info = env.step(None)
        assert reward == 0
        assert terminated is False
        assert truncated is False

    def test_speed(self, env):
        """Test the processing speed for step()."""
        env.reset()

        elapsed = []
        num_steps = 50

        for i in range(1, num_steps + 1):
            print("Iteration", i, "/", num_steps)

            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(None)
            elapsed.append(time.time() - start_time)

            if terminated or truncated:
                env.reset()

        mean = sum(elapsed) / len(elapsed)
        variance = sum((duration - mean) ** 2 for duration in elapsed) / len(elapsed)

        print("Average time:", mean)
        print("Variance:", variance)
        assert mean < 1.0  # 1 second per step is just too absurd

    def test_attention(self, env):
        """Test that visualize_attention() does not crash."""
        env.reset()
        attention = np.random.rand(20, 20) * 0.02
        env.unwrapped.visualize_attention(attention)
        time.sleep(1)
        env.unwrapped.visualize_attention(None)
        time.sleep(1)

    def test_screenshot(self, env):
        """Test the screenshot."""
        obs, info = env.reset()
        assert obs["screenshot"].shape == (210, 160, 3)
        # Upper-left should be the instruction (yellow).
        color_diff = obs["screenshot"][0, 0] - np.array([255.0, 255.0, 0.0])
        for i in range(3):
            assert abs(color_diff[i]) < 5.0
        # Lower-right should be the background (white).
        color_diff = obs["screenshot"][-1, -1] - np.array([255.0, 255.0, 255.0])
        for i in range(3):
            assert abs(color_diff[i]) < 5.0
        # Now click the button to complete the task.
        action = self.create_click_button_action(env, obs, "Click Me!")
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        # The screenshot should be all black
        np.testing.assert_allclose(obs["screenshot"], 0.0)


################################################


class TestMiniWoBSeed(MiniWoBTester):
    """Tests for seed determinism."""

    ENV_NAME = "miniwob/click-button-v1"

    def test_seed(self, env):
        """Test whether the same seed gives the same result."""
        obs_1, info_1 = env.reset(seed=31416)
        obs_2, info_2 = env.reset(seed=227)
        obs_3, info_3 = env.reset(seed=227)
        obs_4, info_4 = env.reset(seed=31416)
        # Check that everything is the same for the same seed
        assert obs_1["utterance"] == obs_4["utterance"]
        assert obs_2["utterance"] == obs_3["utterance"]
        ref_to_text_1 = {x["ref"]: x["text"] for x in obs_1["dom_elements"]}
        ref_to_text_2 = {x["ref"]: x["text"] for x in obs_2["dom_elements"]}
        ref_to_text_3 = {x["ref"]: x["text"] for x in obs_3["dom_elements"]}
        ref_to_text_4 = {x["ref"]: x["text"] for x in obs_4["dom_elements"]}
        assert ref_to_text_1 == ref_to_text_4
        assert ref_to_text_2 == ref_to_text_3
        assert ref_to_text_1 != ref_to_text_2
        # Compute the correct action from obs 1
        # and apply it on obs 4 (same seed)
        action = self.create_click_button_action(
            env, obs_1, field_lookup(obs_1["fields"], "target")
        )
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward > 0
        assert terminated is True


class TestMiniWoBMode(MiniWoBTester):
    """Tests for the data mode (available in some tasks)."""

    ENV_NAME = "miniwob/click-test-transfer-v1"

    def test_mode(self, env):
        """Test if setting the mode works.

        - mode = 'train': click on button ONE
        - mode = 'test':  click on button TWO
        """
        # Training time
        obs, info = env.reset()
        assert obs["utterance"] == "Click button ONE."
        action = self.create_click_button_action(env, obs, "ONE")
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward > 0
        obs, info = env.reset()
        assert obs["utterance"] == "Click button ONE."
        action = self.create_click_button_action(env, obs, "TWO")
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward < 0
        # Test time
        env.unwrapped.set_data_mode("test")
        obs, info = env.reset()
        assert obs["utterance"] == "Click button TWO."
        action = self.create_click_button_action(env, obs, "ONE")
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward < 0
        # Test time again; mode should be persistent
        obs, info = env.reset()
        assert obs["utterance"] == "Click button TWO."
        action = self.create_click_button_action(env, obs, "TWO")
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward > 0
        # Training time again: set mode with reset()
        obs, info = env.reset(options={"data_mode": "train"})
        assert obs["utterance"] == "Click button ONE."
        action = self.create_click_button_action(env, obs, "ONE")
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward > 0
        # Training time again; mode should be persistent
        obs, info = env.reset()
        assert obs["utterance"] == "Click button ONE."
        action = self.create_click_button_action(env, obs, "TWO")
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward < 0


################################################


class TestFindGreatest(MiniWoBTester):
    """Tests for the find-greatest task (regression for issue #108).

    The task asks the agent to reveal the card with the greatest number and
    submit. Only one card can be revealed at a time, so the agent must reveal
    each card in turn to learn its value before submitting the greatest one.
    """

    ENV_NAME = "miniwob/find-greatest-v1"

    def _card_divs(self, obs):
        """Return the card <div>s as (left, top) tuples, ordered left-to-right.

        The card divs have a fixed size and are always present in the
        observation, even while hidden; the numbers themselves only appear
        once a card is revealed.
        """
        cards = [
            (element["left"].item(), element["top"].item())
            for element in obs["dom_elements"]
            if "card" in element["classes"].split()
        ]
        return sorted(cards)

    def _create_click_card(self, env, card):
        """Create an action that clicks on the given card."""
        left, top = card
        return env.unwrapped.create_action(
            ActionTypes.CLICK_COORDS,
            coords=np.array([left + 5, top + 5], dtype=np.float32),
        )

    def _revealed_value(self, obs):
        """Return the number shown on the single currently-revealed card."""
        for element in obs["dom_elements"]:
            text = element["text"].strip()
            if text.isdigit():
                return int(text)
        assert False, "no revealed card value found"

    def _reveal_all_values(self, env, obs):
        """Reveal each card once and return a {card: value} mapping."""
        values = {}
        for card in self._card_divs(obs):
            action = self._create_click_card(env, card)
            obs, reward, terminated, truncated, info = env.step(action)
            assert terminated is False
            values[card] = self._revealed_value(obs)
        return obs, values

    def test_wrong_card_gives_negative_reward(self, env):
        """Submitting a non-greatest card must not yield a positive reward.

        Regression for issue #108: the wrong-card branch previously called
        ``core.endEpisode(0.1, true)``, so an incorrect submission produced a
        positive terminal reward. Downstream wrappers that treat any positive
        raw reward as success (e.g. BrowserGym) then counted a wrong answer as
        a successful run.
        """
        obs, info = env.reset()
        assert len(self._card_divs(obs)) == 3
        obs, values = self._reveal_all_values(env, obs)

        greatest_card = max(values, key=lambda card: values[card])
        wrong_card = next(card for card in values if card != greatest_card)

        action = self._create_click_card(env, wrong_card)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is False
        action = self.create_click_button_action(env, obs, "Submit")
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated is True
        assert reward < 0

    def test_greatest_card_gives_positive_reward(self, env):
        """Submitting the greatest card still yields a positive reward."""
        obs, info = env.reset()
        assert len(self._card_divs(obs)) == 3
        obs, values = self._reveal_all_values(env, obs)

        greatest_card = max(values, key=lambda card: values[card])

        action = self._create_click_card(env, greatest_card)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is False
        action = self.create_click_button_action(env, obs, "Submit")
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated is True
        assert reward > 0


################################################


class TestMiniWoBFields(MiniWoBTester):
    """Tests for field extraction."""

    ENV_NAME = "miniwob/email-inbox-forward-nl-v1"

    def test_fields(self, env):
        """Test field extraction."""
        # Training time
        obs, info = env.reset()
        assert {"by", "to"} <= {x[0] for x in obs["fields"]}
        assert field_lookup(obs["fields"], "by") in obs["utterance"]
        assert field_lookup(obs["fields"], "to") in obs["utterance"]
        # Test time
        obs, info = env.reset(options={"data_mode": "test"})
        assert not obs["fields"]
        assert obs["utterance"]
        # Training time again
        obs, info = env.reset(options={"data_mode": "train"})
        assert {"by", "to"} <= {x[0] for x in obs["fields"]}
        assert field_lookup(obs["fields"], "by") in obs["utterance"]
        assert field_lookup(obs["fields"], "to") in obs["utterance"]


################################################


class RewardProcessorTester(MiniWoBTester):
    """Base class for testing reward processors."""

    ENV_NAME = "miniwob/ascending-numbers-v1"

    def _create_click_number(self, env, initial_obs, number):
        for element in initial_obs["dom_elements"]:
            if element["tag"] == "text" and element["text"] == str(number):
                left = element["left"].item() + 5
                top = element["top"].item() + 5
                return env.unwrapped.create_action(
                    ActionTypes.CLICK_COORDS,
                    coords=np.array([left, top], dtype=np.float32),
                )
        assert False, f"Number {number} not found"


class TestGetOriginalReward(RewardProcessorTester):
    @pytest.fixture
    def env(self):
        env = gymnasium.make(self.ENV_NAME, reward_processor=get_original_reward)
        yield env
        env.close()

    @pytest.mark.parametrize(
        "numbers,raw_reward",
        [
            # correct --> reward = 1 * time left
            ([1, 2, 3, 4, 5], 1.0),
            # 2 out of 5 correct --> reward = 0.4 * time left
            ([1, 2, 5], 0.4),
            # initially incorrect --> reward = -1 (no time scaling)
            ([2], -1.0),
        ],
    )
    def test_get_original_reward(self, env, numbers, raw_reward):
        """Test the get_original_reward reward processor."""
        before_reset = time.time()
        initial_obs, info = env.reset()
        after_reset = time.time()

        time.sleep(1)

        for number in numbers[:-1]:
            action = self._create_click_number(env, initial_obs, number)
            obs, reward, terminated, truncated, info = env.step(action)
            assert terminated is False
            assert reward == 0

        action = self._create_click_number(env, initial_obs, numbers[-1])

        before_final_step = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        after_final_step = time.time()

        assert terminated is True

        if raw_reward < 0:
            assert reward == raw_reward
            return

        episode_max_time = 10.0

        min_elapsed = before_final_step - after_reset
        max_elapsed = after_final_step - before_reset

        min_remaining = max(0.0, 1.0 - max_elapsed / episode_max_time)
        max_remaining = max(0.0, 1.0 - min_elapsed / episode_max_time)

        assert raw_reward * min_remaining <= reward <= raw_reward * max_remaining


class TestGetRawReward(RewardProcessorTester):
    @pytest.fixture
    def env(self):
        env = gymnasium.make(self.ENV_NAME, reward_processor=get_raw_reward)
        yield env
        env.close()

    @pytest.mark.parametrize(
        "numbers,check_reward",
        [
            # correct --> reward = 1
            ([1, 2, 3, 4, 5], lambda r: r == 1),
            # 2 out of 5 correct --> reward = 0.4
            ([1, 2, 5], lambda r: r == 0.4),
            # initially incorrect --> reward = -1
            ([2], lambda r: r == -1),
        ],
    )
    def test_get_raw_reward(self, env, numbers, check_reward):
        """Test the get_raw_reward reward processor."""
        initial_obs, info = env.reset()
        for number in numbers[:-1]:
            action = self._create_click_number(env, initial_obs, number)
            obs, reward, terminated, truncated, info = env.step(action)
            assert terminated is False
            assert reward == 0
        action = self._create_click_number(env, initial_obs, numbers[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert check_reward(reward)


class TestGetBinaryReward(RewardProcessorTester):
    @pytest.fixture
    def env(self):
        env = gymnasium.make(self.ENV_NAME, reward_processor=get_binary_reward)
        yield env
        env.close()

    @pytest.mark.parametrize(
        "numbers,check_reward",
        [
            # correct --> reward = 1
            ([1, 2, 3, 4, 5], lambda r: r == 1),
            # 2 out of 5 correct --> reward = -1 (no partial reward)
            ([1, 2, 5], lambda r: r == -1),
            # initially incorrect --> reward = -1
            ([2], lambda r: r == -1),
        ],
    )
    def test_get_binary_reward(self, env, numbers, check_reward):
        """Test the get_binary_reward reward processor."""
        initial_obs, info = env.reset()
        for number in numbers[:-1]:
            action = self._create_click_number(env, initial_obs, number)
            obs, reward, terminated, truncated, info = env.step(action)
            assert terminated is False
            assert reward == 0
        action = self._create_click_number(env, initial_obs, numbers[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert check_reward(reward)


class TestGetThresholdedReward(RewardProcessorTester):
    @pytest.fixture
    def env(self):
        env = gymnasium.make(
            self.ENV_NAME,
            reward_processor=functools.partial(get_thresholded_reward, threshold=0.5),
        )
        yield env
        env.close()

    @pytest.mark.parametrize(
        "numbers,check_reward",
        [
            # correct --> reward = 1
            ([1, 2, 3, 4, 5], lambda r: r == 1),
            # 3 out of 5 correct --> raw reward = 0.6 --> reward = 1
            ([1, 2, 3, 5], lambda r: r == 1),
            # 2 out of 5 correct --> raw reward = 0.4 --> reward = -1
            ([1, 2, 5], lambda r: r == -1),
            # initially incorrect --> reward = -1
            ([2], lambda r: r == -1),
        ],
    )
    def test_get_thresholded_reward(self, env, numbers, check_reward):
        """Test the get_thresholded_reward reward processor."""
        initial_obs, info = env.reset()
        for number in numbers[:-1]:
            action = self._create_click_number(env, initial_obs, number)
            obs, reward, terminated, truncated, info = env.step(action)
            assert terminated is False
            assert reward == 0
        action = self._create_click_number(env, initial_obs, numbers[-1])
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert check_reward(reward)
