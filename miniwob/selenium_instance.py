"""Low-level interface with ChromDriver via Selenium."""

import json
import logging
import pathlib
import shutil
import sys
import time
import traceback
import urllib.parse
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from miniwob.action import Action, ActionSpaceConfig
from miniwob.constants import (
    FLIGHT_TASK_HEIGHT,
    FLIGHT_TASK_WIDTH,
    FLIGHT_WINDOW_HEIGHT,
    FLIGHT_WINDOW_WIDTH,
    TASK_HEIGHT,
    TASK_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from miniwob.dom import DOMElement
from miniwob.fields import FieldExtractor, get_field_extractor
from miniwob.http_server import start_http_server
from miniwob.observation import (
    Observation,
    create_empty_observation,
    create_empty_screenshot,
    create_observation,
)
from miniwob.reward import RewardProcessor, get_original_reward
from miniwob.screenshot import get_screenshot, pil_to_numpy_array
from miniwob.selenium_actions import execute_action_on_chromedriver


HTML_DIR = pathlib.Path(__file__).parent / "html"
DEFAULT_BASE_URL = f"file://{HTML_DIR}/miniwob/"


class SeleniumInstance(Thread):
    """Interface between Python and ChromeDriver via Selenium."""

    def __init__(
        self,
        index: int,
        subdomain: str,
        headless: bool = False,
        base_url: str | None = None,
        threading: bool = False,
        field_extractor: FieldExtractor | None = None,
        reward_processor: RewardProcessor | None = None,
        wait_ms: float = 0.0,
        block_on_reset: bool = True,
        refresh_freq: int = 0,
        data_mode: str = "train",
    ):
        """Starts a new Selenium WebDriver session.

        Args:
            index: Instance index
            subdomain: MiniWoB task name (e.g., "click-test")
            headless: Whether to render GUI
            base_url: Base URL, which is usually one of the following
                - http://localhost:8000/miniwob/     (served by http-serve)
                - file:///path/to/miniwob-plusplus/html/miniwob/
                If None, set the default value as follows:
                - For FlightWoB tasks, starts a HTTP server and use
                http://localhost:[port]/ as base_url.
                - For other tasks, use the inferred file:// path as base_url.
            threading: Whether to run this instance as a Thread
            field_extractor: A function that takes the utterance and returns
                a list of fields as key-value tuples (see miniwob.fields).
                If None, uses the default one for the task.
            reward_processor: A function that takes the metadata and return
                a reward (see miniwob.reward)
            wait_ms: Pause the instance after each action for this amount
                of time (in milliseconds).
            block_on_reset: On reset, block until the page loads.
            refresh_freq: Every this number of episodes, refresh the page at
                the beginning of the next episode. Takes time but cleans up
                any lingering states and memory leaks.
                *** Must specify `seeds` at each reset call.
            data_mode: Data mode (e.g., "train", "test")
        """
        super().__init__()
        # Overrides Thread.daemon: Kill this thread when the parent is killed
        self.daemon = True
        self.died = False
        self.index = index
        self.headless = headless
        if subdomain.startswith("flight."):
            if not base_url:
                base_url = start_http_server(str(HTML_DIR))
            assert not base_url.startswith("file://"), (
                "For {} domain, MINIWOB_BASE_URL cannot be file://."
            ).format(subdomain)
            self.url = urllib.parse.urljoin(
                base_url, subdomain.replace(".", "/") + "/wrapper.html"
            )
            self.window_width = FLIGHT_WINDOW_WIDTH
            self.window_height = FLIGHT_WINDOW_HEIGHT
            self.task_width = FLIGHT_TASK_WIDTH
            self.task_height = FLIGHT_TASK_HEIGHT
        else:
            if not base_url:
                base_url = DEFAULT_BASE_URL
            self.url = urllib.parse.urljoin(base_url, f"{subdomain}.html")
            self.window_width = WINDOW_WIDTH
            self.window_height = WINDOW_HEIGHT
            self.task_width = TASK_WIDTH
            self.task_height = TASK_HEIGHT
        self.inner_height = self.window_height
        self.inner_width = self.window_width
        self.threading = threading
        if not field_extractor:
            self.field_extractor = get_field_extractor(subdomain)
        else:
            self.field_extractor = field_extractor
        if not reward_processor:
            self.reward_processor = get_original_reward
        else:
            self.reward_processor = reward_processor
        self.wait_ms = wait_ms
        self.block_on_reset = block_on_reset
        self.refresh_freq = refresh_freq
        self.num_episodes = 0
        self.mode = data_mode
        self.record_screenshots = True
        self.start_time = float("inf")
        self.task_queue = Queue()
        if not threading:
            # Hack: override the start method of Thread
            self.start = self.create_driver
        self.cached_fields = []

    def run(self):
        """Overrides `Thread.run`."""
        try:
            self.create_driver()
            # Wait for command
            while True:
                func, args = self.task_queue.get()
                try:
                    func(*args)
                except Exception:
                    logging.error("Error in instance %d", self.index)
                    traceback.print_exc()
                    self.died = True
                self.task_queue.task_done()
                if func == self.close:
                    break
        finally:
            self.close()
            logging.info("Closed instance %d", self.index)

    def call(self, func, *args):
        """Call the given function with the given argument.

        If threading is enabled, the function execution is added to the
        task queue, making the call non-blocking.
        """
        if self.threading:
            self.task_queue.put((func, args))
        else:
            func(*args)

    def wait(self):
        """Wait until all tasks in the queue is finished."""
        if self.threading:
            self.task_queue.join()

    ################################
    # Possible Functions

    def create_driver(self):
        """Create a WebDriver."""
        assert not hasattr(self, "driver"), "Instance {} already has a driver".format(
            self.index
        )
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("headless")
            options.add_argument("disable-gpu")
            options.add_argument("no-sandbox")
        else:
            options.add_argument("app=" + self.url)

        if sys.platform == "linux":
            options.binary_location = (
                shutil.which("google-chrome")
                or shutil.which("chromium-browser")
                or shutil.which("chromium")
                or ""
            )
            raise Exception(f"Test: {options.binary_location=}")
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(5)
        if self.headless:
            self.driver.get(self.url)
        try:
            WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, self.SYNC_SCREEN_ID))
            )
        except TimeoutException as e:
            logging.error("Page did not load properly. Wrong URL?")
            raise e
        self.inner_width, self.inner_height = self.driver.execute_script(
            "return [window.innerWidth, window.innerHeight];"
        )

    def close(self):
        """Tear down the WebDriver."""
        # Note: close() will close the current window
        # quit() closes everything, so it is probably cleaner
        try:
            self.driver.quit()
        except Exception:
            logging.error("Error closing the driver of instance %d", self.index)
            traceback.print_exc()
        self.died = True

    def reset(self, obs: list[Any], infos: list[Any], seed: Any):
        """Force stop and start this instance.

        Also sets obs[i] to be the initial observation (where i = self.index).

        Args:
            obs: A list to store observations. The entry obs[i] will be modified.
            infos: A list to store info dicts. The entry infos[i] will be modified.
            seed: Seed to set for the next episode
        """
        if self.refresh_freq:
            assert (
                seed is not None
            ), "reset() must specify seed if refresh_freq is specified"
        i = self.index
        self.force_stop()
        self.begin_task(seed=seed)
        obs[i], extra_metadata = self.get_observation(use_cached_fields=False)
        metadata = self.get_metadata()
        metadata.update(extra_metadata)
        infos[i] = metadata

    def step(
        self,
        action: Action | None,
        action_space_config: ActionSpaceConfig,
        obs: list[Any],
        rewards: list[Any],
        dones: list[Any],
        infos: list[Any],
    ):
        """Apply an action on this instance.

        Also sets obs[i], rewards[i], dones[i], and infos[i] (where i = self.index).

        Args:
            action: The action to execute from the action space, or None (do nothing).
            action_space_config: ActionSpaceConfig object.
            obs: A list to store observations. The entry obs[i] will be modified.
            rewards: A list to store rewards. The entry rewards[i] will be modified.
            dones: A list to store termination statuses. The entry dones[i] will be modified.
            infos: A list to store info dicts. The entry infos[i] will be modified.
        """
        i = self.index
        self.perform(action, action_space_config)
        metadata = self.get_metadata()
        rewards[i] = self.reward_processor(metadata)
        dones[i] = metadata["done"]
        if metadata["done"]:
            obs[i] = self.get_empty_observation()
            extra_metadata = {}
        else:
            obs[i], extra_metadata = self.get_observation(use_cached_fields=True)
        metadata["elapsed"] = max(0.0, time.time() - self.start_time)
        metadata.update(extra_metadata)
        infos[i] = metadata

    ################################
    # Primitive actions

    SYNC_SCREEN_ID = "sync-task-cover"
    RESET_BLOCK_SLEEP_TIME = 0.05  # 50ms
    RESET_BLOCK_MAX_ATTEMPT = 20  # up to 1s

    def force_stop(self):
        """Force stop the task and go back to the sync screen."""
        self.driver.execute_script("return core.endEpisode(0);")

    def begin_task(self, seed: Any = None):
        """Start the task. Only available when done is True.

        The sync screen will disappear and the countdown timer will start.

        Args:
            seed: New seed to set for the next episode
        """
        self.num_episodes += 1
        if self.refresh_freq and self.num_episodes % self.refresh_freq == 0:
            self.driver.get(self.url)
        if seed is not None:
            self.set_seed(seed)
        self.set_mode(self.mode)
        self.driver.execute_script("core.startEpisodeReal();")
        if self.block_on_reset:
            for _ in range(self.RESET_BLOCK_MAX_ATTEMPT):
                if self.driver.execute_script("return WOB_TASK_READY;"):
                    break
                time.sleep(self.RESET_BLOCK_SLEEP_TIME)
            else:
                raise RuntimeError(f"Instance {self.index} does not load properly")
        elif self.wait_ms:
            time.sleep(self.wait_ms / 1000.0)
        self.start_time = time.time()

    def perform(self, action: Action | None, action_space_config: ActionSpaceConfig):
        """Perform an action.

        Args:
            action: The action to execute from the action space, or None (do nothing).
            action_space_config: ActionSpaceConfig object.
        """
        if action is not None:
            if self.get_metadata()["done"]:
                logging.warning(
                    "Cannot call %s on instance %d, which is already done",
                    action,
                    self.index,
                )
            else:
                execute_action_on_chromedriver(
                    action, self.cached_fields, action_space_config, self.driver
                )
        if self.wait_ms:
            time.sleep(self.wait_ms / 1000.0)

    def get_empty_observation(self) -> Observation:
        """Get an empty observation for a terminated session."""
        return create_empty_observation(self.task_width, self.task_height)

    def get_observation(
        self, use_cached_fields: bool = False
    ) -> tuple[Observation, dict[str, Any]]:
        """Get the current observation.

        Args;
            use_cached_fields: Use the cached fields instead of running the field extractor.

        Returns:
            a tuple (observation, extra_metadata).
            observation: Observation object from the observation space.
            extra_metadata: A dict containing the following extra information:
                - root_dom: DOMElement object for the root DOM element.
        """
        # Get the utterance
        response = self.driver.execute_script("return core.getUtterance();")
        if isinstance(response, dict):
            utterance = response["utterance"]
        else:
            utterance = response
        if use_cached_fields:
            fields = self.cached_fields
        else:
            if isinstance(response, dict):
                fields = list(response["fields"].items())
            else:
                fields = self.field_extractor(utterance)
            self.cached_fields = fields
        # Get the DOM
        dom_info = self.driver.execute_script("return core.getDOMInfo();")
        root_dom = DOMElement(dom_info)
        # Get screenshot if requested
        if self.record_screenshots:
            img = get_screenshot(
                self.driver,
                true_width=self.inner_width,
                true_height=self.inner_height,
                crop_width=self.task_width,
                crop_height=self.task_height,
            )
            img = pil_to_numpy_array(img)
        else:
            img = create_empty_screenshot(self.task_width, self.task_height)
        observation = create_observation(utterance, root_dom, img, fields)
        return observation, {"root_dom": root_dom}

    def get_metadata(self) -> dict[str, Any]:
        """Get other metadata.

        Returns:
            dict with the following keys:
            - done (bool)
            - env_reward (float; only well-defined when done is True):
                Environment-defined reward, possibly scaled by time
            - raw_reward (float; only well-defined when done is True):
                Environment-defined reward, NOT scaled by time
            - reason (any): reason for giving the reward (for debugging);
                will likely be None if done is False
        """
        return self.driver.execute_script(
            "return {"
            '"done": WOB_DONE_GLOBAL,'
            '"env_reward": WOB_REWARD_GLOBAL,'
            '"raw_reward": WOB_RAW_REWARD_GLOBAL,'
            '"reason": WOB_REWARD_REASON,'
            "};"
        )

    def visualize_attention(self, attention: np.ndarray | None):
        """Sends the attention weights to be visualized.

        Args:
            attention: one of the following:
                - None: Do not do anything
                - np.array or 2d list of shape (num_grid_rows, num_grid_cols)
                - np.array or 2d list of shape (0, 0): Clear the visualization
        """
        if attention is None:
            return
        # Encode as JSON
        if isinstance(attention, np.ndarray):
            attention = attention.tolist()
        encoded = json.dumps(attention)
        # Send to the driver
        self.driver.execute_script(f"core.visualizeAttention({encoded});")

    def set_seed(self, seed: Any):
        """Set the seed to a new value.

        Args:
            seed: The new seed.
        """
        self.driver.execute_script(f"Math.seedrandom({repr(seed)});")

    def set_mode(self, mode: str):
        """Set the task generation mode (e.g., "train" or "test") to a new value.

        Args:
            mode: The new mode
        """
        self.driver.execute_script(f'core.setDataMode("{mode}");')
