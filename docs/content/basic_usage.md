# Basic Usage

## Simple Example

The following code performs a deterministic action on the
[`click-test-2`](/environments/click-test-2) environment.

```python
import time
import gymnasium
import miniwob
from miniwob.action import ActionTypes

gymnasium.register_envs(miniwob)

env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')

# Wrap the code in try-finally to ensure proper cleanup.
try:
  # Start a new episode.
  observation, info = env.reset()
  assert observation["utterance"] == "Click button ONE."
  assert observation["fields"] == (("target", "ONE"),)
  time.sleep(2)       # Only here to let you look at the environment.
  
  # Find the HTML element with text "ONE".
  for element in observation["dom_elements"]:
    if element["text"] == "ONE":
      break

  # Click on the element.
  action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=element["ref"])
  observation, reward, terminated, truncated, info = env.step(action)

  # Check if the action was correct. 
  print(reward)      # Should be around 0.8 since 2 seconds has passed.
  assert terminated is True
  time.sleep(2)

finally:
  env.close()
```

The output should look something like this:

```{image} /_static/img/example-usage-1.png
:width: 50%
:align: center
```

After 2 seconds:

```{image} /_static/img/example-usage-2.png
:width: 50%
:align: center
```

## Environment Initialization

An environment can be created using
[`gymnasium.make`](https://gymnasium.farama.org/api/registry/#gymnasium.make):

```python
env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
```

Common arguments include:

* **`render_mode`:** Render mode. Supported values are:
    - `None` (default): Headless Chrome, which does not show the browser window.
    - `"human"`: Show the browser window.
* **`action_space_config`:** Configuration for the action space.
  Supported values are:
    - An [`ActionSpaceConfig`](/content/action_space.md#action-configs) object.
    - A [preset name](/content/action_space.md#presets), which will instantiate an `ActionSpaceConfig` object.

## Observation Space

```python
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)
```

```{warning}
Not all MiniWoB++ environments are fully deterministic even when seeded.
Sources of non-determinism include wall-clock-dependent rendering (focus rings, CSS animations),
jQuery UI widget state that leaks across episode resets, variable `Math.random()` consumption in
JS generation loops, and browser font-metric differences that affect layout.

32 out of 128 environments are registered with `nondeterministic=True` in Gymnasium.
Please [submit a bug report](https://github.com/Farama-Foundation/miniwob-plusplus/issues/new/choose) if you find any new ones.
```

The [`reset`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)
and [`step`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) methods
return an observation, which is a `dict` with the following fields:

* **`utterance`:** Task instruction string, such as `"Click button ONE."`.
* **`fields`:** Environment-specific key-value pairs extracted from the utterance, such as `(("target", "ONE"),)`.
* **`screenshot`:** A numpy array of shape `(height, width, 3)` containing the RGB values.
* **`dom_elements`:** A tuple of dicts, each listing properties like the geometry and HTML attributes of a visible DOM element.

For example, the `observation` from the `reset` command above is
```python
{
  'utterance': 'Click button ONE.',
  'fields': (('target', 'ONE'),),
  'screenshot': array([[[255, 255,   0], ...], ...], dtype=uint8),
  'dom_elements': (
    {'ref': 1, 'parent': 0, 'tag': 'body', ...},
    {'ref': 2, 'parent': 1, 'tag': 'div', ...},
    {'ref': 3, 'parent': 2, 'tag': 'div', ...},
    {'ref': 4, 'parent': 3, 'tag': 'button', 'text': 'ONE', ...},
    {'ref': 5, 'parent': 3, 'tag': 'button', 'text': 'TWO', ...},
  ),
}
```

See the [Observation Space](/content/observation_space) page for more details.

## Action Space

```python
action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=element["ref"])
observation, reward, terminated, truncated, info = env.step(action)
```

The [`step`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) method
takes an `action` object, which should be a `dict` with the following fields:

* **`action_type`:** The action type index from `env.unwrapped.action_space_config.action_types`.
* Other fields such as `ref`, `coords`, `text`, etc. should be specified based on the action type.
  The action space `env.unwrapped.action_space` specifies which fields should be included.

For example, the `action` from the `create_action` command above is
```python
{
  'action_type': 8,     # ActionTypes.CLICK_ELEMENT in the default action config.
  'ref': 4,             # The button with text 'ONE' from observation['dom_elements'].
  ...                   # Other fields are ignored for CLICK_ELEMENT.
}  
```
In actual code, the web agent should generate an action based on the observation.

See the [Action Space](/content/action_space) page for more details.
