# Creating a Custom Environment

Creating a custom MiniWoB++ environment is as simple as creating a new task HTML page, and then specifying the URL to the HTML file when registering the environment.

The following tutorial illustrates how to create a custom environment with the standard [observation space](/content/observation_space) and [action space](/content/action_space). For more advanced needs (customizing the spaces, creating a package, etc.), see this [Gymnasium tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

## Creating the Task HTML Page

```{image} /_static/img/custom-environment-1.png
:width: 50%
:align: center
```

To properly interface with the MiniWoB++ library, the HTML file must include
[`core.css`](https://github.com/Farama-Foundation/miniwob-plusplus/blob/main/miniwob/html/core/core.css)
and
[`core.js`](https://github.com/Farama-Foundation/miniwob-plusplus/blob/main/miniwob/html/core/core.js).
These files are also available from the Python package (under `html/core/`).

Here is an example HTML file (`custom.html`):

```html
<!DOCTYPE html>
<html>
<head>
<title>Custom Task</title>
<!-- The following CSS and JS are required. -->
<link rel="stylesheet" type="text/css" href="core.css">
<script src="core.js"></script>
<script>
// genProblem() will be called at the start of each episode.
var genProblem = function() {
  // core.randi(a, b) (from core.js) samples an integer between a and b (inclusive).
  let targetAmount = core.randi(2, 5), clickedAmount = 0;

  // Set the task utterance. This will become the "utterance" in the observation.
  document.getElementById("query").textContent =
      `Click Hello ${targetAmount} times, then click Submit.`;

  // Create the task UI.
  let hello = document.createElement("button");
  let submit = document.createElement("button");
  let amount = document.createElement("span");
  document.getElementById("area").replaceChildren(hello, submit, amount);

  hello.textContent = "Hello";
  hello.addEventListener("click", () => {
    clickedAmount++;
    amount.textContent = clickedAmount;
  });

  submit.textContent = "Submit";
  submit.addEventListener("click", () => {
    if (clickedAmount === targetAmount) {
      // core.endEpisode(reward, timeProportional) ends the episode and sets
      // the reward (which can be read by Selenium). If timeProportional is
      // true, then the reward is scaled by the amount of time left.
      core.endEpisode(1.0, true);
    } else {
      core.endEpisode(-1.0, false);
    }
  });
}

window.onload = function() {
  // core.startEpisode() prepares the web page for an episode by showing the "START"
  // screen (it does not actually start the episode until "START" is clicked).
  // core.startEpisode() will also be automatically called at the end of each episode.
  core.startEpisode();
}
</script>
</head>
<body>
<!-- The following 3 divs are required. -->
<div id="wrap">
  <div id="query"></div>
  <div id="area"></div>
</div>
</body>
</html>
```

## Registering the Environment

The generic environment class `MiniWoBEnvironment` from `miniwob.environment` can be used as the entry point for the environment.
The following keyword arguments should be specified:

* `subdomain`: This should match the HTML filename.
* `base_url`: This should point to where the HTML is; the URL `{base_url}/{subdomain}.html` should point to the task page.
* `field_extractor`: This should be a function that takes the utterance string (that appears in the yellow box) and returns a list of fields as key-value tuples. One option is to not produce any field by specifying `lambda x: []`.

Here is example code (`custom_registry.py`):


```python
import pathlib
from gymnasium.envs.registration import register
from miniwob.fields import create_regex_field_extractor

register(
  id='miniwob/custom-v0',
  entry_point='miniwob.environment:MiniWoBEnvironment',
  kwargs={
    'subdomain': 'custom',
    # Assuming that this file is in the same directory as `custom.html`:
    'base_url': 'file://{}/'.format(pathlib.Path(__file__).parent),
    # This helper method will produce a function that takes a string and returns
    # [('amount', ___)], where ___ is from the capturing group in the regex.
    'field_extractor': create_regex_field_extractor(
      r'Click Hello (\d+) times, then click Submit\.', ['amount'],
    )
  },
)
```

## Using the Environment

The following code is adapted from the example in the [Basic Usage](/content/basic_usage) page:

```python
import time
import gymnasium
from miniwob.action import ActionTypes
from miniwob.fields import field_lookup

# Import `custom_registry.py` above to register the task.
import custom_registry
gymnasium.register_envs(custom_registry)

# Create an environment.
env = gymnasium.make('miniwob/custom-v0', render_mode='human')

# Wrap the code in try-finally to ensure proper cleanup.
try:
  # Start a new episode.
  observation, info = env.reset()

  # Find the relevant HTML elements.
  for hello_button in observation["dom_elements"]:
    if hello_button["text"] == "Hello":
      break
  for submit_button in observation["dom_elements"]:
    if submit_button["text"] == "Submit":
      break

  amount = int(field_lookup(observation['fields'], 'amount'))
  for _ in range(amount):
    # Click Hello.
    action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=hello_button["ref"])
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.5)

  # Click Submit.
  action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=submit_button["ref"])
  observation, reward, terminated, truncated, info = env.step(action)
  
  # Check if the action was correct. 
  assert reward >= 0
  assert terminated is True
  time.sleep(0.5)

finally:
  env.close()
```
