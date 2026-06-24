---
hide-toc: true
firstpage:
lastpage:
---


```{project-logo} _static/img/miniwobplusplus-text.png
:alt: MiniWoB logo
```

```{image} _static/img/showcase.gif
:width: 100%
:align: center
```

The MiniWoB++ library contains a collection of over 100 **web interaction environments**,
along with JavaScript and Python interfaces for programmatically interacting with them.
The Python interface follows the [Gymnasium](https://gymnasium.farama.org/) API
and uses [Selenium WebDriver](https://www.selenium.dev/documentation/webdriver/)
to perform actions on the web browser.

MiniWoB++ is an extension of the
[OpenAI MiniWoB benchmark](http://proceedings.mlr.press/v70/shi17a/shi17a.pdf),
and was introduced in the paper
[Reinforcement Learning on Web Interfaces using Workflow-Guided
Exploration](https://arxiv.org/abs/1802.08802).

The Gymnasium interface allows an agent to initialize and interact with a MiniWoB++ environment as follows:
```python
import gymnasium
import miniwob
gymnasium.register_envs(miniwob)
env = gymnasium.make('miniwob/click-test-2-v1', render_mode='human')
try:
  observation, info = env.reset(seed=42)
  for _ in range(1000):
    action = policy(observation)  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
      observation, info = env.reset()
finally:
  env.close()
```

```{toctree}
:hidden:
:caption: Introduction

content/getting_started
content/basic_usage
```

```{toctree}
:hidden:
:caption: API

content/observation_space
content/action_space
content/reward
```

```{toctree}
:hidden:
:caption: Environments

content/viewing
environments/list
content/custom_environment
content/javascript_api
content/demonstrations
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/miniwob-plusplus>
Contribute to the Docs <https://github.com/Farama-Foundation/miniwob-plusplus/blob/main/docs/README.md>
```
