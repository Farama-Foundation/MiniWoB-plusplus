"""Generate Environments List page content."""
import os

import gymnasium as gym
import pandas as pd

import miniwob
from utils import extract_description_from_docstring, get_all_registered_miniwob_envs


gym.register_envs(miniwob)
gym.logger.min_level = 100

ENV_TYPES = [
    {
        "name": "Original Tasks",
        "description": "These are the original MiniWoB tasks.",
        "envs": [
            "bisect-angle",
            "book-flight",
            "choose-date",
            "choose-list",
            "circle-center",
            "click-button",
            "click-button-sequence",
            "click-checkboxes",
            "click-collapsible",
            "click-collapsible-2",
            "click-color",
            "click-dialog",
            "click-dialog-2",
            "click-link",
            "click-menu",
            "click-menu-2",
            "click-option",
            "click-pie",
            "click-scroll-list",
            "click-shades",
            "click-shape",
            "click-tab",
            "click-tab-2",
            "click-test",
            "click-test-2",
            "click-widget",
            "copy-paste",
            "copy-paste-2",
            "count-shape",
            "count-sides",
            "drag-box",
            "drag-circle",
            "drag-cube",
            "drag-items",
            "drag-items-grid",
            "drag-shapes",
            "drag-sort-numbers",
            "email-inbox",
            "enter-date",
            "enter-password",
            "enter-text",
            "enter-text-2",
            "enter-text-dynamic",
            "enter-time",
            "find-midpoint",
            "find-word",
            "focus-text",
            "focus-text-2",
            "grid-coordinate",
            "guess-number",
            "highlight-text",
            "highlight-text-2",
            "identify-shape",
            "login-user",
            "navigate-tree",
            "number-checkboxes",
            "read-table",
            "read-table-2",
            "resize-textarea",
            "right-angle",
            "scroll-text",
            "scroll-text-2",
            "search-engine",
            "simple-algebra",
            "simple-arithmetic",
            "social-media",
            "terminal",
            "text-editor",
            "text-transform",
            "tic-tac-toe",
            "use-autocomplete",
            "use-colorwheel",
            "use-colorwheel-2",
            "use-slider",
            "use-slider-2",
            "use-spinner",
            "visual-addition",
        ],
    },
    {
        "name": "No-delay Tasks",
        "description": """The UI elements in some tasks have animation delays
or change the state when the browser is defocused.
We provide the "no-delay" version without these issues.""",
        "envs": [
            "book-flight-nodelay",
            "choose-date-nodelay",
            "click-collapsible-nodelay",
            "click-collapsible-2-nodelay",
            "click-pie-nodelay",
            "use-autocomplete-nodelay",
        ],
    },
    {
        "name": "Additional Tasks",
        "description": """These tasks were introduced in MiniWoB++.
Some are harder versions of the existing tasks, while some are completely new.""",
        "envs": [
            "click-checkboxes-large",
            "click-checkboxes-soft",
            "click-checkboxes-transfer",
            "click-tab-2-hard",
            "login-user-popup",
            "multi-layouts",
            "multi-orderings",
            "social-media-all",
            "social-media-some",
            "email-inbox-forward-nl",
            "email-inbox-forward-nl-turk",
            "email-inbox-nl-turk",
        ],
    },
    {
        "name": "Debug Tasks",
        "description": "These are easier versions of existing tasks, suitable for debugging.",
        "envs": [
            "choose-date-easy",
            "choose-date-medium",
            "click-tab-2-easy",
            "click-tab-2-medium",
            "click-test-transfer",
            "email-inbox-delete",
            "email-inbox-forward",
            "email-inbox-important",
            "email-inbox-noscroll",
            "email-inbox-reply",
            "email-inbox-star-reply",
            "unicode-test",
        ],
    },
    {
        "name": "Flight Search Tasks",
        "description": """These are ports of the FormWoB tasks in the original World of Bits paper.,
* The prompt is a list of key-value pairs (e.g., Departure City: New York).
* If the required fields are not filled, or if the agent navigates away from the page, the reward is -1.,
* Otherwise, the reward is the fraction of key-value pairs that are satisfied.""",
        "envs": ["flight.Alaska", "flight.Alaska-auto", "flight.AA"],
    },
    {
        "name": "Hidden Test Tasks",
        "description": "These are tasks intended to be used as the hidden test set. They were originally not available from the OpenAI website.",
        "envs": [
            "ascending-numbers",
            "buy-ticket",
            "daily-calendar",
            "drag-single-shape",
            "drag-shapes-2",
            "draw-circle",
            "draw-line",
            "find-greatest",
            "form-sequence",
            "form-sequence-2",
            "form-sequence-3",
            "generate-number",
            "hot-cold",
            "odd-or-even",
            "order-food",
            "phone-book",
            "sign-agreement",
            "stock-market",
        ],
    },
]


# Get environment descriptions
ENVS_DESCRIPTIONS = {}
filtered_envs = get_all_registered_miniwob_envs()
for env_spec in filtered_envs:
    try:
        env_spec = gym.spec(env_spec.id)

        split = env_spec.entry_point.split(":")
        mod = __import__(split[0], fromlist=[split[1]])
        env_class = getattr(mod, split[1])
        docstring = env_class.__doc__

        if not docstring:
            docstring = env_class.__class__.__doc__

        description = extract_description_from_docstring(docstring)
        ENVS_DESCRIPTIONS[env_spec.name] = description
    except Exception as e:
        print(e)


file_start_content = """# Environments List

```{toctree}
:hidden:
:glob:

../environments/*
```

"""

list_md_path = os.path.join(os.path.dirname(__file__), "..", "environments", "list.md")
with open(list_md_path, "w") as fp:
    content = file_start_content
    for env_type in ENV_TYPES:
        df = pd.DataFrame(
            {
                "Name": [f"[{env}](./{env})" for env in env_type["envs"]],
                "Description": [ENVS_DESCRIPTIONS[env] for env in env_type["envs"]],
            }
        )
        type_name = env_type["name"]
        type_desc = env_type["description"]
        content += f"## {type_name}\n\n"
        content += f"{type_desc}\n\n"
        content += df.to_markdown(index=False) + "\n\n"

    content += """
## Excluded Tasks

The following tasks require the agent to wait for events to happen before acting,
and a 'no-delay' version is impossible to make.

| Name         | Description                                                      |
|:------------ |:---------------------------------------------------------------- |
| chase-circle | Keep your mouse inside a moving circle.                          |
| moving-items | Click moving items before they disappear.                        |
| simon-says   | Push the buttons in the order shown.                             |
| button-delay | Wait a certain period of time before clicking the second button. |
| hover-shape  | Hover over the colored shape.                                    |
"""
    fp.write(content)
