# Getting Started

## Install the MiniWoB++ Library

To install the MiniWoB++ library, use `pip install miniwob`.

Alternatively, clone the repository from GitHub:

```sh
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git
cd miniwob-plusplus
pip install -e .
```

If this gives you problems, try again and add pip's `--ignore-installed` flag.

## Install Chrome/Chromium and ChromeDriver

We strongly recommend using Chrome or Chromium as the web browser,
as other browsers may render the environments differently.

The MiniWoB++ Python interface uses [Selenium](https://www.selenium.dev/documentation/webdriver/),
which interacts with the browser via the [WebDriver API](https://w3c.github.io/webdriver/).
Follow one of the
[instruction methods](https://www.selenium.dev/documentation/webdriver/getting_started/install_drivers/)
to install ChromeDriver. The simplest method is to
[download](https://chromedriver.chromium.org/downloads) ChromeDriver with the matching version,
unzip it, and then add the directory containing the `chromedriver` executable to the `PATH` environment variable:

```sh
export PATH=$PATH:/path/to/chromedriver
```

For Chromium, the driver may also be available in a software package; for example, in Debian/Ubuntu:

```sh
sudo apt install chromium-driver
```

## Development Environment

For information about setting up a development environment,
see [`CONTRIBUTING.md`](https://github.com/Farama-Foundation/miniwob-plusplus/blob/main/CONTRIBUTING.md).
