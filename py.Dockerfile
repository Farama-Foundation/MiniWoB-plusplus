# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION
# TODO: Maybe use a Selenium docker?

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update
RUN apt-get install -y chromium chromium-driver

COPY . /usr/local/miniwob-plusplus/
WORKDIR /usr/local/miniwob-plusplus/
RUN pip install -e .[testing] --no-cache-dir
