ARG PLATFORM
ARG UV_ARGUMENTS=""

# We select the image based on the platform argument

# Define GPU image
FROM "nvidia/cuda:12.2.2-base-ubuntu22.04" AS gpu
ARG PLATFORM
RUN echo "Using GPU image"

# Define cpu image
FROM ubuntu:22.04 as cpu
ARG PLATFORM
ARG UV_ARGUMENTS="--extra cpu"
RUN echo "Using CPU-only image"

# Select image
FROM ${PLATFORM:-gpu}
ARG PLATFORM

COPY --from=ghcr.io/astral-sh/uv:0.5.8 /uv /uvx /bin/

RUN apt-get update && \
   apt-get install -y  --no-install-recommends  \
      nano curl ffmpeg libsm6 libxext6 chromium-browser libnss3 libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice libjpeg-dev


# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app

# Define the build argument with a default value of an empty string (optional)

RUN uv sync --frozen ${UV_ARGUMENTS}


# make uv's python the default python for the image
ENV PATH="/app/.venv/bin:$PATH"

ENV DASK_DISTRIBUTED__WORKER__DAEMON=False

ENTRYPOINT /bin/bash

