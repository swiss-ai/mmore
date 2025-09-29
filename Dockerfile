ARG PLATFORM
ARG UV_ARGUMENTS=""

FROM nvidia/cuda:12.2.2-base-ubuntu22.04 AS gpu
ARG PLATFORM
RUN echo "Using GPU image"

FROM ubuntu:22.04 AS cpu
ARG PLATFORM
ARG UV_ARGUMENTS="--extra cpu"
RUN echo "Using CPU-only image"

FROM ${PLATFORM:-gpu} AS build
ARG UV_ARGUMENTS

ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3-venv python3-pip \
      tzdata nano curl ffmpeg libsm6 libxext6 chromium-browser libnss3 libgconf-2-4 \
      libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxrender1 \
      libasound2 libatk1.0-0 libgtk-3-0 libreoffice libjpeg-dev libpango-1.0-0 \
      libpangoft2-1.0-0 weasyprint && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd --gid ${USER_GID} mmoreuser \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m mmoreuser

WORKDIR /app
RUN chown -R mmoreuser:mmoreuser /app

USER mmoreuser

RUN python3 -m venv .venv \
 && .venv/bin/pip install --no-cache-dir uv

COPY pyproject.toml poetry.lock* /app/
COPY --chown=mmoreuser:mmoreuser . /app

RUN .venv/bin/uv pip install --no-cache ${UV_ARGUMENTS} -e .

ENV PATH="/app/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False

ENTRYPOINT ["/bin/bash"]
