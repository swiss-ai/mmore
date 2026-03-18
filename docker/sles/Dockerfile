ARG TARGET_PLATFORM=linux/amd64
ARG DEVICE
ARG UV_OVERRIDE

# --- Base images ---
FROM --platform=${TARGET_PLATFORM} opensuse/leap:15.6 AS gpu
ENV UV_ARGUMENTS="--extra all,cu126"
RUN echo "Using GPU image"
RUN zypper --non-interactive addrepo \
      https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/x86_64/cuda-opensuse15.repo && \
    zypper --non-interactive --gpg-auto-import-keys refresh && \
    zypper --non-interactive install -y cuda-12-6

FROM --platform=${TARGET_PLATFORM} opensuse/leap:15.6 AS cpu
ENV UV_ARGUMENTS="--extra all,cpu"
RUN echo "Using CPU-only image"

# --- Build ---
FROM --platform=${TARGET_PLATFORM} ${DEVICE:-gpu} AS build
ARG UV_OVERRIDE
ENV UV_ARGUMENTS=${UV_OVERRIDE:-$UV_ARGUMENTS}

ARG USER_UID=1000
ARG USER_GID=1000

# Install system dependencies
RUN zypper --non-interactive refresh && \
    zypper --non-interactive install -y \
      python311 python311-pip \
      nano curl ffmpeg \
      libSM6 libXext6 mozilla-nss \
      libXi6 libXrandr2 libXcomposite1 libXcursor1 libXdamage1 libXfixes3 libXrender1 \
      libasound2 libatk-1_0-0 gtk3 libreoffice libjpeg8-devel libpango-1_0-0 && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    echo "Europe/Zurich" > /etc/timezone && \
    zypper clean --all

# Create non-root user
RUN groupadd --gid ${USER_GID} mmoreuser \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m mmoreuser

WORKDIR /app
RUN chown mmoreuser:mmoreuser /app
USER mmoreuser

# Set up Python virtual environment
RUN python3.11 -m venv .venv \
 && .venv/bin/pip install --no-cache-dir uv weasyprint

# Install dependencies (cached unless uv.lock changes)
COPY --chown=mmoreuser:mmoreuser pyproject.toml uv.lock /app/
RUN .venv/bin/uv sync --frozen --no-install-project ${UV_ARGUMENTS}

# Install the project
COPY --chown=mmoreuser:mmoreuser . /app
RUN .venv/bin/uv pip install --no-cache --no-deps -e .

# --- Runtime ---
ENV PATH="/app/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False

ENTRYPOINT ["/bin/bash"]
