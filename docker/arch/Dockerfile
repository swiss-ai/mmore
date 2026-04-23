ARG DEVICE
ARG UV_OVERRIDE

# --- Base images ---
FROM archlinux:latest AS gpu
ENV UV_ARGUMENTS="--extra all,cu126"
RUN echo "Using GPU image"

FROM archlinux:latest AS cpu
ENV UV_ARGUMENTS="--extra all,cpu"
RUN echo "Using CPU-only image"

# --- Build ---
FROM ${DEVICE:-gpu} AS build
ARG UV_OVERRIDE
ENV UV_ARGUMENTS=${UV_OVERRIDE:-$UV_ARGUMENTS}
RUN echo "DisableSandbox" >> /etc/pacman.conf

ARG USER_UID=1000
ARG USER_GID=1000

# Install system dependencies
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm --needed \
      base-devel python uv \
      tzdata curl ffmpeg \
      libsm libxext nss \
      libxi libxrandr libxcomposite libxcursor libxdamage libxfixes libxrender \
      alsa-lib atk gtk3 libreoffice-fresh libjpeg-turbo pango && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    echo "Europe/Zurich" > /etc/timezone && \
    pacman -Sc --noconfirm

# Create non-root user
RUN groupadd --gid ${USER_GID} mmoreuser \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m mmoreuser

WORKDIR /app
RUN chown mmoreuser:mmoreuser /app
USER mmoreuser

# Set up Python virtual environment
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv --python 3.11 .venv \
 && uv pip install --no-cache weasyprint

# Install dependencies (cached unless uv.lock changes)
COPY --chown=mmoreuser:mmoreuser pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project ${UV_ARGUMENTS}

# Install mmore from local source code
COPY --chown=mmoreuser:mmoreuser . /app
RUN uv pip install --no-cache --no-deps -e .

# --- Runtime ---
ENV PATH="/app/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False
ENV HF_HOME="/home/mmoreuser/.cache/huggingface"
ENV TORCH_HOME="/home/mmoreuser/.cache/torch"
ENV XDG_CACHE_HOME="/home/mmoreuser/.cache"

ENTRYPOINT ["/bin/bash"]
