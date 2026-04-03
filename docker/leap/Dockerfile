ARG DEVICE
ARG UV_OVERRIDE

# --- Base images ---
FROM opensuse/leap:15.6 AS gpu
ENV UV_ARGUMENTS="--extra all,cu126"
RUN echo "Using GPU image"

FROM opensuse/leap:15.6 AS cpu
ENV UV_ARGUMENTS="--extra all,cpu"
RUN echo "Using CPU-only image"

# --- Build ---
FROM ${DEVICE:-gpu} AS build
ARG UV_OVERRIDE
ENV UV_ARGUMENTS=${UV_OVERRIDE:-$UV_ARGUMENTS}

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

WORKDIR /app

# Set up Python virtual environment
RUN python3.11 -m venv .venv \
 && .venv/bin/pip install --no-cache-dir uv weasyprint

# Install dependencies (cached unless uv.lock changes)
COPY pyproject.toml uv.lock /app/
RUN .venv/bin/uv sync --frozen --no-install-project ${UV_ARGUMENTS}

# Install mmore from local source code
COPY . /app
RUN .venv/bin/uv pip install --no-cache --no-deps -e .

# --- Runtime ---
ENV PATH="/app/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False
ENV HF_HOME="/root/.cache/huggingface"
ENV TORCH_HOME="/root/.cache/torch"
ENV XDG_CACHE_HOME="/root/.cache"

ENTRYPOINT ["/bin/bash"]
