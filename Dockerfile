ARG BASE_IMAGE="nvidia/cuda:12.2.2-base-ubuntu22.04"

FROM ${BASE_IMAGE}

RUN apt-get update && \
   apt-get install -y  --no-install-recommends  \
      nano curl ffmpeg libsm6 libxext6 chromium-browser libnss3 libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice libjpeg-dev

ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

RUN echo 'export RYE_HOME="/opt/rye"' >> ~/.bashrc
RUN echo 'export PATH="$RYE_HOME/shims:$PATH"' >> ~/.bashrc

RUN curl -sSf https://rye.astral.sh/get | RYE_TOOLCHAIN_VERSION="3.11" RYE_INSTALL_OPTION="--yes" bash
RUN rye config --set-bool behavior.global-python=true

COPY .python-version pyproject.toml .python-version requirements.lock README.md ./
COPY src ./src

RUN rye sync

ENV PATH="/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False

ENTRYPOINT /bin/bash