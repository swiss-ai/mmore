# openSUSE Leap

Based on `opensuse/leap:15.6`. CUDA 12.6 is installed manually via the NVIDIA zypper repository for the GPU version.

## Build

> **Note:** The default target architecture is `linux/amd64`. Pass `--build-arg TARGET_PLATFORM=<value>` to override:
> - `linux/amd64` — x86_64 servers (e.g. RCP)
> - `linux/arm64` — ARM64 machines (e.g. Apple Silicon)

GPU (default):
```bash
sudo docker build -f docker/sles/Dockerfile . -t mmore:sles
```

CPU-only:
```bash
sudo docker build -f docker/sles/Dockerfile --build-arg DEVICE=cpu -t mmore:sles-cpu .
```

Custom extras (overrides the default `--extra all,cu126` or `--extra all,cpu`):
```bash
sudo docker build -f docker/sles/Dockerfile --build-arg UV_OVERRIDE="--extra all,cu126" -t mmore:sles .
```

Custom user UID/GID (e.g. for RCP):
```bash
sudo docker build -f docker/sles/Dockerfile --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t mmore:sles .
```

## Run

```bash
# GPU
sudo docker run --gpus all -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore:sles

# CPU-only
sudo docker run -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore:sles-cpu
```
