# Arch Linux

Based on `archlinux:latest`. CUDA and cuDNN are installed manually via `pacman` for the GPU version.

## Build

> **Note:** The default target architecture matches the build host. Pass `--build-arg TARGETPLATFORM=<value>` to override:
> - `linux/amd64` — x86_64 servers (e.g. RCP)
> - `linux/arm64` — ARM64 machines (e.g. Apple Silicon)

GPU (default):

> **Warning:** Building the GPU image takes ~15 minutes and produces an image of ~27 GB. This is due to the lack of an optimized base image with CUDA pre-installed (it is compiled and installed from scratch during the build).

```bash
sudo docker build -f docker/arch/Dockerfile . -t mmore:arch
```

CPU-only:
```bash
sudo docker build -f docker/arch/Dockerfile --build-arg DEVICE=cpu -t mmore:arch-cpu .
```

Custom extras (overrides the default `--extra all,cu126` or `--extra all,cpu`):
```bash
sudo docker build -f docker/arch/Dockerfile --build-arg UV_OVERRIDE="--extra all,cu126" -t mmore:arch .
```

Custom user UID/GID (e.g. for RCP):
```bash
sudo docker build -f docker/arch/Dockerfile --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t mmore:arch .
```

## Run

```bash
# GPU
sudo docker run --gpus all -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore:arch

# CPU-only
sudo docker run -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore:arch-cpu
```
