# Arch Linux

Based on `archlinux:latest`. CUDA and cuDNN are installed manually via `pacman` for the GPU version.

## Build

GPU (default):
```bash
sudo docker build -f docker/arch/Dockerfile . -t mmore:arch
```

CPU-only:
```bash
sudo docker build -f docker/arch/Dockerfile --build-arg PLATFORM=cpu -t mmore:arch-cpu .
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
