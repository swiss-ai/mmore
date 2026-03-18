# Ubuntu (default)

Based on `ubuntu:22.04` (CPU) or `nvidia/cuda:12.6.3-base-ubuntu22.04` (GPU).

## Build

GPU (default):
```bash
sudo docker build -f docker/ubuntu/Dockerfile . -t mmore
```

CPU-only:
```bash
sudo docker build -f docker/ubuntu/Dockerfile --build-arg PLATFORM=cpu -t mmore:cpu .
```

Custom extras (overrides the default `--extra all,cu126` or `--extra all,cpu`):
```bash
sudo docker build -f docker/ubuntu/Dockerfile --build-arg UV_OVERRIDE="--extra all,cu126" -t mmore .
```

Custom user UID/GID (e.g. for RCP):
```bash
sudo docker build -f docker/ubuntu/Dockerfile --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t mmore .
```

## Run

```bash
# GPU
sudo docker run --gpus all -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore

# CPU-only
sudo docker run -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore:cpu
```
