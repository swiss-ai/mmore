# openSUSE Leap

Based on `opensuse/leap:15.6`.

## Build

> **Note:** The default target architecture matches the build host. Pass `--platform=<value>` to override:
> - `linux/amd64` — x86_64 servers (e.g. RCP)
> - `linux/arm64` — ARM64 machines (e.g. Apple Silicon)

GPU (default):

```bash
sudo docker build -f docker/leap/Dockerfile -t mmore:leap .
```

CPU-only:
```bash
sudo docker build -f docker/leap/Dockerfile --build-arg DEVICE=cpu -t mmore:leap-cpu .
```

Custom extras (overrides the default `--extra all,cu126` or `--extra all,cpu`):
```bash
sudo docker build -f docker/leap/Dockerfile --build-arg UV_OVERRIDE="--extra all,cu126" -t mmore:leap .
```

Custom user UID/GID (e.g. for RCP):
```bash
sudo docker build -f docker/leap/Dockerfile --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t mmore:leap .
```

## Run

```bash
# GPU
sudo docker run --gpus all -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore:leap

# CPU-only
sudo docker run -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore:leap-cpu
```
