# openSUSE Leap

Based on `opensuse/leap:15.6`. CUDA 12.6 is installed manually via the NVIDIA zypper repository for the GPU version.

## Build

> **Note:** The default target architecture matches the build host. Pass `--platform=<value>` to override:
> - `linux/amd64` — x86_64 servers (e.g. RCP)
> - `linux/arm64` — ARM64 machines (e.g. Apple Silicon)
> - `windows/amd64` — x86_64 windows machines

GPU (default):

> **Warning:** Building the GPU image takes ~15 minutes and produces an image of ~26 GB. This is due to the lack of an optimized base image with CUDA pre-installed (it is installed and compiled from scratch during the build).

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
