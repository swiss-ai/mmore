# CSCS (NVIDIA PyTorch)

Based on `nvcr.io/nvidia/pytorch:24.12-py3`. This image is tailored for [CSCS](https://www.cscs.ch/) clusters, where the base image provides CUDA, cuDNN, and PyTorch pre-installed.

## Build

CSCS Alps uses ARM64 (`linux/arm64`). If building from a non-ARM host, pass `--platform linux/arm64`.

```bash
sudo docker build -f docker/cscs/Dockerfile --platform linux/arm64 -t mmore:cscs .
```

Custom extras (overrides the default `--extra all`):

```bash
sudo docker build -f docker/cscs/Dockerfile --platform linux/arm64 --build-arg UV_OVERRIDE="--extra all" -t mmore:cscs .
```
 
## Push to a registry

Tag and push to a registry accessible from CSCS (replace `<registry>` with your target):

```bash
docker tag mmore:cscs <registry>/mmore:latest
docker push <registry>/mmore:latest
```
