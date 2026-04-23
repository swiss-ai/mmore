# RCP and Production Deployment

This document provides comprehensive guidelines for deploying MMORE on the RCP (and later to production).

## Docker Image Requirements

**Important**: You must build your own Docker image with your specific user ID and group ID to avoid permission issues in the production environment.

1. **Check your user and group IDs on the RCP**:
   ```bash
   id -u  # Your user ID
   id -g  # Your group ID
   ```

2. **Build Docker image with custom IDs** — choose one of the two options below:

   **Option A — CI build (recommended):** Trigger the [Build Student Image](../.github/workflows/push-to-registry.yml) workflow manually from the GitHub Actions tab (*Run workflow*) and input your user UID and group GID. This builds a custom student GPU image published to GHCR, tagged as:
   ```
   ghcr.io/swiss-ai/mmore:student-uid<user-id>-gid<group-id>-gpu
   ```
   You can then pull it directly with `docker pull`. Skip to step 5.

   **Option B — local build** (replace `<user-id>` and `<group-id>` with your actual IDs):
   ```bash
   sudo docker build -f docker/ubuntu/Dockerfile --build-arg USER_UID=<user-id> --build-arg USER_GID=<group-id> -t mmore .
   ```

3. **Login to DockerHub** *(option B only)*:
   ```bash
   docker login docker.io
   ```

4. **Push to registry** *(option B only)* (replace `<username>` with your DockerHub username):
   ```bash
   docker tag mmore docker.io/<username>/mmore:latest
   docker push docker.io/<username>/mmore:latest
   ```

5. **Identify your image reference** — all `runai` commands below use `<image>` as a placeholder. Replace it with:
   - Option A: `ghcr.io/swiss-ai/mmore:student-uid<user-id>-gid<group-id>-gpu`
   - Option B: `docker.io/<username>/mmore:latest`

For detailed installation instructions, see [Installation Guide](./installation.md).

## Running on the RCP

### Environment Setup

First, set up your environment variables:

```bash
export ROOT_OUT_DIR=/lightscratch/users/$GASPAR/mmore-data/out
export ROOT_IN_DIR=/lightscratch/users/$GASPAR/mmore-data/in
```

### Directory Structure Initialization

Create the required directory structure on the persistent volume:

```bash
mkdir -p /lightscratch/users/$GASPAR/mmore-data/in
mkdir -p /lightscratch/users/$GASPAR/mmore-data/out
mkdir -p /lightscratch/users/$GASPAR/mmore-data/out/db
mkdir -p /lightscratch/users/$GASPAR/mmore-data/out/process/outputs/images
mkdir -p /lightscratch/users/$GASPAR/mmore-data/in/sample_data/
mkdir -p /lightscratch/users/$GASPAR/.cache
```

### Interactive Development Session

For development, debugging, or manual operations, start an interactive session (replace `<group-id>` with your actual group ID):

```bash
runai submit swissaimmore \
  --image <image> \
  --node-pool h100 \
  --pvc light-scratch:/lightscratch \
  --gpu 1 \
  --run-as-gid <group-id> \
  --preemptible \
  --attach \
  --interactive \
  --tty \
  --command /bin/bash
```

This provides a direct terminal access to the container.

### Production Pipeline Execution

For production workloads, submit jobs that run specific pipeline stages:

#### 1. Document Processing

Process raw documents and extract multimodal content (replace `<group-id>` with your actual group ID):

```bash
runai submit \
  --name swissaimmore-process \
  --image <image> \
  --backoff-limit 0 \
  --pvc light-scratch:/lightscratch \
  --run-as-gid <group-id> \
  --node-pool h100 \
  --gpu 1 \
  -e ROOT_IN_DIR=/lightscratch/users/$GASPAR/mmore-data/in \
  -e ROOT_OUT_DIR=/lightscratch/users/$GASPAR/mmore-data/out \
  -e XDG_CACHE_HOME=/lightscratch/users/$GASPAR/.cache \
  -e HF_HOME=/lightscratch/users/$GASPAR/.cache/huggingface \
  -e TORCH_HOME=/lightscratch/users/$GASPAR/.cache/torch \
  --command "python3 -m mmore process --config-file production-config/process/config.yaml"
```

#### 2. Post-processing

Clean and structure the extracted data:

```bash
runai submit \
  --name swissaimmore-postprocess \
  --image <image> \
  --backoff-limit 0 \
  --pvc light-scratch:/lightscratch \
  --run-as-gid <group-id> \
  --node-pool h100 \
  --gpu 1 \
  -e ROOT_IN_DIR=/lightscratch/users/$GASPAR/mmore-data/in \
  -e ROOT_OUT_DIR=/lightscratch/users/$GASPAR/mmore-data/out \
  -e XDG_CACHE_HOME=/lightscratch/users/$GASPAR/.cache \
  -e HF_HOME=/lightscratch/users/$GASPAR/.cache/huggingface \
  -e TORCH_HOME=/lightscratch/users/$GASPAR/.cache/torch \
  --command "python3 -m mmore postprocess --config-file production-config/postprocessor/config.yaml --input-data /lightscratch/users/$GASPAR/mmore-data/out/process/outputs/merged/merged_results.jsonl"
```

#### 3. Vector Indexing

Create searchable vector indexes:

```bash
runai submit \
  --name swissaimmore-index \
  --image <image> \
  --backoff-limit 0 \
  --pvc light-scratch:/lightscratch \
  --run-as-gid 84257 \
  --node-pool h100 \
  --gpu 1 \
  -e ROOT_IN_DIR=/lightscratch/users/$GASPAR/mmore-data/in \
  -e ROOT_OUT_DIR=/lightscratch/users/$GASPAR/mmore-data/out \
  -e XDG_CACHE_HOME=/lightscratch/users/$GASPAR/.cache \
  -e HF_HOME=/lightscratch/users/$GASPAR/.cache/huggingface \
  -e TORCH_HOME=/lightscratch/users/$GASPAR/.cache/torch \
  --command "python3 -m mmore index --config-file production-config/index/config.yaml --documents-path /lightscratch/users/$GASPAR/mmore-data/out/postprocessor/outputs/merged/final_pp.jsonl"
```

#### 4. RAG Service Deployment

Deploy the retrieval API service:

```bash
runai submit \
  --name swissaimmore-rag \
  --image <image> \
  --backoff-limit 0 \
  --pvc light-scratch:/lightscratch \
  --run-as-gid <group-id> \
  --node-pool h100 \
  --gpu 1 \
  -e ROOT_IN_DIR=/lightscratch/users/$GASPAR/mmore-data/in \
  -e ROOT_OUT_DIR=/lightscratch/users/$GASPAR/mmore-data/out \
  -e HF_TOKEN=$HF_TOKEN \
  -e XDG_CACHE_HOME=/lightscratch/users/$GASPAR/.cache \
  -e HF_HOME=/lightscratch/users/$GASPAR/.cache/huggingface \
  -e TORCH_HOME=/lightscratch/users/$GASPAR/.cache/torch \
  --command "python3 -m mmore live-retrieval --config-file production-config/retriever_api/config.yaml"
```

### Port-forwarding to access locally

Use `runai port-forward swissaimmore-rag 8080:8080` to access the service locally!
