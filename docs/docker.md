### Installation Option 3: Docker

**Note:** For manual installation without Docker, refer to the section below.

#### Step 1: Install Docker

Follow the official [Docker installation guide](https://docs.docker.com/get-started/get-docker/).

#### Step 2: Build the Docker image

```bash
docker build . --tag mmore
```

To build for CPU-only platforms (results in a smaller image size):

```bash
docker build --build-arg PLATFORM=cpu -t mmore .
```

#### Step 3: Start an interactive session

```bash
docker run -it -v ./test_data:/app/test_data mmore
```

*Note:* The `test_data` folder is mapped to `/app/test_data` inside the container, corresponding to the default path in `examples/process_config.yaml`.