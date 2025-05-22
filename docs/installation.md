# Installation

To install `mmore`, run the following:

1. Clone the repository
   ```bash
   git clone https://github.com/swiss-ai/mmore
   ```

2. Install the package
   ```bash
   pip install -e .
   ```

### Alternative #1: `uv`

##### Step 1: Install system dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6 chromium-browser libnss3 \
  libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 \
  libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice \
  libpango-1.0-0 libpangoft2-1.0-0 weasyprint
```

##### Step 2: Install `uv`

Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for detailed instructions.
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

##### Step 3: Clone this repository

```bash
git clone https://github.com/swiss-ai/mmore
cd mmore
```

##### Step 4: Install project and dependencies

```bash
uv sync
```

For CPU-only installation, use:

```bash
uv sync --extra cpu
```

##### Step 5: Run a test command

Activate the virtual environment before running commands:

```bash
source .venv/bin/activate
```
### Alternative #2: `Docker`

**Note:** For manual installation without Docker, refer to the section below.

##### Step 1: Install Docker

Follow the official [Docker installation guide](https://docs.docker.com/get-started/get-docker/).

##### Step 2: Build the Docker image

```bash
sudo docker build . --tag mmore
```

To build for CPU-only platforms (results in a smaller image size):

```bash
sudo docker build --build-arg PLATFORM=cpu -t mmore .
```

##### Step 3: Start an interactive session

```bash
sudo docker run -it -v ./examples:/app/examples mmore
```

*Note:* The `examples` folder is mapped to `/app/examples` inside the container, corresponding to the default path in `examples/process/config.yaml`.
