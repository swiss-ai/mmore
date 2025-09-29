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

*Running on RCP:* you can specify a `USER_UID` and a `USER_GID` variable. Set it to your RCP user ID and group ID to run it there.

##### Step 3: Start an interactive session

```bash
sudo docker run --gpus all -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore
```

For CPU-only platforms:
```bash
sudo docker run -it -v ./examples:/app/examples -v ./.cache:/mmoreuser/.cache mmore
```

> [!WARNING]
> You may need the Nvidia toolkit so the containers can access your GPUs.
> Read [this tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if something breaks here!
>
> Configure the production repository:
>
> ```sh
> curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
>   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
>   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
>   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
> ```
>
> ```sh
> sudo apt update
> sudo apt install -y nvidia-container-toolkit
> ```
>
> Modify the Docker daemon to use Nvidia:
>
> ```sh
> sudo nvidia-ctk runtime configure --runtime=docker
> sudo systemctl restart docker
> ```
>
> You can now use `docker run --gpus all`!

*Note:* The `examples` folder is mapped to `/app/examples` inside the container, corresponding to the default path in `examples/process/config.yaml`.
