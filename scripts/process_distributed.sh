#!/bin/bash

# Default values
INPUT_FOLDER=""
MMORE_FOLDER=$(pwd)
RANK=0

# Helper function to show usage
usage() {
  echo "Usage: $0 --mmore-folder <path> --input_folder <path> [--rank <value>]"
  echo ""
  echo "Required arguments:"
  echo "  --input_folder  Path to the test file or folder."
  echo ""
  echo "Optional arguments:"
  echo "  --mmore-folder       Path to the mmore folder."
  echo "  --rank                 Node rank (default: 0)."
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mmore-folder)
      MMORE_FOLDER="$2"
      shift 2
      ;;
    --input_folder)
      INPUT_FOLDER="$2"
      shift 2
      ;;
    --rank)
      RANK="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

# Check required arguments
if [[ -z "$MMORE_FOLDER" || -z "$INPUT_FOLDER" ]]; then
  echo "Error: Missing required arguments."
  usage
fi

# Update and install dependencies
echo "Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    nano curl ffmpeg libsm6 libxext6 chromium-browser libnss3 libgconf-2-4 libxi6 libxrandr2 \
    libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxrender1 libasound2 libatk1.0-0 \
    libgtk-3-0 libreoffice libjpeg-dev

# Install Rye
echo "Setting up Rye"
export RYE_HOME="/opt/rye"
export PATH="$RYE_HOME/shims:$PATH"
if [ ! -d "$RYE_HOME" ]; then
  curl -sSf https://rye.astral.sh/get | RYE_TOOLCHAIN_VERSION="3.11" RYE_INSTALL_OPTION="--yes" bash
  echo 'export RYE_HOME="/opt/rye"' >> ~/.bashrc
  echo 'export PATH="$RYE_HOME/shims:$PATH"' >> ~/.bashrc
fi

# Navigate to the project directory
echo "Navigating to mmore folder: $MMORE_FOLDER"
cd "$MMORE_FOLDER" || { echo "Directory $MMORE_FOLDER does not exist! Exiting."; exit 1; }

# Sync Rye to install dependencies
echo "Syncing Rye (installing dependencies)"
rye sync

# Configure environment variables
echo "Setting up environment variables"
export PATH="/.venv/bin:$PATH"
export DASK_DISTRIBUTED__WORKER__DAEMON=False

# Start the worker
echo "Starting the worker..."
source .venv/bin/activate
dask config set distributed.worker.daemon false

# Start the Dask scheduler if the current node is the MASTER (rank 0)
if [ "$RANK" -eq 0 ]; then
  echo "Starting the scheduler because it is the MASTER node (rank 0)"
  dask scheduler --scheduler-file scheduler-file.json &
fi

# Start the Dask worker
echo "Starting the worker of every node"
dask worker --scheduler-file scheduler-file.json &

# Run the end-to-end test if the current node is the MASTER (rank 0)
if [ "$RANK" -eq 0 ]; then
  echo "Running the end-to-end test in the MASTER node (rank 0)"
  python run_process.py --config_file examples/process_config.yaml
fi
