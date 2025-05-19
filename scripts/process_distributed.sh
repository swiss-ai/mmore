#!/bin/bash

# Default values
CONFIG_PATH=""
MMORE_FOLDER=$(pwd)
RANK=""

# Helper function to show usage
usage() {
  echo "Usage: $0 --mmore-folder <path> --config-path <path> --rank <value>"
  echo ""
  echo "Required arguments:"
  echo "  --mmore-folder       Absolute path to the mmore folder."
  echo "  --config_path    Absolute path to the config.yaml file."
  echo "  --rank                                       Node rank."
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mmore-folder)
      MMORE_FOLDER="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
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
if [[ -z "$MMORE_FOLDER" || -z "$CONFIG_PATH" || -z "$RANK" ]]; then
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
echo "Setting up UV"
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the project directory
echo "Navigating to mmore folder: $MMORE_FOLDER"
cd "$MMORE_FOLDER" || { echo "Directory $MMORE_FOLDER does not exist! Exiting."; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

# Sync Rye to install dependencies
echo "Syncing UV (installing dependencies)"
pip install -e '.[process]'

# Extract the distributed configuration from the YAML file
distributed=$(grep -A3 'dispatcher_config:' "$CONFIG_PATH" | grep 'distributed:' | awk '{print $2}')
scheduler_file=$(grep 'scheduler_file:' "$CONFIG_PATH" | awk '{print $2}')


# Configure environment variables
echo "Setting up environment variables"
export PATH="/.venv/bin:$PATH"
export DASK_DISTRIBUTED__WORKER__DAEMON=False

# Dask part of the script
source .venv/bin/activate

if [ "$distributed" = "true" ]; then
  pip list | grep dask 
  echo "Distributed mode enabled"
  # Start the Dask scheduler if the current node is the MASTER (rank 0)
  if [ "$RANK" -eq 0 ]; then
    echo "Starting the scheduler because it is the MASTER node (rank 0)"
    dask -h
    dask scheduler --scheduler-file "$scheduler_file" &
  fi

  # Start the Dask worker
  echo "Starting the worker of every node"
  dask worker --scheduler-file "$scheduler_file" &
fi


# Run the end-to-end test if the current node is the MASTER (rank 0)
if [ "$RANK" -eq 0 ]; then
  echo "Running the end-to-end test in the MASTER node (rank 0)"
  echo "Command to execute: python \"$MMORE_FOLDER/src/mmore/run_process.py\" --config_file \"$CONFIG_PATH\""
  echo "Should maybe exit here and wait until all the workers are ready!"
  echo "Type 'go' to execute the command, or type 'exit' to stop and run it manually later."

  # waiting for the user to type 'go' or 'exit'
  while true; do
    read -r user_input
    if [ "$user_input" = "go" ]; then
      python "$MMORE_FOLDER/src/mmore/run_process.py" --config_file "$CONFIG_PATH"
      break
    elif [ "$user_input" = "exit" ]; then
      echo "Exiting without running the command. You can run it manually later:"
      echo "python \"$MMORE_FOLDER/src/mmore/run_process.py\" --config_file \"$CONFIG_PATH\""
      exit 0
    else
      echo "Invalid input. Type 'go' to run the command or 'exit' to stop."
    fi
  done
fi
