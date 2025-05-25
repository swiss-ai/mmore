#!/bin/bash

# Default values
CONFIG_PATH=""
RANK=""

# Helper function to show usage
usage() {
  echo "Usage: $0 --mmore-folder <path> --config-file <path> --rank <value>"
  echo ""
  echo "Required arguments:"
  echo "  --config-file    Absolute path to the config.yaml file."
  echo "  --rank                                       Node rank."
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config-file)
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
if [[ -z "$CONFIG_PATH" || -z "$RANK" ]]; then
  echo "Error: Missing required arguments."
  usage
fi

# Update and install dependencies
echo "Updating system and installing dependencies..."

# Extract the distributed configuration from the YAML file
distributed=$(grep -A3 'dispatcher_config:' "$CONFIG_PATH" | grep 'distributed:' | awk '{print $2}')
scheduler_file=$(grep 'scheduler_file:' "$CONFIG_PATH" | awk '{print $2}')


# Configure environment variables
echo "Setting up environment variables"
export DASK_DISTRIBUTED__WORKER__DAEMON=False

# Dask part of the script

if [ "$distributed" = "true" ]; then
  echo "Distributed mode enabled"
  # Start the Dask scheduler if the current node is the MASTER (rank 0)
  if [ "$RANK" -eq 0 ]; then
    echo "Starting the scheduler because it is the MASTER node (rank 0)"
    dask scheduler --scheduler-file "$scheduler_file" &> dask_scheduler.log &
    SCHEDULER_PID=$!
  fi

  # Start the Dask worker
  echo "Starting the worker of every node"
  dask worker --scheduler-file "$scheduler_file" &> "dask_scheduler_worker_$RANK.log" &
fi


# Run the end-to-end test if the current node is the MASTER (rank 0)
if [ "$RANK" -eq 0 ]; then
  echo "Running the end-to-end test in the MASTER node (rank 0)"
  echo "Command to execute: python -m mmore process --config-file \"$CONFIG_PATH\""
  echo "Should maybe exit here and wait until all the workers are ready!"
  echo "Type 'go' to execute the command, or type 'exit' to stop and run it manually later."

  # waiting for the user to type 'go' or 'exit'
  while true; do
    read -r user_input
    if [ "$user_input" = "go" ]; then
      echo "Starting processing"
      python -m mmore process --config-file "$CONFIG_PATH"
      break
    elif [ "$user_input" = "exit" ]; then
      echo "Exiting without running the command. You can run it manually later:"
      echo "python -m mmore process --config-file \"$CONFIG_PATH\""
      exit 0
    else
      echo "Invalid input. Type 'go' to run the command or 'exit' to stop."
    fi
  done

  kill -9 $SCHEDULER_PID
fi