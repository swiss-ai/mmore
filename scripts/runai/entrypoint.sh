#!/bin/bash

EXECUTOR="python"
SCRIPT="run_process.py"
while getopts e:s:c:p: flag
do
    case "${flag}" in
        e) EXECUTOR="${OPTARG}";;
        s) SCRIPT="${OPTARG}";;
        c) CONFIG="${OPTARG}";;
        p) REPO_PATH="${OPTARG}";;
    esac
done

# Going to repo dir
if [ -z "$REPO_PATH" ]; then
    REPO_PATH="/mnt/mlo/scratch/homes/$(whoami)/mmore"
fi
cd $REPO_PATH

# Loading env vars
set -o allexport
source .env
set +o allexport

# TODO: Update when final version is released (Install libraries)
#bash scripts/setup.sh
pip install -r rag_requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Start time: $(date)"

# Launch script
echo "Running: $EXECUTOR $SCRIPT --config-file $CONFIG"
$EXECUTOR $SCRIPT --config-file $CONFIG

echo "End time: $(date)"