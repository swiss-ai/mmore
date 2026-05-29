#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package lists
sudo apt update

# Install system dependencies
sudo apt install -y ffmpeg libsm6 libxext6 libnss3 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice

EXTRA="${1:-process}"
case "$EXTRA" in
    process|colvision|rag|api|websearch|all) ;;
    *)
        echo "Unknown extra '$EXTRA'. Valid: process, colvision, rag, api, websearch, all." >&2
        exit 1
        ;;
esac

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrcuv sync
uv venv
uv sync --extra "$EXTRA"
source .venv/bin/activate