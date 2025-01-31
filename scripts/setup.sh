#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package lists
sudo apt update

# Install system dependencies
sudo apt install -y ffmpeg libsm6 libxext6 chromium-browser libnss3 libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrcuv sync
source .venv/bin/activate

python -m ensurepip --default-pip 
pip install -r requirements.txt 
pip install -r rag_requirements.txt 
pip install -r graphrag_requirements.txt 
huggingface-cli login

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/mloscratch/homes/ordonnea/mmore/src/ 