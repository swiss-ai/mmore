#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package lists
sudo apt update

# Install system dependencies
sudo apt install -y ffmpeg libsm6 libxext6 chromium-browser libnss3 libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice

# Install Rye
curl -sSf https://rye.astral.sh/get | RYE_TOOLCHAIN_VERSION="3.11" RYE_INSTALL_OPTION="--yes" bash
echo "export PATH=\"\$HOME/.rye/bin:\$PATH\"" >> "$HOME/.bashrc"
source "$HOME/.rye/env"
source .venv/bin/activate
# Install Python dependencies   
rye sync