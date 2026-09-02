#!/bin/bash
# DexteraAI setup script
set -e

echo "[DexteraAI] Setting up environment..."

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt || true

# Install frontend dependencies
cd apps/web && npm install && cd ../..

# Run initial migrations or setup
# (Add any additional setup steps here)

echo "[DexteraAI] Setup complete!"
