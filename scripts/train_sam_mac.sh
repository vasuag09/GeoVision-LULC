#!/bin/bash

# Setup for SAM training on Mac M4 Pro
# Ensure you are in the project root

echo "🚀 Starting SAM Fine-Tuning on Local Mac (MPS)..."

uv run scripts/train.py \
    --config configs/sam_mac.yaml
