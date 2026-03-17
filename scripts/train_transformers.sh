#!/bin/bash
set -e

echo ""
echo "========================================"
echo "1. Training SegFormer Transformer"
echo "========================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run scripts/train.py --config configs/segformer.yaml

echo ""
echo "========================================"
echo "2. Training Swin-UNet Transformer"
echo "========================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run scripts/train.py --config configs/swin_unet.yaml

echo ""
echo "========================================"
echo "🎉 All transformer sequences have been completed successfully!"
echo "========================================"
