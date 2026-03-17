#!/bin/bash
set -e

echo "Starting sequential GeoVision training pipeline..."
echo "This will train the models one after the other to prevent Mac memory OOMs."

echo ""
echo "========================================"
echo "1. Training UNet Baseline"
echo "========================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run scripts/train.py --config configs/default.yaml

echo ""
echo "========================================"
echo "2. Training DeepLabV3+ Baseline"
echo "========================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run scripts/train.py --config configs/deeplabv3plus.yaml

echo ""
echo "========================================"
echo "3. Training SegFormer Transformer"
echo "========================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run scripts/train.py --config configs/segformer.yaml

echo ""
echo "========================================"
echo "4. Training Swin-UNet Transformer"
echo "========================================"
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run scripts/train.py --config configs/swin_unet.yaml

echo ""
echo "========================================"
echo "🎉 All model training sequences have been completed successfully!"
echo "========================================"
