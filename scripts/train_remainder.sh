#!/bin/bash
set -e

echo "Waiting for any currently running UNet baseline training to complete..."
echo "This will poll every 60 seconds."

while pgrep -f "train.py --config configs/default.yaml" > /dev/null; do
    sleep 60
done

echo "UNet training finished or stopped. Proceeding with the queue..."

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
