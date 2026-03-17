# Project: GeoVision-LULC

## Objective

Develop a state-of-the-art deep learning system for Land Use Land Cover (LULC) semantic segmentation using Sentinel-2 satellite imagery.

The system should train and benchmark multiple segmentation architectures on the Sen-2 LULC dataset and produce geospatial LULC maps.

## Dataset

Sen-2 LULC Dataset

Characteristics:

- 213,761 RGB satellite image patches
- Resolution: 10m
- Patch size: 64x64
- 7 classes:
  - Water
  - Dense Forest
  - Sparse Forest
  - Barren Land
  - Built Up
  - Agriculture Land
  - Fallow Land

Dataset structure:

train_images
train_masks
val_images
val_masks
test_images
test_masks

Each image has a corresponding pixel mask.

## Tasks

### 1 Dataset Pipeline

Implement:

- PyTorch Dataset class
- Augmentations
- Patch reconstruction utilities
- Class balancing

Augmentations:

- Random rotation
- Horizontal flip
- Random crop
- Color jitter

## Models to Implement

Baseline CNN models:

1 UNet
2 DeepLabV3+
3 SegNet

Transformer models:

4 SegFormer
5 Swin-UNet

Foundation model integration:

6 SAM fine tuning
7 DINOv2 feature encoder

## Training Setup

Loss functions:

- Cross entropy
- Dice loss
- Focal loss

Metrics:

- mIoU
- pixel accuracy
- F1 score

Training details:

- PyTorch
- mixed precision
- early stopping
- learning rate scheduler

## Experiments

Experiment groups:

1 CNN baseline comparison
2 Transformer comparison
3 CNN vs Transformer
4 SAM assisted segmentation
5 multi-scale training

Produce tables:

Model | mIoU | F1 | Accuracy | Params

## Explainability

Implement:

GradCAM visualization

Show:

- feature attention
- class activation

## Geospatial Output

Generate LULC maps:

- reconstruct satellite tiles
- export GeoTIFF outputs
- overlay predictions

## Visualization

Build Streamlit dashboard:

- upload satellite tile
- generate LULC map
- view prediction overlays
- download GeoTIFF

## Research Paper Output

Generate:

- figures
- ablation tables
- confusion matrices
- segmentation visualizations

## Code Quality

Requirements:

- modular PyTorch code
- reproducible experiments
- configuration files
- Docker support