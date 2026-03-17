# Google Colab Setup Guide for GeoVision-LULC

To shrink the 55-hour training time down to just a few hours, you can run this repository on a free **Google Colab T4 GPU** (or premium A100 GPU).

## Step 1: Zip the Project (Locally)
Google Drive is extremely slow at uploading 200,000+ tiny image patches individually. You must compress the folder first.
Open your MacOS terminal and run this from outside your project folder:
```bash
cd "/Users/vasuagrawal/Desktop/Machine Learning/Projects"
zip -r GeoVision-LULC.zip GeoVision-LULC -x "GeoVision-LULC/.venv/*" -x "GeoVision-LULC/.git/*"
```
*(This zips the 4.1GB dataset and code, but excludes the massive virtual environment).*

## Step 2: Upload to Google Drive
Upload `GeoVision-LULC.zip` into the root of your Google Drive.

## Step 3: Open Colab and Mount Drive
1. Go to [colab.research.google.com](https://colab.research.google.com/) and create a **New Notebook**.
2. Go to **Runtime > Change runtime type** -> Hardware accelerator: **T4 GPU** (or A100).
3. Paste and run this cell to mount your drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 4: Extract and Install (Colab Cell 2)
Copy and run this cell to unzip your project into the fast local Colab storage and install `uv`:
```python
# Unzip to Colab's high-speed local disk
!cp "/content/drive/MyDrive/GeoVision-LULC.zip" /content/
!unzip -q /content/GeoVision-LULC.zip -d /content/

# Install the exact dependencies using uv
!pip install uv

%cd /content/GeoVision-LULC
!uv venv --clear
!uv pip install -e .
```

## Step 5: Start Training! (Colab Cell 3)
Because you are now on a powerful NVIDIA CUDA GPU, I've already re-enabled `mixed_precision = true` and pumped the batch size up to `64` natively in `configs/deeplabv3plus.yaml`!
```python
%cd /content/GeoVision-LULC

# Run the 80 epochs! (Using explicitly pathed uv)
!uv run scripts/train.py --config configs/deeplabv3plus.yaml
```

## Step 6: Save the Checkpoint Back to Drive (Colab Cell 4)
When it finishes, copy your fully trained models back to your Drive so you don't lose them when the Colab shuts down:
```python
!cp -r /content/GeoVision-LULC/checkpoints_deeplabv3plus "/content/drive/MyDrive/"
!cp -r /content/GeoVision-LULC/results "/content/drive/MyDrive/"
```
