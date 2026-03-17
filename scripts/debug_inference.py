import torch
import os
import yaml
from src.models.builder import build_model
from src.data.dataset import Sen2LULCDataset
from src.data.transforms import get_val_transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("mps")

# Test Dataset
data_dir = "src/data/SEN-2 LULC"
dataset = Sen2LULCDataset(data_dir, split="test", transforms=get_val_transforms(64))
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load UNet
unet = build_model(config).to(device)
ckpt = torch.load("checkpoints/best_model_unet.pt", map_location=device, weights_only=False)
unet.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
unet.eval()

# Check parameter shapes or unique counts
print("Running diagnostic inference on test batch 0...")
for imgs, masks in loader:
    imgs = imgs.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        out = unet(imgs)
        preds = torch.argmax(out, dim=1)
        
    print(f"Mask uniques: {torch.unique(masks)}")
    print(f"Pred uniques: {torch.unique(preds)}")
    
    # Calculate simple accuracy
    correct = (preds == masks).sum().item()
    total = masks.numel()
    print(f"Batch Accuracy: {correct/total*100:.2f}%")
    break
