import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

from src.data.dataset import Sen2LULCDataset
from src.data.transforms import get_val_transforms
from src.models.builder import build_model
from src.utils.visualization import plot_qualitative_results, plot_confusion_matrix

def main():
    # Setup
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/visuals", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Classes
    classes = ["Water", "Dense Forest", "Sparse Forest", "Barren Land", "Built Up", "Agriculture Land", "Fallow Land"]
    
    # 1. Models to Evaluate
    models_to_eval = [
        {
            "name": "DeepLabV3+",
            "config": "configs/deeplabv3plus.yaml",
            "checkpoint": "checkpoints_deeplabv3plus/best_model_deeplabv3+.pt",
            "mIoU": 0.4243
        },
        {
            "name": "SegFormer",
            "config": "configs/segformer.yaml",
            "checkpoint": "checkpoints_segformer/best_model_segformer.pt",
            "mIoU": 0.4614
        }
    ]
    
    # 2. Data
    # For artifacts, we only need a small set
    data_dir = "src/data/SEN-2 LULC"
    # We use a 10% subset for fast evaluation of artifacts
    val_dataset = Sen2LULCDataset(data_dir, split="val", transforms=get_val_transforms({'img_size': 128}), subset_fraction=0.1, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 3. Quantitative Comparison
    df_benchmark = pd.DataFrame(models_to_eval)
    df_benchmark = df_benchmark[["name", "mIoU"]]
    # Add model sizes manually based on file size checks
    df_benchmark["Size (MB)"] = [476.1, 44.8]
    df_benchmark.to_csv(f"{results_dir}/benchmark_comparison.csv", index=False)
    print("\n[✓] Generated benchmark_comparison.csv")
    
    # 4. Generate Visualizations for SegFormer (Best Model)
    best_model_info = models_to_eval[1] # SegFormer
    with open(best_model_info["config"], 'r') as f:
        config = yaml.safe_load(f)
    
    model = build_model(config).to(device)
    checkpoint = torch.load(best_model_info["checkpoint"], map_location=device, weights_only=False)
    
    # Align state dict keys
    state_dict = checkpoint['model_state_dict']
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    # Strategy: If the checkpoint keys are prefixed with 'model.' but the target model is not,
    # OR vice versa, adjust accordingly.
    if not all(k in model_keys for k in ckpt_keys):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.') and (k[6:] in model_keys):
                new_state_dict[k[6:]] = v
            elif (f"model.{k}" in model_keys):
                new_state_dict[f"model.{k}"] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
            
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[✓] Loaded {best_model_info['name']} for visualization")
    
    # Pick a few diverse samples
    all_imgs, all_masks, all_preds = [], [], []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Convert to CPU/Numpy
            img_np = images.cpu().permute(0, 2, 3, 1).numpy()
            # De-normalize for visualization
            img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            all_imgs.append(img_np)
            all_masks.append(masks.numpy())
            all_preds.append(preds.cpu().numpy())
            
            if len(all_imgs) >= 2: # Get 16 samples
                break
                
    imgs = np.concatenate(all_imgs, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    
    # A. Qualitative Grid
    plot_qualitative_results(imgs, masks, preds, classes, save_path=f"{results_dir}/qualitative_showcase.png", num_samples=6)
    print(f"[✓] Generated research qualitative showcase: qualitative_showcase.png")
    
    # B. Confusion Matrix
    # We compute CM over the 10% subset
    y_true, y_pred = [], []
    print("Computing Confusion Matrix...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            outputs = model(images)
            ps = torch.argmax(outputs, dim=1).cpu().numpy()
            target = masks.numpy()
            
            # Mask out ignore index (-1 mapped to 255)
            mask = (target != 255) & (target < 7)
            y_true.extend(target[mask])
            y_pred.extend(ps[mask])
            
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plot_confusion_matrix(cm, classes, save_path=f"{results_dir}/confusion_matrix.png")
    print(f"[✓] Generated confusion matrix: confusion_matrix.png")

    # C. Model Comparison Chart (mIoU vs Size)
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    for i, row in df_benchmark.iterrows():
        plt.scatter(row["Size (MB)"], row["mIoU"], s=200, label=row["name"])
        plt.text(row["Size (MB)"]+10, row["mIoU"], row["name"], fontsize=12)
    
    plt.xlabel("Model Size (MB)")
    plt.ylabel("mIoU")
    plt.title("Model Efficiency: Accuracy vs. Size")
    plt.grid(True)
    plt.savefig(f"{results_dir}/efficiency_comparison.png", dpi=300)
    print(f"[✓] Generated efficiency chart: efficiency_comparison.png")

if __name__ == "__main__":
    main()
