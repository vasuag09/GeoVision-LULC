import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay

def plot_training_curves(train_losses, val_losses, val_mious, val_f1s, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss Curve
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # mIoU Curve
    axes[1].plot(val_mious, label="Val mIoU", color='green')
    axes[1].set_title("Validation mean IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].legend()
    axes[1].grid(True)
    
    # F1 Score
    axes[2].plot(val_f1s, label="Val F1", color='purple')
    axes[2].set_title("Validation F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    # Normalize by row
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_benchmark_table(model_results, save_path="benchmark.csv"):
    """
    model_results: list of dicts. 
    Format: [{"Model": "UNet", "mIoU": 0.85, "F1": 0.86, "Accuracy": 0.90, "Params (M)": 24.5}, ...]
    """
    df = pd.DataFrame(model_results)
    df = df.sort_values(by="mIoU", ascending=False).reset_index(drop=True)
    print("\n--- Benchmark Table ---")
    print(df.to_markdown())
    
    if save_path.endswith('.csv'):
        df.to_csv(save_path, index=False)
    elif save_path.endswith('.tex'):
        df.to_latex(save_path, index=False)
        
    return df

def plot_qualitative_results(images, masks, preds, classes, save_path="qualitative.png", num_samples=5):
    """
    images: (B, H, W, 3) - expected in [0, 1] float
    masks: (B, H, W) - integer indices
    preds: (B, H, W) - integer indices
    """
    B = min(len(images), num_samples)
    fig, axes = plt.subplots(B, 3, figsize=(15, 5 * B))
    
    # Use distinct colors from tab10 or similar
    from matplotlib import cm as matplotlib_cm
    cmap = matplotlib_cm.get_cmap("tab10", len(classes))
    colors = (cmap(np.arange(len(classes)))[:, :3] * 255).astype(np.uint8)
    
    if len(axes.shape) == 1:
         axes = np.expand_dims(axes, 0)
         
    for i in range(B):
        # Image
        # Ensure image is in [0, 1] for imshow if float, or [0, 255] if uint8
        img_disp = images[i]
        if img_disp.max() > 1.0 and img_disp.dtype != np.uint8:
            img_disp = img_disp / 255.0
            
        axes[i, 0].imshow(img_disp)
        axes[i, 0].set_title("Satellite Image")
        axes[i, 0].axis('off')
        
        # Ground Truth
        # Initialize as uint8 for consistent coloring
        h, w = images[i].shape[:2]
        colorized_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for c_idx in range(len(classes)):
            colorized_mask[masks[i] == c_idx] = colors[c_idx]
            
        axes[i, 1].imshow(colorized_mask)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')
        
        # Prediction
        colorized_pred = np.zeros((h, w, 3), dtype=np.uint8)
        for c_idx in range(len(classes)):
            colorized_pred[preds[i] == c_idx] = colors[c_idx]
            
        axes[i, 2].imshow(colorized_pred)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
