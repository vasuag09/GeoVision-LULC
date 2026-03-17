import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader, Subset
from src.data.dataset import Sen2LULCDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.builder import build_model
from tqdm import tqdm

def get_basic_transforms(img_size=64):
    # Step 4: Disable heavy augmentations temporarily
    return A.Compose([
        A.Resize(width=img_size, height=img_size, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

def step6_compute_class_distribution(dataset, num_classes=7):
    print("\n--- Step 6: Computing Class Distribution ---")
    class_counts = np.zeros(num_classes, dtype=np.int64)
    # We will sample 1000 images for speed or all of them. Let's do 1000 for sanity if it's too large, or all if we can.
    # The dataset has 200k images. Doing all takes time. Let's sample 5000 images uniformly.
    indices = np.linspace(0, len(dataset)-1, 5000, dtype=int)
    for idx in tqdm(indices, desc="Computing distribution (subset)"):
        _, mask = dataset[idx]
        mask_np = mask.numpy()
        counts = np.bincount(mask_np.flatten(), minlength=num_classes)
        class_counts += counts[:num_classes]

    total_pixels = class_counts.sum()
    print("Class Pixel Distribution:")
    for c in range(num_classes):
        print(f"Class {c}: {class_counts[c]} pixels ({class_counts[c]/total_pixels*100:.2f}%)")
    print("-" * 40)


def step_3_4_5_overfit_sanity_test(config):
    print("\n--- Step 3: Overfit Sanity Test ---")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = config['data']['dataset_dir']
    img_size = config['data']['img_size']
    num_classes = config['data']['num_classes']

    # Step 4 transforms applied
    transforms = get_basic_transforms(img_size)
    dataset = Sen2LULCDataset(data_dir, split='train', transforms=transforms)

    # Step 6
    step6_compute_class_distribution(dataset, num_classes)

    # Step 3 Overfit single image
    # Find an image that has multiple classes for a better test
    target_idx = 0
    for i in range(100):
        _, mask = dataset[i]
        if len(torch.unique(mask)) >= 3:
            target_idx = i
            break
            
    img, mask = dataset[target_idx]
    
    img_batch = img.unsqueeze(0).to(device)
    mask_batch = mask.unsqueeze(0).to(device)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    model.train()
    epochs = 300
    pbar = tqdm(range(epochs), desc="Overfitting 1 batch")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(img_batch)
        loss = criterion(outputs, mask_batch)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == mask_batch).sum().item()
        total = mask_batch.numel()
        acc = correct / total
        
        # approximate mIoU
        intersection = torch.logical_and(preds == mask_batch, mask_batch < 255).sum()
        union = torch.logical_or(preds == mask_batch, mask_batch < 255).sum()
        miou = (intersection / (union + 1e-6)).item()
        
    print(f"\nFinal Overfit Metrics - Loss: {loss.item():.4f}, Acc: {acc:.4f}, mIoU: {miou:.4f}")

    # Step 5: Visualize predictions
    print("\n--- Step 5: Visualizing Predictions ---")
    model.eval()
    with torch.no_grad():
        outputs = model(img_batch)
        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

    img_np = img.permute(1, 2, 0).cpu().numpy()
    # Unnormalize for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_disp = std * img_np + mean
    img_disp = np.clip(img_disp * 255, 0, 255).astype(np.uint8)

    mask_np = mask.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_disp)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(mask_np, vmin=0, vmax=6, cmap='tab10')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(preds, vmin=0, vmax=6, cmap='tab10')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    os.makedirs('results/debug', exist_ok=True)
    plt.savefig('results/debug/overfit_visualization.png')
    print("Visualization saved to results/debug/overfit_visualization.png")
    print("Done!")

if __name__ == "__main__":
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    step_3_4_5_overfit_sanity_test(config)
