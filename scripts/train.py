import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import yaml
import torch
from torch.utils.data import DataLoader
import argparse

from src.data.dataset import Sen2LULCDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.utils import calculate_class_weights
from src.models.builder import build_model
from src.training.losses import get_loss
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoVision-LULC Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Needs to be mocked or points to real dataset
    data_dir = config['data']['dataset_dir']
    img_size = config['data']['img_size']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    # Ensure dirs exist simply to prevent immediate crashing if mocking
    os.makedirs(f"{data_dir}/train_images", exist_ok=True)
    os.makedirs(f"{data_dir}/train_masks", exist_ok=True)
    os.makedirs(f"{data_dir}/val_images", exist_ok=True)
    os.makedirs(f"{data_dir}/val_masks", exist_ok=True)
    
    subset_fraction = config['data'].get('subset_fraction', 1.0)
    seed = config.get('seed', 42)
    
    train_dataset = Sen2LULCDataset(data_dir, split="train", transforms=get_train_transforms(config['data']), 
                                    subset_fraction=subset_fraction, seed=seed)
    val_dataset = Sen2LULCDataset(data_dir, split="val", transforms=get_val_transforms(config['data']),
                                  subset_fraction=subset_fraction, seed=seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )
    
    model = build_model(config).to(device)
    
    # Calculate class weights if needed
    if config['training'].get('loss') == 'cross_entropy':
        if len(train_dataset) > 0:
            config['training']['class_weights'] = calculate_class_weights(train_loader, config['data']['num_classes']).tolist()
            
    criterion = get_loss(config).to(device)
    
    # Handle nested optimizer configuration from advanced YAML
    opt_config = config.get('optimizer', {})
    lr = opt_config.get('learning_rate', config['model'].get('learning_rate', 0.001))
    wd = opt_config.get('weight_decay', config['model'].get('weight_decay', 0.0001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Handle nested scheduler configuration
    sched_config = config.get('scheduler', {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device)
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    elif args.resume:
        print(f"Warning: Resume checkpoint not found at '{args.resume}'. Starting from scratch.")
    
    trainer.train()

if __name__ == "__main__":
    main()
