# Colab Patches — One Cell to Rule Them All

Every time you unzip `GeoVision-LULC.zip` in a fresh Colab runtime, paste this **single cell** and run it. It patches the source code and adds the new Transformer/SAM configurations.

---

## Cell 1: All-in-One Setup

```python
import textwrap, os

patches = {
    # ── PATCH 1/6: transforms.py ──
    "src/data/transforms.py": textwrap.dedent('''\
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import cv2

        def get_train_transforms(data_config):
            img_size = data_config.get('img_size', 64)
            aug_cfg = data_config.get('augmentation', {})
            
            transforms_list = [A.Resize(width=img_size, height=img_size, p=1.0)]
            
            if aug_cfg.get('horizontal_flip', False):
                transforms_list.append(A.HorizontalFlip(p=0.5))
                
            if aug_cfg.get('vertical_flip', False):
                transforms_list.append(A.VerticalFlip(p=0.5))
                
            if aug_cfg.get('normalize', True):
                transforms_list.append(
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    )
                )
                
            transforms_list.append(ToTensorV2())
            return A.Compose(transforms_list)


        def get_val_transforms(data_config):
            img_size = data_config.get('img_size', 64)
            aug_cfg = data_config.get('augmentation', {})
            
            transforms_list = [A.Resize(width=img_size, height=img_size, p=1.0)]
            
            if aug_cfg.get('normalize', True):
                transforms_list.append(
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    )
                )
                
            transforms_list.append(ToTensorV2())
            return A.Compose(transforms_list)

        def get_inference_transforms():
            return A.Compose(
                [
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
            )
    '''),

    # ── PATCH 2/6: losses.py ──
    "src/training/losses.py": textwrap.dedent('''\
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class DiceLoss(nn.Module):
            def __init__(self, smooth=1.0, ignore_index=255):
                super(DiceLoss, self).__init__()
                self.smooth = smooth
                self.ignore_index = ignore_index

            def forward(self, logits, targets):
                num_classes = logits.shape[1]
                probs = F.softmax(logits, dim=1)
                targets_long = targets.long()
                targets_one_hot = F.one_hot(torch.clamp(targets_long, 0, num_classes-1), num_classes=num_classes)
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
                mask = (targets != self.ignore_index).unsqueeze(1).float()
                probs = probs * mask
                targets_one_hot = targets_one_hot * mask
                dims = (0, 2, 3)
                intersection = torch.sum(probs * targets_one_hot, dims)
                cardinality = torch.sum(probs + targets_one_hot, dims)
                dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
                return 1.0 - torch.mean(dice_score)


        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.5, gamma=2.0, ignore_index=255):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ignore_index = ignore_index
                self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

            def forward(self, logits, targets):
                ce_loss = self.ce(logits, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                if self.ignore_index is not None:
                     mask = (targets != self.ignore_index).float()
                     focal_loss = focal_loss * mask
                     return focal_loss.sum() / mask.sum()
                return focal_loss.mean()

        class HybridLoss(nn.Module):
            def __init__(self, weight=None, ignore_index=255, ce_w=0.5, dice_w=0.5):
                super(HybridLoss, self).__init__()
                self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
                self.dice = DiceLoss(ignore_index=ignore_index)
                self.ce_w = ce_w
                self.dice_w = dice_w

            def forward(self, logits, targets):
                ce_loss = self.ce(logits, targets)
                dice_loss = self.dice(logits, targets)
                return self.ce_w * ce_loss + self.dice_w * dice_loss

        def get_loss(config):
            loss_cfg = config['training'].get('loss', 'cross_entropy')
            if isinstance(loss_cfg, dict):
                loss_name = loss_cfg.get('type', 'cross_entropy').lower()
            else:
                loss_name = loss_cfg.lower()
                loss_cfg = {}
            Weights = config.get('training', {}).get('class_weights', None)
            if Weights is not None:
                Weights = torch.tensor(Weights)
            if loss_name == 'cross_entropy':
                return nn.CrossEntropyLoss(weight=Weights)
            elif loss_name == 'dice':
                return DiceLoss()
            elif loss_name == 'focal':
                return FocalLoss()
            elif loss_name == 'hybrid':
                ce_weight = loss_cfg.get('ce_weight', 0.5)
                dice_weight = loss_cfg.get('dice_weight', 0.5)
                return HybridLoss(weight=Weights, ce_w=ce_weight, dice_w=dice_weight)
            else:
                raise ValueError(f"Loss {loss_name} is not supported.")
    '''),

    # ── PATCH 3/6: trainer.py ──
    "src/training/trainer.py": textwrap.dedent('''\
        import os
        import torch
        import numpy as np
        from tqdm import tqdm
        from src.training.metrics import Evaluator

        class Trainer:
            def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device):
                self.model = model
                self.train_loader = train_loader
                self.val_loader = val_loader
                self.criterion = criterion
                self.optimizer = optimizer
                self.scheduler = scheduler
                self.config = config
                self.device = device
                
                self.num_classes = config['data']['num_classes']
                self.evaluator = Evaluator(self.num_classes)
                
                self.epochs = config['training'].get('epochs', 50)
                self.patience = config['training'].get('early_stopping_patience', 10)
                self.save_dir = config['training'].get('save_dir', 'checkpoints')
                self.use_amp = config['training'].get('mixed_precision', True)
                
                os.makedirs(self.save_dir, exist_ok=True)
                
                self.scaler = torch.amp.GradScaler('cuda') if self.use_amp and self.device.type == 'cuda' else None
                
                self.best_mIoU = 0.0
                self.epochs_no_improve = 0
                self.start_epoch = 1

            def train_epoch(self, epoch):
                self.model.train()
                train_loss = 0.0
                
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
                for images, masks in pbar:
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, dtype=torch.long, non_blocking=True)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    if self.use_amp and self.device.type in ['cuda', 'mps']:
                        with torch.autocast(device_type=self.device.type):
                            outputs = self.model(images)
                            loss = self.criterion(outputs, masks)
                            
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        loss.backward()
                        self.optimizer.step()
                        
                    train_loss += loss.item()
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    
                return train_loss / len(self.train_loader)

            @torch.no_grad()
            def validate(self, epoch):
                self.model.eval()
                self.evaluator.reset()
                val_loss = 0.0
                
                pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
                for images, masks in pbar:
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, dtype=torch.long, non_blocking=True)
                    
                    if self.use_amp and self.device.type in ['cuda', 'mps']:
                        with torch.autocast(device_type=self.device.type):
                            outputs = self.model(images)
                            loss = self.criterion(outputs, masks)
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1)
                    self.evaluator.update(preds, masks)
                    
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    
                metrics = self.evaluator.get_metrics()
                metrics["Loss"] = val_loss / len(self.val_loader)
                
                import matplotlib.pyplot as plt
                import os
                os.makedirs("results/visuals", exist_ok=True)
                
                preds_np = preds[0].cpu().numpy()
                mask_np = masks[0].cpu().numpy()
                img_np = images[0].permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_vis = std * img_np + mean
                img_vis = np.clip(img_vis * 255.0, 0, 255).astype(np.uint8)
                
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(img_vis); axs[0].set_title("Input Image"); axs[0].axis('off')
                axs[1].imshow(mask_np, vmin=0, vmax=6, cmap='tab10'); axs[1].set_title("Ground Truth"); axs[1].axis('off')
                axs[2].imshow(preds_np, vmin=0, vmax=6, cmap='tab10'); axs[2].set_title("Predicted Mask"); axs[2].axis('off')
                plt.savefig(f"results/visuals/{self.config['model']['name']}_val_epoch_{epoch}.png")
                plt.close(fig)
                
                return metrics

            def save_checkpoint(self, path, epoch=None):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_mIoU': self.best_mIoU
                }, path)

            def load_checkpoint(self, path):
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and checkpoint.get('scheduler_state_dict'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.best_mIoU = checkpoint.get('best_mIoU', 0.0)
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"  --> Resumed from epoch {self.start_epoch-1} with best mIoU: {self.best_mIoU:.4f}")

            def train(self):
                print(f"Starting training on {self.device}")
                print(f"Resuming from epoch {self.start_epoch} / {self.epochs}")
                for epoch in range(self.start_epoch, self.epochs + 1):
                    train_loss = self.train_epoch(epoch)
                    val_metrics = self.validate(epoch)
                    val_loss, val_mIoU, val_f1 = val_metrics["Loss"], val_metrics["mIoU"], val_metrics["mF1"]
                    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f} | Val F1: {val_f1:.4f}")
                    if self.scheduler:
                        self.scheduler.step(val_loss) if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else self.scheduler.step()
                    if val_mIoU > self.best_mIoU:
                        self.best_mIoU = val_mIoU
                        self.epochs_no_improve = 0
                        self.save_checkpoint(os.path.join(self.save_dir, f"best_model_{self.config['model']['name']}.pt"), epoch=epoch)
                        print(f"  --> Saved new best model with mIoU: {val_mIoU:.4f}")
                    else:
                        self.epochs_no_improve += 1
                        print(f"  --> No improvement for {self.epochs_no_improve} epochs.")
                    if self.epochs_no_improve >= self.patience:
                        print("Early stopping triggered. Training stopped."); break
                self.save_checkpoint(os.path.join(self.save_dir, f"last_model_{self.config['model']['name']}.pt"), epoch=epoch)
    '''),

    # ── PATCH 4/6: train.py ──
    "scripts/train.py": textwrap.dedent('''\
        import os, yaml, torch, argparse
        from torch.utils.data import DataLoader
        from src.data.dataset import Sen2LULCDataset
        from src.data.transforms import get_train_transforms, get_val_transforms
        from src.data.utils import calculate_class_weights
        from src.models.builder import build_model
        from src.training.losses import get_loss
        from src.training.trainer import Trainer

        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", type=str, default="configs/default.yaml")
            parser.add_argument("--resume", type=str, default=None)
            return parser.parse_args()

        def main():
            args = parse_args()
            with open(args.config, 'r') as f: config = yaml.safe_load(f)
            torch.manual_seed(config.get('seed', 42))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data_dir = config['data']['dataset_dir']
            os.makedirs(data_dir, exist_ok=True)
            train_dataset = Sen2LULCDataset(data_dir, split="train", transforms=get_train_transforms(config['data']))
            val_dataset = Sen2LULCDataset(data_dir, split="val", transforms=get_val_transforms(config['data']))
            train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'], drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
            model = build_model(config).to(device)
            if config['training'].get('loss') == 'cross_entropy' and len(train_dataset) > 0:
                config['training']['class_weights'] = calculate_class_weights(train_loader, config['data']['num_classes']).tolist()
            criterion = get_loss(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('optimizer', {}).get('learning_rate', 0.001), weight_decay=config.get('optimizer', {}).get('weight_decay', 0.01))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
            trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device)
            if args.resume and os.path.exists(args.resume):
                trainer.load_checkpoint(args.resume)
            trainer.train()

        if __name__ == "__main__": main()
    '''),

    # ── PATCH 5/6: segformer.yaml ──
    "configs/segformer.yaml": textwrap.dedent('''\
        model:
          name: "segformer"
          in_channels: 3
          learning_rate: 0.00006
          weight_decay: 0.01

        data:
          dataset_dir: "src/data/SEN-2 LULC"
          img_size: 128
          batch_size: 32
          num_workers: 4
          num_classes: 7
          classes: ["Water", "Dense Forest", "Sparse Forest", "Barren Land", "Built Up", "Agriculture Land", "Fallow Land"]
          augmentation:
            horizontal_flip: true
            vertical_flip: true
            normalize: true

        training:
          epochs: 80
          early_stopping_patience: 10
          mixed_precision: true
          save_dir: "checkpoints_segformer"
          loss:
            type: "hybrid"
            ce_weight: 0.5
            dice_weight: 0.5

        optimizer:
          type: "AdamW"
          learning_rate: 0.00006
          weight_decay: 0.01

        scheduler:
          type: "cosine"
          T_max: 80
    '''),

    # ── PATCH 6/6: sam_seg.yaml ──
    "configs/sam_seg.yaml": textwrap.dedent('''\
        model:
          name: "sam"
          in_channels: 3
          learning_rate: 0.00005
          weight_decay: 0.01

        data:
          dataset_dir: "src/data/SEN-2 LULC"
          img_size: 1024
          batch_size: 2
          num_workers: 4
          num_classes: 7
          classes: ["Water", "Dense Forest", "Sparse Forest", "Barren Land", "Built Up", "Agriculture Land", "Fallow Land"]
          augmentation:
            horizontal_flip: true
            vertical_flip: true
            normalize: true

        training:
          epochs: 40
          early_stopping_patience: 5
          mixed_precision: true
          save_dir: "checkpoints_sam"
          loss:
            type: "hybrid"
            ce_weight: 0.5
            dice_weight: 0.5

        optimizer:
          type: "AdamW"
          learning_rate: 0.00005
          weight_decay: 0.01

        scheduler:
          type: "cosine"
          T_max: 40
    '''),
}

for filepath, content in patches.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content.strip())
    print(f"✅ Patched: {filepath}")

print("\n🎉 Everything is ready for training on Colab!")
```

---

## Cell 2: Run Training

### Option A: SegFormer (Recommended)
```python
!uv run scripts/train.py --config configs/segformer.yaml
```

### Option B: SAM (Fine-Tuning)
```python
!uv run scripts/train.py --config configs/sam_seg.yaml
```
