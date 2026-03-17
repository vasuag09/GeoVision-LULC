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
                with torch.amp.autocast(device_type=self.device.type):
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
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
            val_loss += loss.item()
            
            # Predictions
            preds = torch.argmax(outputs, dim=1)
            self.evaluator.update(preds, masks)
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        metrics = self.evaluator.get_metrics()
        metrics["Loss"] = val_loss / len(self.val_loader)
        
        # Step 5: Visualize predictions
        import matplotlib.pyplot as plt
        import os
        os.makedirs("results/visuals", exist_ok=True)
        
        preds_np = preds[0].cpu().numpy()
        mask_np = masks[0].cpu().numpy()
        
        # Unnormalize to display
        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = std * img_np + mean
        img_vis = np.clip(img_vis * 255.0, 0, 255).astype(np.uint8)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_vis)
        axs[0].set_title("Input Image")
        axs[0].axis('off')

        axs[1].imshow(mask_np, vmin=0, vmax=6, cmap='tab10')
        axs[1].set_title("Ground Truth")
        axs[1].axis('off')

        axs[2].imshow(preds_np, vmin=0, vmax=6, cmap='tab10')
        axs[2].set_title("Predicted Mask")
        axs[2].axis('off')

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
        """Load a checkpoint to resume training from where it left off."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_mIoU = checkpoint.get('best_mIoU', 0.0)
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"  --> Resumed from epoch {self.start_epoch - 1} with best mIoU: {self.best_mIoU:.4f}")

    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Resuming from epoch {self.start_epoch} / {self.epochs}")
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            val_loss = val_metrics["Loss"]
            val_mIoU = val_metrics["mIoU"]
            val_f1 = val_metrics["mF1"]
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f} | Val F1: {val_f1:.4f}")
            
            # Formatted Class-wise IoU output
            ious = [round(iou, 4) for iou in val_metrics.get("IoU_per_class", [])]
            print(f"Class-wise IoU: {ious}")
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early Stopping and Checkpointing Check
            if val_mIoU > self.best_mIoU:
                self.best_mIoU = val_mIoU
                self.epochs_no_improve = 0
                self.save_checkpoint(os.path.join(self.save_dir, f"best_model_{self.config['model']['name']}.pt"), epoch=epoch)
                print(f"  --> Saved new best model with mIoU: {val_mIoU:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"  --> No improvement for {self.epochs_no_improve} epochs.")
                
            if self.epochs_no_improve >= self.patience:
                print("Early stopping triggered. Training stopped.")
                break
                
        # Save last checkpoint
        self.save_checkpoint(os.path.join(self.save_dir, f"last_model_{self.config['model']['name']}.pt"), epoch=epoch)
        print("Training complete.")
