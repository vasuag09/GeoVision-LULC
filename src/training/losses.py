import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Ensure contiguity for MPS
        logits = logits.contiguous()
        targets = targets.contiguous()
        num_classes = logits.shape[1]
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1).contiguous()
        
        # Create one-hot targets, ensuring it accepts LongTensor
        targets_long = targets.long()
        targets_one_hot = F.one_hot(torch.clamp(targets_long, 0, num_classes-1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float().contiguous() # (B, C, H, W)
        
        # Mask out ignore index
        mask = (targets != self.ignore_index).unsqueeze(1).float()
        
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask
        
        dims = (0, 2, 3) # compute over batch, spatial dims
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
        
        # calculate pt
        pt = torch.exp(-ce_loss)
        
        # focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Exclude ignore index
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
