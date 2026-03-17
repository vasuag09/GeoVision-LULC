import numpy as np
import torch
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_mask, true_mask):
        """
        Updates confusion matrix given predictions and true masks.
        Inputs are torch Tensors or numpy arrays.
        """
        if torch.is_tensor(pred_mask):
            pred = pred_mask.detach().cpu().numpy().flatten()
        else:
            pred = pred_mask.flatten()
            
        if torch.is_tensor(true_mask):
            target = true_mask.detach().cpu().numpy().flatten()
        else:
            target = true_mask.flatten()
            
        # Exclude ignore index
        mask = (target >= 0) & (target < self.num_classes)
        
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist

    def get_metrics(self):
        """
        Calculates mIoU, Pixel Accuracy, F1 score.
        """
        hist = self.confusion_matrix
        
        # Pixel Accuracy
        acc = np.diag(hist).sum() / (hist.sum() + 1e-6)
        
        # Class-wise Accuracy
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-6)
        mean_acc_cls = np.nanmean(acc_cls)
        
        # IoU
        intersection = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
        iou = intersection / (union + 1e-6)
        mIoU = np.nanmean(iou)
        
        # F1 Score
        precision = intersection / (hist.sum(axis=0) + 1e-6)
        recall = intersection / (hist.sum(axis=1) + 1e-6)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        mF1 = np.nanmean(f1)
        
        return {
            "mIoU": mIoU,
            "PixelAccuracy": acc,
            "ClasswiseAcc": mean_acc_cls,
            "mF1": mF1,
            "IoU_per_class": iou.tolist()
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
