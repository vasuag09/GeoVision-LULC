import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class SemanticSegmentationTarget:
    """
    Target for Grad-CAM for Semantic Segmentation.
    """
    def __init__(self, category_mask):
        self.category_mask = category_mask

    def __call__(self, model_output):
        # The output is (B, C, H, W). We take the sum of logits for the predicted class
        return (model_output[0, :, :, :] * self.category_mask).sum()

def run_gradcam(model, input_tensor, target_layer, predicted_class, original_image):
    """
    Run GradCAM on a segmentation model.
    Args:
        model: PyTorch model.
        input_tensor: Tensor of shape (1, C, H, W).
        target_layer: The CNN layer to compute gradients from.
        predicted_class: The class index to visualize.
        original_image: NumPy array of shape (H, W, 3) normalized between 0 and 1.
    """
    with GradCAM(model=model, target_layers=[target_layer], use_cuda=input_tensor.is_cuda) as cam:
        # Create a blank mask, set 1 where we want to explain
        output = model(input_tensor)
        preds = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        category_mask = torch.zeros_like(output[0])
        # We want to explain regions predicted as `predicted_class`
        category_mask[predicted_class, :, :] = torch.tensor((preds == predicted_class), dtype=torch.float32)
        category_mask = category_mask.to(input_tensor.device)

        targets = [SemanticSegmentationTarget(category_mask)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay
        cam_image = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        return cam_image, grayscale_cam

def visualize_attention_rollout(model, input_tensor):
    """
    Placeholder for attention rollout for Vision Transformers (e.g. SegFormer/Swin).
    Standard libraries like pytorch-grad-cam do not uniformly support all HF segmentation models out-of-the-box
    without careful layer wrapping. For a robust pipeline, GradCAM on last feature map is typically sufficient,
    or pulling attention weights from HF outputs.
    """
    # Assuming HF models: model(pixel_values, output_attentions=True)
    pass
