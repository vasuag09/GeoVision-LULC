import os
import cv2
import yaml
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.dataset import Sen2LULCDataset
from src.data.transforms import get_val_transforms
from src.models.builder import build_model

class SemanticSegmentationTarget:
    """
    Custom Target Function to compute CAM for semantic segmentation.
    We target a specific class (category) across the entire predicted mask map.
    """
    def __init__(self, category_id, mask):
        self.category_id = category_id
        # mask is the ground truth (or predicted) mask where we want to highlight features
        self.mask = torch.from_numpy(mask).float()
        
    def __call__(self, model_output):
        # We want to maximize the output of the specified class at positions where the mask is active
        # Output shape: [batch, num_classes, H, W]
        # model_output is typically the logits
        if len(self.mask.shape) == 2:
            self.mask = self.mask.unsqueeze(0) # add batch dim
        
        self.mask = self.mask.to(model_output.device)
        
        # When passed to sum(), model_output for semantic segmentation in pytorch-grad-cam
        # might either be [num_classes, H, W] or [batch, num_classes, H, W]
        # Let's handle both.
        if len(model_output.shape) == 3:
            class_output = model_output[self.category_id, :, :]
        else:
            class_output = model_output[:, self.category_id, :, :]
        
        # Multiply only the target pixels we care about
        loss = (class_output * self.mask).sum()
        return loss


def generate_gradcam_visualizations(config_path, weights_path, data_dir, output_dir, num_samples=3):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(config)
    
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    else:
        print(f"Weights not found at {weights_path}, skipping GradCAM.")
        return
        
    model = model.to(device)
    model.eval()
    
    # Identify the target layer for GradCAM based on CNN Architecture
    target_layers = []
    
    model_name = config['model']['name'].lower()
    
    if model_name == "unet":
        # Usually target the last convolutional layer of the decoder before the final 1x1 classifier
        target_layers = [model.up4.conv.double_conv[-1]] # ReLU of the last double conv layer
    elif model_name == "deeplabv3+":
        # Target the resnet50 base output from layer4 
        target_layers = [model.model.backbone.layer4[-1].conv3]
    elif model_name == "segnet":
        # Target the last layer of the highest decoder block
        target_layers = [model.dec_conv1[-1]]
    elif model_name == "segformer":
        # Target the last layer norm of the encoder backbone
        # Layers in HF SegFormer: model.segformer.encoder.layer_norm[0...3]
        target_layers = [model.model.segformer.encoder.layer_norm[-1]]
    elif model_name == "swin_unet":
        # Target the final layernorm in the Swin backbone
        target_layers = [model.model.swin.layernorm]
    else:
        print(f"Explainability GradCAM not currently mapping layers for architecture: {model_name}")
        return
        
    dataset = Sen2LULCDataset(data_dir, split="test", transforms=get_val_transforms(config['data']['img_size']))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    classes = config['data']['classes']
    os.makedirs(output_dir, exist_ok=True)
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    print(f"Generating Explainability Maps for {model_name}...")
    
    samples_processed = 0
    
    for i, (images, masks) in enumerate(loader):
        if samples_processed >= num_samples:
            break
            
        images = images.to(device)
        masks = masks.to(device)
        
        # Denormalize the image specifically for the CAM visualization
        # The pytorch_grad_cam library expects the original non-normalized image in [0, 1] range as a numpy Float32
        input_tensor = images.clone()
        
        with torch.no_grad():
            outputs = model(input_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            
        # Get raw image format
        denorm_img = input_tensor.cpu().numpy()[0].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = np.clip((denorm_img * std) + mean, 0, 1)
        
        # Iterate over unique classes present in the PREDICTED mask
        unique_classes = np.unique(preds)
        
        for cls_id in unique_classes:
            class_name = classes[cls_id]
            
            # Create a binary mask where the prediction exactly equals this predicted class
            # This isolates the regions the model thinks are 'Forest' or 'Water'
            target_mask = np.float32(preds == cls_id)
            
            targets = [SemanticSegmentationTarget(cls_id, target_mask)]
            
            # Extract GradCAM Map 
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay CAM softly over the original RGB patch
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Resize for better visibility if it's currently 64x64
            viz_width = 256
            cam_image_resized = cv2.resize(cam_image, (viz_width, viz_width), interpolation=cv2.INTER_NEAREST)
            orig_rgb_resized = cv2.resize((rgb_img * 255).astype(np.uint8), (viz_width, viz_width), interpolation=cv2.INTER_NEAREST)
            
            # Combine side-by-side: [Original RGB | GradCAM attention]
            combined = np.hstack((orig_rgb_resized, cam_image_resized))
            
            # Convert RGB to BGR for OpenCV saving
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(output_dir, f"{model_name}_sample{i}_class{cls_id}_{class_name}.png")
            cv2.imwrite(save_path, combined_bgr)
            
        samples_processed += 1
        
    print(f"Successfully exported Attention Heatmaps to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights .pth")
    parser.add_argument("--data_dir", type=str, default="src/data/SEN-2 LULC")
    parser.add_argument("--output_dir", type=str, default="results/explainability")
    parser.add_argument("--samples", type=int, default=3, help="Number of test patches to generate CAMs for")
    args = parser.parse_args()
    
    generate_gradcam_visualizations(args.config, args.weights, args.data_dir, args.output_dir, args.samples)
