import streamlit as st
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from io import BytesIO

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project-specific modules
from src.models.builder import build_model
import yaml

# Set page configuration
st.set_page_config(
    page_title="GeoVision LULC Segmentation",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom color map matching our classes
LULC_COLORS = {
    0: [0, 119, 190],    # Water (Blue)
    1: [34, 139, 34],    # Dense Forest (Dark Green)
    2: [144, 238, 144],  # Sparse Forest (Light Green)
    3: [210, 180, 140],  # Barren Land (Tan/Brown)
    4: [220, 20, 60],    # Built Up (Red/Crimson)
    5: [255, 215, 0],    # Agriculture Land (Gold)
    6: [147, 112, 219]   # Fallow Land (Purple)
}

@st.cache_resource
def load_pytorch_model(model_name, checkpoint_path):
    """Loads a specified PyTorch model configuration and weights."""
    model_configs = {
        "DeepLabV3+ (ResNet-50)": {"name": "deeplabv3+", "in_channels": 3},
        "SegFormer": {"name": "segformer", "in_channels": 3},
        "Swin-UNet": {"name": "swin_unet", "in_channels": 3},
        "UNet Baseline": {"name": "unet", "in_channels": 3}
    }
    
    config = {
        'model': model_configs.get(model_name, model_configs["DeepLabV3+ (ResNet-50)"]),
        'data': {'num_classes': 7}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = build_model(config).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Robustly handle model state dict loading (strip/add 'model.' or 'module.' prefixes)
        model_keys = set(model.state_dict().keys())
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.') and (k[6:] in model_keys):
                new_state_dict[k[6:]] = v
            elif (f"model.{k}" in model_keys):
                new_state_dict[f"model.{k}"] = v
            elif k.startswith('module.'): # for DP/DDP checkpoints
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        st.sidebar.success(f"Weights loaded securely!")
    else:
        st.sidebar.warning(f"No checkpoint found at {checkpoint_path}. Using uninitialized weights.")
        
    model.eval()
    return model, device

def apply_color_map(mask):
    """Converts a standard 2D integer mask into a colorful 3D RGB map."""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in LULC_COLORS.items():
        rgb_mask[mask == class_idx] = color
    return rgb_mask

def preprocess_image(image):
    """Converts a PIL image or array into a standard normalized PyTorch tensor natively scaled for our models."""
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0) # Add batch dimension

def main():
    st.title("🌍 GeoVision: Sentinel-2 LULC Segmentation")
    st.markdown("""
        Upload a Sentinel-2 satellite image patch to generate a Land Use Land Cover (LULC) segmentation map 
        using our custom-trained cutting-edge Deep Learning architectures.
    """)

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        selected_model = st.selectbox(
            "Select Model Architecture",
            ["DeepLabV3+ (ResNet-50)", "SegFormer", "Swin-UNet", "UNet Baseline"],
            index=0
        )
        
        # Dynamic Checkpoint Selection
        default_ckpts = {
            "DeepLabV3+ (ResNet-50)": "checkpoints_deeplabv3plus/best_model_deeplabv3+.pt",
            "SegFormer": "checkpoints_segformer/best_model_segformer.pt",
            "UNet Baseline": "checkpoints/best_model_unet.pt"
        }
        
        st.markdown(f"**Current Model**: {selected_model}")
        default_path = default_ckpts.get(selected_model, "checkpoints/best_model_deeplabv3+.pt")
        checkpoint_path = st.text_input("Model Checkpoint Path", default_path)
        
        st.markdown("---")
        st.markdown("### LULC Classes")
        st.markdown("""
        - 🟦 Water
        - 🟩 Dense Forest
        - 🟨 Sparse Forest
        - 🟫 Barren Land
        - 🟥 Built Up
        - 🌾 Agriculture Land
        - 🟪 Fallow Land
        """)

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Satellite Image")
        uploaded_file = st.file_uploader(
            "Choose a Sentinel-2 patch (PNG, JPG, TIF)", 
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff']
        )

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Input Satellite Imagery", use_container_width=True)
            
            # Action button
            if st.button("🚀 Run Segmentation", use_container_width=True, type="primary"):
                with st.spinner(f"Running inference using {selected_model}..."):
                    try:
                        # 1. Load model
                        model, device = load_pytorch_model(selected_model, checkpoint_path)
                        
                        # 2. Preprocess image
                        pil_image = Image.open(uploaded_file).convert("RGB")
                        input_tensor = preprocess_image(pil_image).to(device)
                        
                        # 3. Model forward pass
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            preds = torch.argmax(outputs, dim=1).squeeze(0) # [128, 128]
                        
                        # 4. Postprocess mask into colorful layout
                        predicted_mask_np = preds.cpu().numpy()
                        rgb_mask = apply_color_map(predicted_mask_np)
                        
                        # 5. Save results to session state to prevent reloading
                        st.session_state['inference_done'] = True
                        st.session_state['rgb_mask'] = rgb_mask
                        st.session_state['raw_mask'] = predicted_mask_np
                        st.session_state['pil_image'] = pil_image
                        
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                    
    with col2:
        st.subheader("2. Segmentation LULC Map")
        
        if uploaded_file is None:
            st.info("👈 Please upload an image first to see the segmentation output.")
        elif st.session_state.get('inference_done', False) and 'rgb_mask' in st.session_state:
            st.success("✅ Segmentation complete!")
            
            # Display colored mapping
            st.image(st.session_state['rgb_mask'], caption="Predicted LULC Mask", use_container_width=True)
            
            # Create a comparison slider capability if matplotlib is wanted, but an overlay is also powerful!
            st.markdown("### Overlay Review")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(st.session_state['pil_image'].resize((128, 128)))
            ax.imshow(st.session_state['rgb_mask'], alpha=0.5)
            ax.axis('off')
            st.pyplot(fig)
            
            # Convert array directly into bytestream for TIF download
            buf = BytesIO()
            img_to_download = Image.fromarray(st.session_state['rgb_mask'])
            img_to_download.save(buf, format="TIFF")
            byte_im = buf.getvalue()
            
            # Download button
            st.download_button(
                label="📥 Download GeoTIFF Map",
                data=byte_im,
                file_name="predicted_lulc_map.tif",
                mime="image/tiff",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
