import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader

from src.data.dataset import Sen2LULCDataset
from src.data.transforms import get_val_transforms
from src.models.builder import build_model
from src.training.metrics import Evaluator
from src.utils.visualization import plot_confusion_matrix, generate_benchmark_table, plot_qualitative_results
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(config_path, weights_path, data_dir, output_dir):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    
    model.eval()
    
    dataset = Sen2LULCDataset(data_dir, split="test", transforms=get_val_transforms(config['data']))
    loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    evaluator = Evaluator(config['data']['num_classes'])
    
    print(f"Evaluating {config['model']['name']}...")
    
    images_to_plot = []
    masks_to_plot = []
    preds_to_plot = []
    
    from tqdm import tqdm
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader, desc="EVAL")):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            evaluator.update(preds, masks)
            
            # Save first batch for qualitative analysis
            if i == 0:
                 # Denormalize image for plotting
                 denorm_imgs = images.cpu().numpy().transpose(0, 2, 3, 1)
                 mean = np.array([0.485, 0.456, 0.406])
                 std = np.array([0.229, 0.224, 0.225])
                 denorm_imgs = np.clip((denorm_imgs * std) + mean, 0, 1) * 255.0
                 denorm_imgs = denorm_imgs.astype(np.uint8)
                 
                 images_to_plot = denorm_imgs[:5]
                 masks_to_plot = masks.cpu().numpy()[:5]
                 preds_to_plot = preds.cpu().numpy()[:5]

    metrics = evaluator.get_metrics()
    
    class_ious = [round(float(iou), 4) for iou in metrics.get('IoU_per_class', [])]
    print(f"\n--- Class-wise IoU for {config['model']['name']} ---")
    for cls_name, iou in zip(config['data']['classes'], class_ious):
        print(f"{cls_name}: {iou:.4f}")
    print("-" * 40 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot CM
    cm_path = os.path.join(output_dir, f"{config['model']['name']}_cm.png")
    plot_confusion_matrix(evaluator.confusion_matrix, config['data']['classes'], cm_path)
    
    # Plot Qualitative
    qual_path = os.path.join(output_dir, f"{config['model']['name']}_qualitative.png")
    plot_qualitative_results(images_to_plot, masks_to_plot, preds_to_plot, config['data']['classes'], qual_path)
    
    # Return metrics for benchmarking
    params_m = count_parameters(model) / 1e6
    result = {
        "Model": config['model']['name'],
        "mIoU": metrics['mIoU'],
        "Pixel Accuracy": metrics['PixelAccuracy'],
        "F1 Score": metrics['mF1'],
        "Params (M)": params_m
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", type=str, default="configs", help="Directory of all model configs to benchmark")
    parser.add_argument("--data_dir", type=str, default="src/data/SEN-2 LULC")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    results = []
    
    # Evaluate specific config if passed, else all in dir
    if os.path.isfile(args.configs_dir):
        files_to_eval = [args.configs_dir]
        args.configs_dir = os.path.dirname(args.configs_dir)
    else:
        files_to_eval = [os.path.join(args.configs_dir, f) for f in os.listdir(args.configs_dir) if f.endswith(".yaml")]
        
    for cfg_path in files_to_eval:
        with open(cfg_path, 'r') as _f:
            _cfg = yaml.safe_load(_f)

        model_name = _cfg.get('model', {}).get('name', '')
        ckpt_dir = _cfg.get('training', {}).get('save_dir', 'checkpoints')
        weight_name = f"best_model_{model_name}.pt"
             
        weight_path = os.path.join(ckpt_dir, weight_name)
        
        if os.path.exists(weight_path):
            res = evaluate_model(cfg_path, weight_path, args.data_dir, args.output_dir)
            results.append(res)
        else:
            print(f"Skipping {cfg_name}, no checkpoint found at {weight_path}")
            
    # Generate overall benchmark
    if results:
        generate_benchmark_table(results, os.path.join(args.output_dir, "benchmark_table.csv"))
        generate_benchmark_table(results, os.path.join(args.output_dir, "benchmark_table.tex"))
        print("\nEvaluation successfully completed.")
