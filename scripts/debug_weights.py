import torch

# Load best UNet
ckpt_best = torch.load("checkpoints/best_model_unet.pt", map_location="cpu", weights_only=False)
sd_best = ckpt_best.get('model_state_dict', ckpt_best)

# Load last UNet
ckpt_last = torch.load("checkpoints/last_model_unet.pt", map_location="cpu", weights_only=False)
sd_last = ckpt_last.get('model_state_dict', ckpt_last)

print("Best model dict keys sample:")
print(list(sd_best.keys())[:5])

# Compare conv layers
for k in sd_best:
    if 'conv' in k or 'weight' in k:
        layer_best = sd_best[k]
        layer_last = sd_last[k]
        
        diff = torch.abs(layer_best - layer_last).mean()
        print(f"{k}: Mean={layer_best.mean():.4f}, Std={layer_best.std():.4f}, Last_Diff={diff:.6f}")
        break
