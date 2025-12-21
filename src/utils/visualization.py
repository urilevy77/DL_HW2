import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def denormalize(tensor):
    """
    Reverts the ImageNet normalization for visualization.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def save_attention_map(img1, img2, attn_map, prediction, label, epoch, batch_idx, save_dir="logs/vis"):
    """
    Saves a figure showing the two images and the attention map overlay.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Prepare images
    img1_disp = denormalize(img1)
    img2_disp = denormalize(img2)
    
    # 2. Resize attention map to image size for overlay
    # attn_map is [1, 7, 7], needs to be [224, 224]
    attn_resized = torch.nn.functional.interpolate(
        attn_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    )
    attn_resized = attn_resized.squeeze().cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img1_disp)
    axes[0].set_title("Image 1")
    axes[0].axis('off')
    
    axes[1].imshow(img2_disp)
    axes[1].set_title("Image 2")
    axes[1].axis('off')
    
    # Overlay on the first image (or just show the map itself)
    # Let's show the map itself for clarity, or overlay.
    # Assignment asks for "heatmaps", usually overlay is best.
    axes[2].imshow(img1_disp)
    axes[2].imshow(attn_resized, cmap='jet', alpha=0.5) # Overlay
    axes[2].set_title(f"Attention Map\nPred: {prediction:.4f}, Label: {label}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}_batch_{batch_idx}.png")
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_path="logs/loss_curve.png"):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(save_path)
    plt.close()
