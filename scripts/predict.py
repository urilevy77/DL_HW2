import torch
from torchvision import transforms
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import os
import yaml

from src.models.siamese_main import AttentionalSiameseNetwork
from src.data.transforms import get_transforms
from src.utils.visualization import denormalize

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(img1_path, img2_path, model_path=None):
    # 1. Transformation
    transform = get_transforms(split='test')
    
    try:
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
    except Exception as e:
        print(f"Error opening images: {e}")
        return

    img1_t = transform(img1).unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]
    img2_t = transform(img2).unsqueeze(0).to(DEVICE)
    
    # 2. Model
    model = AttentionalSiameseNetwork().to(DEVICE)
    if model_path is None:
        model_path = f"{config.get('checkpoint_dir', 'checkpoints')}/best_model.pth"
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("Warning: No model found, using random weights!")
        
    model.eval()
    
    # 3. Inference
    with torch.no_grad():
        prediction, attn_map = model(img1_t, img2_t)
        
    prob = prediction.item()
    result = "SAME PERSON" if prob > 0.5 else "DIFFERENT PEOPLE"
    
    print(f"\nPrediction: {prob:.4f}")
    print(f"Result: {result}")
    
    # 4. Visualization
    # Resize attn map
    attn_resized = torch.nn.functional.interpolate(
        attn_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis('off')
    
    # Overlay on Img1 just for viz
    axes[2].imshow(img1)
    axes[2].imshow(attn_resized, cmap='jet', alpha=0.5)
    axes[2].set_title("Attention Map")
    axes[2].axis('off')
    
    plt.suptitle(f"Result: {result} ({prob:.2f})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, required=True, help="Path to first image")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth model file")
    
    args = parser.parse_args()
    
    predict(args.img1, args.img2, args.model)
