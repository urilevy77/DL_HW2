import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.siamese_main import AttentionalSiameseNetwork
from src.data.lfw_dataset import LFWDataset
from src.data.transforms import get_transforms
from src.utils.metrics import calculate_accuracy, calculate_precision_recall_f1
from src.utils.visualization import denormalize

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_failure_case(img1, img2, prediction, label, idx, save_dir):
    """
    Saves a failure case image (side by side).
    """
    img1_disp = denormalize(img1)
    img2_disp = denormalize(img2)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1_disp)
    axes[0].axis('off')
    axes[1].imshow(img2_disp)
    axes[1].axis('off')
    
    type_str = "False Positive" if label == 0 else "False Negative"
    plt.suptitle(f"{type_str}\nPred: {prediction:.4f}, Label: {label}")
    
    plt.savefig(f"{save_dir}/fail_{idx}_{type_str.replace(' ', '_')}.png")
    plt.close()

def evaluate_model():
    print(f"Evaluating on device: {DEVICE}")
    
    # 1. Load Data (Test Split)
    dataset_path = config.get("dataset_path", "data/lfw")
    test_dataset = LFWDataset(
        root_dir=dataset_path, 
        transform=get_transforms(split='test'), 
        split='test',
        train_ratio=config.get("train_split_ratio", 0.8)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 2. Load Model
    model = AttentionalSiameseNetwork().to(DEVICE)
    checkpoint_path = f"{config.get('checkpoint_dir', 'checkpoints')}/best_model.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print("Warning: No checkpoint found! Running with random weights.")

    model.eval()
    
    all_preds = []
    all_labels = []
    
    failure_count = 0
    max_failures = 10 # Save up to 10 failure examples
    save_dir = "logs/failures"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Running Inference...")
    with torch.no_grad():
        for img1, img2, labels in tqdm(test_loader):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs, _ = model(img1, img2)
            probs = torch.sigmoid(outputs)
            
            preds_binary = (probs > 0.5).float()
            
            # Store for metrics
            all_preds.append(probs)
            all_labels.append(labels)
            
            # Check for failures in this batch
            if failure_count < max_failures:
                # Find indices where pred != label
                mismatches = (preds_binary.view(-1) != labels.view(-1)).nonzero(as_tuple=True)[0]
                for idx in mismatches:
                    if failure_count >= max_failures:
                        break
                    
                    save_failure_case(
                        img1[idx], img2[idx], 
                        probs[idx].item(), labels[idx].item(), 
                        failure_count, save_dir
                    )
                    failure_count += 1
    
    # Concat all
    all_preds = torch.cat(all_preds) # [Total, 1]
    all_labels = torch.cat(all_labels) # [Total]
    
    # Calculate Metrics
    acc = calculate_accuracy(all_preds, all_labels.view(-1, 1))
    metrics = calculate_precision_recall_f1(all_preds, all_labels)
    
    print("\nXXX Evaluation Results XXX")
    print(f"Accuracy:  {acc.item():.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("--------------------------")
    print(f"Failure examples saved to: {save_dir}")

if __name__ == "__main__":
    evaluate_model()
