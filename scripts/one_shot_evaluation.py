import torch
import torch.nn as nn
import pandas as pd
import os
import sys
import glob
import random
import numpy as np
import argparse
import json
from PIL import Image
from tqdm import tqdm
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.siamese_main import AttentionalSiameseNetwork
from src.data.transforms import get_transforms

# Load Config
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_set(root_dir):
    """
    Load test set identities and their image paths from CSV.
    
    Returns:
        dict: {person_name: [image_paths]}
    """
    print("Loading test set from CSV...")
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "peopleDevTest.csv")
    test_df = pd.read_csv(csv_path)
    test_people = test_df['name'].tolist()
    
    test_data = {}
    people_with_multiple_images = []
    
    for person in tqdm(test_people, desc="Scanning test images"):
        person_dir = os.path.join(root_dir, person)
        
        if not os.path.exists(person_dir):
            continue
            
        images = glob.glob(os.path.join(person_dir, "*.jpg")) + \
                 glob.glob(os.path.join(person_dir, "*.png")) + \
                 glob.glob(os.path.join(person_dir, "*.jpeg"))
        
        if len(images) >= 2:  # Need at least 2 images (1 support + 1 test)
            test_data[person] = images
            people_with_multiple_images.append(person)
    
    print(f"Test set: {len(test_data)} people with ≥2 images")
    return test_data, people_with_multiple_images

def load_and_preprocess_image(img_path, transform):
    """Load and preprocess a single image."""
    img = Image.open(img_path).convert("RGB")
    if transform:
        img = transform(img)
    return img

def n_way_oneshot_task(model, test_img_path, support_data, transform, device):
    """
    Perform a single N-way one-shot task.
    
    Args:
        model: Trained Siamese network
        test_img_path: Path to test image
        support_data: List of tuples [(class_idx, class_name, support_img_path), ...]
        transform: Image preprocessing
        device: torch.device
        
    Returns:
        predicted_idx: Predicted class index
        similarities: List of similarity scores for each class
    """
    # Load test image
    test_img = load_and_preprocess_image(test_img_path, transform).unsqueeze(0).to(device)
    
    similarities = []
    
    # Compare against each support image
    with torch.no_grad():
        for class_idx, class_name, support_img_path in support_data:
            # Load support image
            support_img = load_and_preprocess_image(support_img_path, transform).unsqueeze(0).to(device)
            
            # Get similarity score
            logits, _ = model(test_img, support_img)
            similarity = torch.sigmoid(logits).item()
            similarities.append(similarity)
    
    # Predict class with highest similarity
    predicted_idx = np.argmax(similarities)
    
    return predicted_idx, similarities

def evaluate_n_way(model, test_data, people_with_multiple, n_way=20, n_trials=400, transform=None, device='cuda'):
    """
    Evaluate model on N-way one-shot classification.
    
    Args:
        model: Trained Siamese network
        test_data: Dictionary {person_name: [image_paths]}
        people_with_multiple: List of people with ≥2 images
        n_way: Number of classes per trial
        n_trials: Number of trials to run
        transform: Image preprocessing
        device: torch.device
        
    Returns:
        results: Dictionary with accuracy and trial details
    """
    model.eval()
    
    correct = 0
    all_trials = []
    
    print(f"\nRunning {n_trials} trials of {n_way}-way one-shot learning...")
    
    for trial_idx in tqdm(range(n_trials), desc=f"{n_way}-way evaluation"):
        # 1. Sample N classes (from people with ≥2 images)
        sampled_classes = random.sample(people_with_multiple, n_way)
        
        # 2. For each class, pick 1 support image
        support_data = []
        for class_idx, class_name in enumerate(sampled_classes):
            support_img = random.choice(test_data[class_name])
            support_data.append((class_idx, class_name, support_img))
        
        # 3. Pick one class as ground truth
        gt_idx = random.randint(0, n_way - 1)
        gt_class = sampled_classes[gt_idx]
        
        # 4. Pick test image from ground truth class (different from support)
        test_img = random.choice(test_data[gt_class])
        # Ensure test image is different from support image
        while test_img == support_data[gt_idx][2]:
            test_img = random.choice(test_data[gt_class])
        
        # 5. Run N-way one-shot task
        predicted_idx, similarities = n_way_oneshot_task(
            model, test_img, support_data, transform, device
        )
        
        # 6. Check if correct
        is_correct = (predicted_idx == gt_idx)
        if is_correct:
            correct += 1
        
        # Store trial details
        trial_result = {
            'trial_idx': trial_idx,
            'gt_idx': gt_idx,
            'gt_class': gt_class,
            'predicted_idx': predicted_idx,
            'predicted_class': sampled_classes[predicted_idx],
            'correct': bool(is_correct),  # Convert to native Python bool for JSON
            'similarities': similarities,
            'test_img': test_img,
            'support_imgs': [s[2] for s in support_data]
        }
        all_trials.append(trial_result)
    
    accuracy = (correct / n_trials) * 100
    
    results = {
        'n_way': n_way,
        'n_trials': n_trials,
        'correct': correct,
        'accuracy': accuracy,
        'random_baseline': (1.0 / n_way) * 100,
        'trials': all_trials
    }
    
    return results

def save_results(results, output_path):
    """Save results to JSON file (excluding image tensors)."""
    # Create a copy without image paths for cleaner JSON
    save_data = {
        'n_way': results['n_way'],
        'n_trials': results['n_trials'],
        'correct': results['correct'],
        'accuracy': results['accuracy'],
        'random_baseline': results['random_baseline'],
        'trial_summary': [
            {
                'trial_idx': t['trial_idx'],
                'correct': t['correct'],
                'gt_class': t['gt_class'],
                'predicted_class': t['predicted_class']
            }
            for t in results['trials']
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='N-Way One-Shot Evaluation')
    parser.add_argument('--n_way', type=int, default=20, help='Number of classes (default: 20)')
    parser.add_argument('--n_trials', type=int, default=400, help='Number of trials (default: 400)')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth', help='Model checkpoint')
    parser.add_argument('--output', type=str, default='../results/oneshot_results.json', help='Output JSON file')
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"N-way: {args.n_way}")
    print(f"Trials: {args.n_trials}")
    
    # 1. Load test set
    dataset_path = config.get("dataset_path", "data/lfw-deepfunneled/lfw-deepfunneled")
    test_data, people_with_multiple = load_test_set(dataset_path)
    
    if len(people_with_multiple) < args.n_way:
        print(f"ERROR: Not enough people with ≥2 images for {args.n_way}-way!")
        print(f"Available: {len(people_with_multiple)}, Required: {args.n_way}")
        return
    
    # 2. Load model
    model = AttentionalSiameseNetwork().to(DEVICE)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    else:
        print(f"WARNING: Checkpoint not found: {args.checkpoint}")
        print("Using random weights (for testing only!)")
    
    model.eval()
    
    # 3. Get transforms
    transform = get_transforms(split='test', input_size=tuple(config.get('input_shape', [224, 224])))
    
    # 4. Run evaluation
    results = evaluate_n_way(
        model=model,
        test_data=test_data,
        people_with_multiple=people_with_multiple,
        n_way=args.n_way,
        n_trials=args.n_trials,
        transform=transform,
        device=DEVICE
    )
    
    # 5. Print results
    print("\n" + "="*50)
    print(f"N-WAY ONE-SHOT EVALUATION RESULTS")
    print("="*50)
    print(f"N-way: {results['n_way']}")
    print(f"Trials: {results['n_trials']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Random Baseline: {results['random_baseline']:.2f}%")
    print(f"Improvement over random: {results['accuracy'] - results['random_baseline']:.2f}%")
    print("="*50)
    
    # 6. Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(results, args.output)
    
    # 7. Show some example trials
    print("\nExample trials:")
    for i in range(min(5, len(results['trials']))):
        trial = results['trials'][i]
        status = "✓" if trial['correct'] else "✗"
        print(f"{status} Trial {i}: GT={trial['gt_class']}, Pred={trial['predicted_class']}")

if __name__ == "__main__":
    main()
