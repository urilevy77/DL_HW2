import os
import glob
import yaml
import matplotlib.pyplot as plt
import numpy as np

def analyze_lfw(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    root_dir = config.get("dataset_path", "data/lfw")
    train_split_ratio = config.get("train_split_ratio", 0.8)
    
    print(f"Analyzing dataset at: {root_dir}")
    
    # Get all people
    people = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    people.sort()
    
    total_people = len(people)
    
    # Count images per person
    images_per_person = []
    total_images = 0
    
    for p in people:
        p_dir = os.path.join(root_dir, p)
        imgs = glob.glob(os.path.join(p_dir, "*.jpg")) + \
               glob.glob(os.path.join(p_dir, "*.png")) + \
               glob.glob(os.path.join(p_dir, "*.jpeg"))
        count = len(imgs)
        images_per_person.append(count)
        total_images += count
        
    images_per_person = np.array(images_per_person)
    
    # Split calculations
    split_idx = int(total_people * train_split_ratio)
    train_people = people[:split_idx]
    test_people = people[split_idx:]
    
    train_images_count = sum(images_per_person[:split_idx])
    test_images_count = sum(images_per_person[split_idx:])
    
    print("-" * 30)
    print(f"Total People: {total_people}")
    print(f"Total Images: {total_images}")
    print("-" * 30)
    print(f"Train Set (People): {len(train_people)} ({len(train_people)/total_people:.1%})")
    print(f"Train Set (Images): {train_images_count} ({train_images_count/total_images:.1%})")
    print("-" * 30)
    print(f"Test Set (People): {len(test_people)} ({len(test_people)/total_people:.1%})")
    print(f"Test Set (Images): {test_images_count} ({test_images_count/total_images:.1%})")
    print("-" * 30)
    print(f"Min images/person: {images_per_person.min()}")
    print(f"Max images/person: {images_per_person.max()}")
    print(f"Avg images/person: {images_per_person.mean():.2f}")
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(images_per_person, bins=range(1, 20), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Images per Person (Clipped at 20)')
    plt.xlabel('Number of Images')
    plt.ylabel('Number of People')
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('dataset_distribution.png')
    print("Saved histogram to dataset_distribution.png")

if __name__ == "__main__":
    analyze_lfw()
