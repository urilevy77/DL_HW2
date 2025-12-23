import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_split(people_list, root_dir, split_name):
    """Analyze a specific split (train or test)"""
    
    # Count images per person for this split
    images_per_person = []
    person_image_map = {}
    total_images = 0
    
    for person in people_list:
        person_dir = os.path.join(root_dir, person)
        if not os.path.exists(person_dir):
            print(f"Warning: Directory not found for {person}")
            continue
            
        imgs = glob.glob(os.path.join(person_dir, "*.jpg")) + \
               glob.glob(os.path.join(person_dir, "*.png")) + \
               glob.glob(os.path.join(person_dir, "*.jpeg"))
        count = len(imgs)
        images_per_person.append(count)
        person_image_map[person] = count
        total_images += count
    
    images_per_person = np.array(images_per_person)
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"{split_name.upper()} SET ANALYSIS")
    print(f"{'='*50}")
    print(f"Total People: {len(people_list)}")
    print(f"Total Images: {total_images}")
    print(f"Min images/person: {images_per_person.min()}")
    print(f"Max images/person: {images_per_person.max()}")
    print(f"Avg images/person: {images_per_person.mean():.2f}")
    print(f"Median images/person: {np.median(images_per_person):.2f}")
    
    # Class distribution statistics
    print(f"\n--- Class Distribution ---")
    print(f"People with 1 image: {sum(images_per_person == 1)} ({sum(images_per_person == 1)/len(images_per_person):.1%})")
    print(f"People with 2-5 images: {sum((images_per_person >= 2) & (images_per_person <= 5))} ({sum((images_per_person >= 2) & (images_per_person <= 5))/len(images_per_person):.1%})")
    print(f"People with 6-10 images: {sum((images_per_person >= 6) & (images_per_person <= 10))} ({sum((images_per_person >= 6) & (images_per_person <= 10))/len(images_per_person):.1%})")
    print(f"People with 11-20 images: {sum((images_per_person >= 11) & (images_per_person <= 20))} ({sum((images_per_person >= 11) & (images_per_person <= 20))/len(images_per_person):.1%})")
    print(f"People with >20 images: {sum(images_per_person > 20)} ({sum(images_per_person > 20)/len(images_per_person):.1%})")
    
    # Top 5 people with most images
    print(f"\n--- Top 5 People (Most Images) ---")
    top_5 = sorted(person_image_map.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (person, count) in enumerate(top_5, 1):
        print(f"{i}. {person}: {count} images")
    
    return images_per_person, total_images

def analyze_lfw():
    # Load train/test splits from CSV files
    print("Loading train/test splits from CSV files...")
    
    try:
        train_df = pd.read_csv('data/peopleDevTrain.csv')
        test_df = pd.read_csv('data/peopleDevTest.csv')
    except FileNotFoundError as e:
        print(f"Error: CSV file not found - {e}")
        print("Make sure peopleDevTrain.csv and peopleDevTest.csv are in data/ folder")
        return
    
    all_train_people = train_df['name'].tolist()
    test_people = test_df['name'].tolist()
    
    # Split train people into train and validation (matching LFWDataset logic)
    import random
    random.seed(42)
    random.shuffle(all_train_people)
    
    val_split_idx = int(len(all_train_people) * 0.8)
    train_people = all_train_people[:val_split_idx]
    val_people = all_train_people[val_split_idx:]
    
    # Dataset path
    root_dir = "data/lfw-deepfunneled/lfw-deepfunneled"
    
    print(f"\n{'='*50}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*50}")
    print(f"Dataset path: {root_dir}")
    print(f"Total people in dataset: {len(all_train_people) + len(test_people)}")
    print(f"Train people: {len(train_people)}")
    print(f"Validation people: {len(val_people)}")
    print(f"Test people: {len(test_people)}")
    
    # Analyze splits
    train_images_per_person, train_total_images = analyze_split(train_people, root_dir, "train")
    val_images_per_person, val_total_images = analyze_split(val_people, root_dir, "validation")
    test_images_per_person, test_total_images = analyze_split(test_people, root_dir, "test")
    
    # Combined statistics
    total_images = train_total_images + val_total_images + test_total_images
    print(f"\n{'='*50}")
    print(f"COMBINED STATISTICS")
    print(f"{'='*50}")
    print(f"Total images: {total_images}")
    print(f"Train images: {train_total_images} ({train_total_images/total_images:.1%})")
    print(f"Val images: {val_total_images} ({val_total_images/total_images:.1%})")
    print(f"Test images: {test_total_images} ({test_total_images/total_images:.1%})")
    
    # Create histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Train histogram
    axes[0].hist(train_images_per_person, bins=range(1, 21), 
                 color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Train Set\n({len(train_people)} people)')
    axes[0].set_xlabel('Number of Images')
    axes[0].set_ylabel('Number of People')
    
    # Val histogram
    axes[1].hist(val_images_per_person, bins=range(1, 21), 
                 color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].set_title(f'Validation Set\n({len(val_people)} people)')
    axes[1].set_xlabel('Number of Images')
    
    # Test histogram
    axes[2].hist(test_images_per_person, bins=range(1, 21), 
                 color='lightcoral', edgecolor='black', alpha=0.7)
    axes[2].set_title(f'Test Set\n({len(test_people)} people)')
    axes[2].set_xlabel('Number of Images')
    
    for ax in axes:
        ax.grid(axis='y', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150)
    print(f"\nSaved histograms to dataset_distribution.png")

if __name__ == "__main__":
    analyze_lfw()
