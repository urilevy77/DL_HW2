import os
import random
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class LFWDataset(Dataset):
    def __init__(self, root_dir, split='train', train_val_split=0.8, transform=None, cache_images=False):
        """
        LFW Dataset for Siamese Network using CSV-based train/test splits.
        
        Args:
            root_dir (str): Path to the LFW dataset images (e.g., 'data/lfw-deepfunneled/lfw-deepfunneled')
            split (str): 'train', 'val', or 'test'
            train_val_split (float): Ratio to split training people into train/val (default: 0.8)
            transform (callable, optional): Optional transform to be applied on images
            cache_images (bool): If True, loads all images into RAM for faster training
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.cache_images = cache_images
        
        # Load people lists from CSV files
        print(f"[{split.upper()}] Loading identities from CSV files...")
        
        try:
            train_df = pd.read_csv('data/peopleDevTrain.csv')
            test_df = pd.read_csv('data/peopleDevTest.csv')
        except FileNotFoundError as e:
            print(f"ERROR: Could not find CSV files: {e}")
            print("Make sure peopleDevTrain.csv and peopleDevTest.csv are in data/ folder")
            raise
        
        train_people = train_df['name'].tolist()  # 4,039 people
        test_people = test_df['name'].tolist()    # 1,712 people
        
        # Split train_people into train and validation
        random.seed(42)  # For reproducibility
        random.shuffle(train_people)
        
        val_split_idx = int(len(train_people) * train_val_split)
        
        if split == 'train':
            self.people = train_people[:val_split_idx]  # 80% of training people
        elif split == 'val':
            self.people = train_people[val_split_idx:]  # 20% of training people
        elif split == 'test':
            self.people = test_people  # All test people
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        print(f"[{split.upper()}] Loaded {len(self.people)} people from CSV")
        
        # Pre-scan images for the selected people
        self.person_images = {}
        self.all_image_paths = []
        
        for person in tqdm(self.people, desc=f"Scanning {split} images"):
            person_dir = os.path.join(root_dir, person)
            
            if not os.path.exists(person_dir):
                print(f"Warning: Directory not found for {person}")
                continue
            
            # Support jpg, png, jpeg
            images = glob.glob(os.path.join(person_dir, "*.jpg")) + \
                     glob.glob(os.path.join(person_dir, "*.png")) + \
                     glob.glob(os.path.join(person_dir, "*.jpeg"))
            
            if len(images) > 0:
                self.person_images[person] = images
                self.all_image_paths.extend(images)
        
        # Remove people with 0 images (just in case)
        self.people = [p for p in self.people if p in self.person_images]
        self.people_with_multiple_images = [p for p in self.people if len(self.person_images[p]) > 1]
        
        print(f"[{split.upper()}] Final count: {len(self.people)} people, {len(self.all_image_paths)} images")
        print(f"[{split.upper()}] People with >1 images: {len(self.people_with_multiple_images)}")

        # Optional: Cache images to RAM
        self.images_cache = {}
        if self.cache_images:
            print(f"[{split.upper()}] Caching {len(self.all_image_paths)} images to RAM...")
            for img_path in tqdm(self.all_image_paths, desc="Caching"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.load()  # Force loading
                    self.images_cache[img_path] = img
                except Exception as e:
                    print(f"Could not cache {img_path}: {e}")
            print(f"[{split.upper()}] Caching complete!")

        # Dataset length = number of images (virtual, since we generate pairs on-the-fly)
        self.dataset_len = len(self.all_image_paths)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        """
        Generates a pair of images (Anchor, Other) and a Label (0 or 1).
        50% chance for same person (1), 50% for different person (0).
        """
        # Randomly choose if we want a positive or negative pair
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # POSITIVE PAIR: Same person
            # Must pick a person who HAS multiple images
            if len(self.people_with_multiple_images) == 0:
                 # Fallback if no one has multiple images
                 should_get_same_class = 0
            else:
                anchor_person = random.choice(self.people_with_multiple_images)
                anchor_img_path = random.choice(self.person_images[anchor_person])
                
                # Same person, different image
                other_img_path = random.choice(self.person_images[anchor_person])
                
                # Ensure it's not the exact same file
                # Since we know len >= 2, this loop will terminate
                while other_img_path == anchor_img_path:
                    other_img_path = random.choice(self.person_images[anchor_person])
                label = 1.0

        if not should_get_same_class:
            # NEGATIVE PAIR: Different people
            anchor_person = random.choice(self.people)
            anchor_img_path = random.choice(self.person_images[anchor_person])
            
            other_person = random.choice(self.people)
            while other_person == anchor_person:
                other_person = random.choice(self.people)
            other_img_path = random.choice(self.person_images[other_person])
            label = 0.0
            
        # Load Images
        try:
            if self.cache_images and anchor_img_path in self.images_cache:
                img1 = self.images_cache[anchor_img_path].copy()
            else:
                img1 = Image.open(anchor_img_path).convert("RGB")

            if self.cache_images and other_img_path in self.images_cache:
                img2 = self.images_cache[other_img_path].copy()
            else:
                img2 = Image.open(other_img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {anchor_img_path} or {other_img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)
