import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
import yaml
from tqdm import tqdm
import time

# Custom Imports
from src.models.siamese_main import AttentionalSiameseNetwork
from src.data.lfw_dataset import LFWDataset
from src.data.transforms import get_transforms
from src.utils.metrics import calculate_accuracy
from src.utils.visualization import save_attention_map, plot_loss_curve

# --- Load Configuration ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Hyperparameters
BATCH_SIZE = config.get("batch_size", 32)
LEARNING_RATE = config.get("learning_rate", 1e-4)
NUM_EPOCHS = config.get("num_epochs", 20)
WEIGHT_DECAY = config.get("weight_decay", 1e-5)
DATASET_PATH = config.get("dataset_path", "data/lfw")
CHECKPOINT_DIR = config.get("checkpoint_dir", "checkpoints")
LOG_DIR = config.get("log_dir", "logs")
DROPOUT_PROB = config.get("dropout_prob", 0.0)
INPUT_SHAPE = tuple(config.get("input_shape", [224, 224]))
UNFREEZE_EPOCH = config.get("unfreeze_epoch", 5)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "vis"), exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    # scaler = torch.amp.GradScaler('cuda') # AMP Disabled due to NaN instability
    
    
    loop = tqdm(loader, desc=f"Train Epoch [{epoch+1}/{NUM_EPOCHS}]")
    
    for batch_idx, (img1, img2, labels) in enumerate(loop):
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        labels = labels.to(DEVICE).float().view(-1, 1)
        
        
        optimizer.zero_grad()
        
        # Forward (No AMP)
        outputs, attn_map = model(img1, img2)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Gradient Clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        # scaler.update()
        
        # Metrics
        # Apply sigmoid to logits for accuracy calculation
        probs = torch.sigmoid(outputs)
        acc = calculate_accuracy(probs, labels)
        running_loss += loss.item()
        running_acc += acc.item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item(), acc=acc.item())
        
        # Logging
        global_step = epoch * len(loader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Training/Batch_Accuracy', acc.item(), global_step)

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)
    return avg_loss, avg_acc

def evaluate(model, loader, criterion, epoch, writer, split="Val"):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        loop = tqdm(loader, desc=f"{split} Epoch [{epoch+1}/{NUM_EPOCHS}]")
        for batch_idx, (img1, img2, labels) in enumerate(loop):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            labels = labels.to(DEVICE).float().view(-1, 1)
            
            outputs, attn_map = model(img1, img2)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)
            acc = calculate_accuracy(probs, labels)
            
            running_loss += loss.item()
            running_acc += acc.item()
            
            # Save visualization for the first batch of the first few epochs
            if batch_idx == 0:
                # Visualize the first pair in the batch
                save_attention_map(
                    img1[0], img2[0], attn_map[0], 
                    probs[0].item(), labels[0].item(), 
                    epoch+1, batch_idx, 
                    save_dir=os.path.join(LOG_DIR, "vis")
                )

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)
    return avg_loss, avg_acc

def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cpu':
        print("WARNING: You are training on CPU. This will be slow!")
        print("Install CUDA or use a machine with Nvidia GPU for faster results.")
    else:
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    
    # 1. Data Preparation
    print("Initializing Datasets...")
    train_dataset = LFWDataset(
        root_dir=DATASET_PATH, 
        transform=get_transforms(split='train', input_size=INPUT_SHAPE), 
        split='train',
        train_ratio=config.get("train_split_ratio", 0.8),
        cache_images=False
    )
    
    # Validation/Test Dataset (Test split from LFW)
    # In one-shot context, this is our "unseen" people evaluation
    test_dataset = LFWDataset(
        root_dir=DATASET_PATH, 
        transform=get_transforms(split='test', input_size=INPUT_SHAPE), 
        split='test',
        train_ratio=config.get("train_split_ratio", 0.8),
        cache_images=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    
    # 2. Model, Loss, Optimizer
    # We freeze the backbone to speed up training and prevent overfitting on small data
    model = AttentionalSiameseNetwork(input_shape=INPUT_SHAPE, dropout_prob=DROPOUT_PROB, freeze_backbone=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Store metrics for plotting later
    train_losses = []
    val_losses = []
    
    best_acc = 0.0

    # 3. Training Loop
    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        
        # Unfreeze Backbone Logic
        if epoch == UNFREEZE_EPOCH:
            print(f"\n[INFO] Reached epoch {epoch}. Unfreezing backbone for fine-tuning!")
            model.unfreeze_backbone()
            # We must re-initialize the optimizer to include the newly unfrozen parameters
            # We keep the low learning rate (or make it even lower if desired)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            # Re-attach scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)
        train_losses.append(train_loss)
        
        # Validate every epoch
        if True: # Run every epoch
            val_loss, val_acc = evaluate(model, test_loader, criterion, epoch, writer, split="Test")
            val_losses.append(val_loss)
            
            # Step Scheduler
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}" 
                  f"\tTrain Loss: {train_loss:.4f} \tTrain Acc: {train_acc:.4f}"
                  f"\tVal Loss: {val_loss:.4f} \tVal Acc: {val_acc:.4f}")
            
            # TensorBoard
            writer.add_scalars('Epoch/Loss', {'Train': train_loss, 'Val': val_loss}, epoch+1)
            writer.add_scalars('Epoch/Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch+1)

            # Checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
                print("Saved Best Model!")
        else:
             print(f"Epoch {epoch+1}" 
                  f"\tTrain Loss: {train_loss:.4f} \tTrain Acc: {train_acc:.4f}")
             writer.add_scalar('Epoch/Loss/Train', train_loss, epoch+1)
             writer.add_scalar('Epoch/Accuracy/Train', train_acc, epoch+1)
            
        # Save regular checkpoint
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/last_model.pth")

    print("Training Complete!")
    writer.close()
    
    # Plot final loss curve
    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(LOG_DIR, "loss_curve.png"))

if __name__ == "__main__":
    main()