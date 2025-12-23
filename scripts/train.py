import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
import sys
import yaml
from tqdm import tqdm
import time
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Custom Imports
from src.models.siamese_main import AttentionalSiameseNetwork
from src.data.lfw_dataset import LFWDataset
from src.data.transforms import get_transforms
from src.utils.metrics import calculate_accuracy
from src.utils.visualization import save_attention_map, plot_loss_curve

# --- Load Configuration ---
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
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
        labels = labels.to(DEVICE, non_blocking=True).float().view(-1, 1)
        
        
        optimizer.zero_grad()
        
        # Forward (No AMP)
        outputs, attn_map = model(img1, img2)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Gradient Clipping to prevent NaN
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
            writer.add_scalar('Training/Gradient_Norm', grad_norm.item(), global_step)
            
            # Prediction distribution monitoring - detect if stuck at ~0.5
            writer.add_scalar('Training/Pred_Min', probs.min().item(), global_step)
            writer.add_scalar('Training/Pred_Max', probs.max().item(), global_step)
            writer.add_scalar('Training/Pred_Mean', probs.mean().item(), global_step)
            
        # Print prediction stats every 50 batches for quick diagnosis
        if batch_idx % 50 == 0:
            print(f"  [Batch {batch_idx}] Predictions: min={probs.min().item():.3f}, "
                  f"max={probs.max().item():.3f}, mean={probs.mean().item():.3f}")

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)
    return avg_loss, avg_acc

def evaluate(model, loader, criterion, epoch, writer, split="Val"):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    
    # משתנים לשמירת מפת קשב לדוגמה
    sample_vis_done = False

    with torch.no_grad():
        loop = tqdm(loader, desc=f"{split} Epoch [{epoch+1}/{NUM_EPOCHS}]")
        for batch_idx, (img1, img2, labels) in enumerate(loop):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            labels = labels.to(DEVICE, non_blocking=True).float().view(-1, 1)
            
            # Forward pass
            outputs, attn_map = model(img1, img2)
            loss = criterion(outputs, labels)
            
            # איסוף נתונים לחישוב מדדים בסוף ה-Epoch
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            running_loss += loss.item()
            
            # ויזואליזציה פעם אחת ב-Epoch
            if not sample_vis_done and batch_idx == 0:
                save_attention_map(
                    img1[0], img2[0], attn_map[0], 
                    probs[0].item(), labels[0].item(), 
                    epoch+1, batch_idx, 
                    save_dir=os.path.join(LOG_DIR, "vis")
                )
                sample_vis_done = True

    # חישוב מדדים מתקדמים
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 1. חישוב עקום ROC ו-AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 2. מציאת הסף האופטימלי (Youden's J statistic)
    # J = TPR - FPR. הנקודה המקסימלית היא הסף הכי מאוזן
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 3. חישוב Accuracy לפי הסף האופטימלי שנמצא
    preds = (all_probs >= optimal_threshold).astype(float)
    acc = (preds == all_labels).mean()
    
    avg_loss = running_loss / len(loader)

    # לוג ל-TensorBoard
    writer.add_scalar(f'{split}/Loss', avg_loss, epoch)
    writer.add_scalar(f'{split}/Accuracy_Optimal', acc, epoch)
    writer.add_scalar(f'{split}/AUC', roc_auc, epoch)
    writer.add_scalar(f'{split}/Optimal_Threshold', optimal_threshold, epoch)

    print(f"  [{split}] Loss: {avg_loss:.4f} | AUC: {roc_auc:.4f} | "
          f"Acc: {acc:.4f} (at threshold {optimal_threshold:.3f})")
    
    return avg_loss, acc, roc_auc, optimal_threshold

def calculate_detailed_metrics(all_labels, all_probs, threshold):
    # הפיכת הסתברויות לחיזויים סופיים לפי הסף
    preds = (all_probs >= threshold).astype(int)
    
    # חישוב המטריצה: מחזירה [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    
    # חישוב מדדים נגזרים שחשובים למאמרים
    fpr = fp / (fp + tn) # False Positive Rate
    fnr = fn / (fn + tp) # False Negative Rate (Miss Rate)
    
    print(f"📊 Confusion Matrix Results:")
    print(f"   True Negatives (Correctly Rejected): {tn}")
    print(f"   False Positives (False Alarms):      {fp}")
    print(f"   True Positives (Correctly Matched):  {tp}")
    print(f"   False Negatives (Missed Matches):    {fn}")
    
    return tn, fp, fn, tp

def plot_confusion_matrix(tn, fp, fn, tp, epoch, writer):
    cm = [[tn, fp], [fn, tp]]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Different', 'Same'], 
                yticklabels=['Different', 'Same'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix Epoch {epoch}')
    
    # הוספה ל-TensorBoard
    writer.add_figure('Validation/Confusion_Matrix', plt.gcf(), epoch)
    plt.close()

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
        split='train',
        train_val_split=config.get("train_split_ratio", 0.8),
        transform=get_transforms(split='train', input_size=INPUT_SHAPE),
        cache_images=False
    )
    
    # Validation Dataset (20% of training people)
    val_dataset = LFWDataset(
        root_dir=DATASET_PATH, 
        split='val',
        train_val_split=config.get("train_split_ratio", 0.8),
        transform=get_transforms(split='test', input_size=INPUT_SHAPE),
        cache_images=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    
    # 2. Model, Loss, Optimizer
    # We freeze the backbone to speed up training and prevent overfitting on small data
    model = AttentionalSiameseNetwork(input_shape=INPUT_SHAPE, dropout_prob=DROPOUT_PROB, freeze_backbone=False).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Store metrics for plotting later
    train_losses = []
    val_losses = []
    
    best_acc = 0.0

    # 3. Training Loop
    print("Starting Training...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        # Unfreeze Backbone Logic
        if epoch == UNFREEZE_EPOCH:
            print(f"\n[INFO] Reached epoch {epoch}. Unfreezing backbone for fine-tuning!")
           
            # We must re-initialize the optimizer to include the newly unfrozen parameters
            # We keep the low learning rate (or make it even lower if desired)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            # Re-attach scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)
        train_losses.append(train_loss)
        
        # Validate every epoch
        if True: # Run every epoch
           val_loss, val_acc, val_auc, opt_thresh = evaluate(model, val_loader, criterion, epoch, writer, split="Val")
        
        # עדכון ה-Scheduler
        scheduler.step(val_loss)

        # שמירת המודל הכי טוב לפי AUC (המדד הכי אמין)
        if val_auc > best_acc: 
            best_acc = val_auc 
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"🌟 New Best Model saved with AUC: {val_auc:.4f}")

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

    total_time = time.time() - start_time
    print(f"Training Complete! Total duration: {total_time/60:.2f} minutes")
    writer.close()
    
    # Plot final loss curve
    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(LOG_DIR, "loss_curve.png"))

if __name__ == "__main__":
    main()