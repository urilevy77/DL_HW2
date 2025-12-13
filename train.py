import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
from tqdm import tqdm  # פס התקדמות יפה

# ייבוא המודל וה-Dataset שלך
# (הנחה: הקבצים נמצאים בתיקיות src/models ו-src/data כפי שהגדרנו)
from src.models.siamese_main import AttentionalSiameseNetwork
from src.data.lfw_dataset import LFWDataset 

# --- הגדרת קבועים (Hyperparameters) ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # קצב למידה התחלתי ל-Adam
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# יצירת תיקיות אם לא קיימות
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def calculate_accuracy(predictions, labels):
    """
    חישוב דיוק פשוט: אם ההסתברות > 0.5 זה 'אותו אדם' (1)
    """
    # predictions הם הסתברויות (בין 0 ל-1)
    predicted_classes = (predictions > 0.5).float()
    correct = (predicted_classes == labels).float().sum()
    return correct / labels.size(0)

def train():
    print(f"Starting training on device: {DEVICE}")
    
    # 1. הכנת הדאטה (Transforms & DataLoader)
    # שים לב: המודל מצפה ל-224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # טעינת ה-Dataset (נניח שמימשנו אותו כבר)
    # root_dir צריך להצביע לתיקיית LFW שלך
    train_dataset = LFWDataset(root_dir='data/raw/lfw', transform=transform, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. אתחול המודל
    model = AttentionalSiameseNetwork().to(DEVICE)
    
    # 3. הגדרת Loss ו-Optimizer
    # אנו משתמשים ב-BCELoss כי המודל מוציא Sigmoid (הסתברות)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. הגדרת TensorBoard
    writer = SummaryWriter(log_dir=LOG_DIR)

    # --- לולאת האימון ---
    for epoch in range(NUM_EPOCHS):
        model.train() # מעבר למצב אימון (חשוב ל-Dropout/BatchNorm)
        running_loss = 0.0
        running_acc = 0.0
        
        # שימוש ב-tqdm לפס התקדמות
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for batch_idx, (img1, img2, labels) in enumerate(loop):
            # העברת נתונים ל-GPU
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            # שינוי צורת ה-Labels ל-[Batch, 1] ושינוי ל-Float
            labels = labels.to(DEVICE).float().view(-1, 1)
            
            # --- Forward Pass ---
            optimizer.zero_grad() # איפוס גרדיאנטים קודמים
            
            # המודל מחזיר גם את התחזית וגם את מפת הקשב
            outputs, attn_map = model(img1, img2)
            
            # --- חישוב Loss ---
            main_loss = criterion(outputs, labels)
            
            # (אופציונלי) רגולריזציה על מפת הקשב
            # נרצה למנוע מצב שהמודל מסמן את "הכל" כחשוב (מפה של הכל 1)
            # sparsity_loss = torch.mean(attn_map) * 0.001 
            # loss = main_loss + sparsity_loss
            loss = main_loss # כרגע נשתמש רק בלוס העיקרי
            
            # --- Backward Pass & Optimization ---
            loss.backward()
            optimizer.step()
            
            # --- ניטור ומעקב ---
            acc = calculate_accuracy(outputs, labels)
            running_loss += loss.item()
            running_acc += acc.item()
            
            # עדכון פס ההתקדמות
            loop.set_postfix(loss=loss.item(), accuracy=acc.item())
            
            # כתיבה ל-TensorBoard כל 10 באצ'ים
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                writer.add_scalar('Training/Loss', loss.item(), global_step)
                writer.add_scalar('Training/Accuracy', acc.item(), global_step)

        # סיכום אפוק
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        
        print(f"Epoch {epoch+1} Finished. Avg Loss: {epoch_loss:.4f}, Avg Acc: {epoch_acc:.4f}")
        
        # שמירת המודל בסוף כל אפוק
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_epoch_{epoch+1}.pth")

    print("Training Complete!")
    writer.close()

if __name__ == "__main__":
    train()