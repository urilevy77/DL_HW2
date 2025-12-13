import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionalSiameseNetwork(nn.Module):
    def __init__(self, input_shape=(224, 224)):
        super(AttentionalSiameseNetwork, self).__init__()
        
        # --- שלב 1: רשת הקשב (ResNet18) ---
        # אנו משתמשים ב-ResNet18 כדי לייצר מפות תכונות מרחביות
        resnet18 = models.resnet18(pretrained=True)
        # אנו מסירים את ה-AvgPool וה-FC הסופיים כדי לשמור על המימדים המרחביים (Spatial Dimensions)
        # הפלט של layer4 הוא בדרך כלל [Batch, 512, 7, 7] עבור תמונה של 224x224
        self.attention_backbone = nn.Sequential(*list(resnet18.children())[:-2])
        
        # שכבת עזר קטנה לכיווץ הערוצים של מפת הקשב (מ-512 ל-1)
        self.attn_conv = nn.Conv2d(512, 1, kernel_size=1) 

        # --- שלב 2: רשת התכונות העמוקות (ResNet50) ---
        resnet50 = models.resnet50(pretrained=True)
        # גם כאן, מסירים את הסוף כדי לקבל [Batch, 2048, 7, 7]
        self.feature_backbone = nn.Sequential(*list(resnet50.children())[:-2])
        
        # --- שלב 3: (Final Layer) ---
        self.final_layer = nn.Linear(1, 1)

        # הקפאת משקולות (אופציונלי - תלוי באסטרטגיית האימון שלך)
        # self._freeze_weights(self.attention_backbone)
        
    def _freeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward_attention(self, img1, img2):
        """
        מייצר את מפת הקשב המשותפת על בסיס הדמיון בין התמונות
        """
        # 1. חילוץ תכונות מרחביות מ-ResNet18
        # Output shape: [Batch, 512, 7, 7]
        f1_coarse = self.attention_backbone(img1)
        f2_coarse = self.attention_backbone(img2)
        
        # 2. חישוב דמיון (Co-Attention)
        # ישנן דרכים רבות לחשב זאת. כאן נשתמש בגישה של הכפלת אלמנטים + קונבולוציה ללמידת הדמיון
        # אנו בודקים איפה הפיקסלים "מסכימים"
        similarity_map = f1_coarse * f2_coarse # Element-wise multiplication
        
        # 3. הפיכה למפת קשב בערוץ בודד [Batch, 1, 7, 7]
        attn_map = self.attn_conv(similarity_map)
        
        # 4. נרמול לטווח [0, 1] באמצעות Sigmoid
        # פיקסלים חשובים יקבלו ~1, רקע יקבל ~0
        attn_map = torch.sigmoid(attn_map)
        
        return attn_map

    def forward(self, img1, img2):
        # --- שלב A: יצירת מפת הקשב (ResNet18) ---
        # המפה אומרת לנו: "איפה שתי התמונות דומות?"
        # Shape: [Batch, 1, 7, 7]
        attention_mask = self.forward_attention(img1, img2)
        
        # --- שלב B: חילוץ תכונות עמוק (ResNet50) ---
        # Output shape: [Batch, 2048, 7, 7]
        feat1_deep = self.feature_backbone(img1)
        feat2_deep = self.feature_backbone(img2)
        
        # --- שלב C: חישוב המרחק הגולמי ---
        # המרחק האוקלידי בריבוע (לפני סכימה מרחבית)
        # Shape: [Batch, 2048, 7, 7]
        raw_diff = torch.pow(feat1_deep - feat2_deep, 2)
        
        # --- שלב D: Soft Gating (הליבה של המודל) ---
        # אנו מכפילים את ההפרש במפת הקשב.
        # אם ה-attention_mask בפיקסל מסוים הוא 0 (רקע), ההפרש שם מתאפס.
        # אנו מבצעים Broadcasting כדי להתאים את המימדים
        gated_diff = raw_diff * attention_mask
        
        # שלב E: סכימה למרחק
        distance = torch.sum(gated_diff, dim=[1, 2, 3]) # [Batch_Size]
        
        # שלב F: המרה להסתברות עם Bias נלמד
        # אנו צריכים לשנות את הצורה ל-[Batch_Size, 1] כדי להכניס לשכבה הליניארית
        distance = distance.view(-1, 1)
        
        # מעבירים דרך השכבה הליניארית (Distance * w + bias)
        logits = self.final_layer(distance)
        
        # פונקציית אקטיבציה לקבלת הסתברות בין 0 ל-1
        prediction = torch.sigmoid(logits)
        
        # במקרים מסוימים נרצה שורש, אבל לרוב עובדים עם המרחק בריבוע או מנרמלים אותו
        return distance, attention_mask