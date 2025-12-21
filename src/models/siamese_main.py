import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionalSiameseNetwork(nn.Module):
    def __init__(self, input_shape=(224, 224), dropout_prob=0.0, freeze_backbone=False):
        super(AttentionalSiameseNetwork, self).__init__()
        
        # --- שלב 1: רשת הקשב (ResNet18) ---
        # אנו משתמשים ב-ResNet18 כדי לייצר מפות תכונות מרחביות
        # weights=models.ResNet18_Weights.DEFAULT equivalent to pretrained=True
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # אנו מסירים את ה-AvgPool וה-FC הסופיים כדי לשמור על המימדים המרחביים (Spatial Dimensions)
        # הפלט של layer4 הוא בדרך כלל [Batch, 512, 7, 7] עבור תמונה של 224x224
        self.attention_backbone = nn.Sequential(*list(resnet18.children())[:-2])
        
        # שכבת עזר קטנה לכיווץ הערוצים של מפת הקשב (מ-512 ל-1)
        self.attn_conv = nn.Conv2d(512, 1, kernel_size=1) 

        # --- שלב 2: רשת התכונות העמוקות (ResNet34 - Balanced) ---
        # User requested "ResNet32" (ResNet34 is the standard pytorch equivalent)
        # This is deeper than 18 but faster than 50.
        resnet_features = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # ResNet34 output is [Batch, 512, 7, 7] (same channels as 18, but deeper features)
        self.feature_backbone = nn.Sequential(*list(resnet_features.children())[:-2])
        
        # --- שלב 3: (Final Layer) ---
        self.final_layer = nn.Linear(1, 1)

        # Dropout Layer
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # הקפאת משקולות (אופציונלי - תלוי באסטרטגיית האימון שלך)
        if freeze_backbone:
             self._freeze_weights(self.attention_backbone)
             self._freeze_weights(self.feature_backbone)
             
        # Ensure these are always trainable
        for param in self.attn_conv.parameters():
            param.requires_grad = True
        for param in self.final_layer.parameters():
            param.requires_grad = True
        
    def _freeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the backbone for fine-tuning."""
        print("Unfreezing backbones for fine-tuning...")
        for param in self.attention_backbone.parameters():
            param.requires_grad = True
        for param in self.feature_backbone.parameters():
            param.requires_grad = True

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
        
        # --- שלב B: חילוץ תכונות עמוק (ResNet34) ---
        # Output shape: [Batch, 512, 7, 7]
        feat1_deep = self.feature_backbone(img1)
        feat2_deep = self.feature_backbone(img2)
        
        # Apply Dropout to feature maps
        feat1_deep = self.dropout(feat1_deep)
        feat2_deep = self.dropout(feat2_deep)
        
        # --- שלב C: חישוב המרחק הגולמי ---
        # המרחק האוקלידי בריבוע (לפני סכימה מרחבית)
        # Shape: [Batch, 512, 7, 7]
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
        
        # Return Logits directly for BCEWithLogitsLoss
        return logits, attention_mask