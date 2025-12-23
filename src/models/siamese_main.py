import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionalSiameseNetwork(nn.Module):
    def __init__(self, input_shape=(224, 224), dropout_prob=0.0, freeze_backbone=False):
        super(AttentionalSiameseNetwork, self).__init__()
        
        # --- שלב 1: רשת הקשב (ResNet18) ---
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.attention_backbone = nn.Sequential(*list(resnet18.children())[:-2])
        
        # שכבת עזר לכיווץ הערוצים (512 similarity + 512 difference)
        self.attn_conv = nn.Conv2d(1024, 1, kernel_size=1) 

        # --- שלב 2: רשת התכונות (ResNet34) ---
        resnet_features = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feature_backbone = nn.Sequential(*list(resnet_features.children())[:-2])
        
        # --- שלב 3: שכבת המרחק המתוקנת ---
        # במקום Linear, אנו מגדירים משקולת ו-Bias ידנית כדי לשלוט בכיוון הלמידה
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

        # Dropout Layer
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        # הקפאת משקולות (אם נבחר)
        if freeze_backbone:
             self._freeze_weights(self.attention_backbone)
             self._freeze_weights(self.feature_backbone)
             
        # מוודאים שהשכבות המכריעות תמיד לומדות
        self.attn_conv.requires_grad_(True)
        self.w.requires_grad_(True)
        self.b.requires_grad_(True)
        
    def _freeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward_attention(self, img1, img2):
        f1_coarse = self.attention_backbone(img1)
        f2_coarse = self.attention_backbone(img2)
        
        similarity_map = f1_coarse * f2_coarse
        difference = torch.abs(f1_coarse - f2_coarse)
        combined = torch.cat([similarity_map, difference], dim=1)
        
        attn_map = self.attn_conv(combined)
        return torch.sigmoid(attn_map)

    def forward(self, img1, img2):
        # יצירת מפת קשב
        attention_mask = self.forward_attention(img1, img2)
        
        # חילוץ תכונות
        feat1_deep = self.feature_backbone(img1)
        feat2_deep = self.feature_backbone(img2)
        
        feat1_deep = self.dropout(feat1_deep)
        feat2_deep = self.dropout(feat2_deep)
        
        feat1_deep = F.normalize(feat1_deep, p=2, dim=1)
        feat2_deep = F.normalize(feat2_deep, p=2, dim=1)
        
        # חישוב מרחק L1
        raw_diff = torch.abs(feat1_deep - feat2_deep) 
        
        # החלת ה-Attention
        gated_diff = raw_diff * attention_mask
        
        # מרחק ממוצע (0-1 בערך)
        distance = torch.mean(gated_diff, dim=[1, 2, 3]) 
        distance = distance.view(-1, 1)
        
        # --- תיקון לוגי קריטי ---
        # ככל שהמרחק גדל, הלוגיט קטן (מקרב ל-0 בהסתברות)
        # זה מתאים ל-BCEWithLogitsLoss שבו '1' זה דומים ו-'0' זה שונים
        logits = -self.w * distance + self.b
        
        return logits, attention_mask