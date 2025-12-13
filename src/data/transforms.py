from torchvision import transforms

# --- קבועים סטטיסטיים של ImageNet ---
# המודלים (ResNet18/50) אומנו על ImageNet, ולכן מצפים לנרמול לפי הערכים הללו.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(split='train'):
    """
    מחזיר את אובייקט הטרנספורמציות המתאים (לאימון או לבדיקה).
    
    Args:
        split (str): 'train' או 'val'/'test'
    """
    
    # טרנספורמציות בסיסיות שקורות תמיד (גם באימון וגם בטסט)
    base_transforms = [
        # שינוי גודל ל-224x224 (הגודל הסטנדרטי ל-ResNet)
        # מכיוון שוויתרנו על MTCNN, אנחנו חייבים לוודא שהתמונה בגודל אחיד כאן.
        transforms.Resize((224, 224)),
        
        # המרה לטנסור (הופך ל-Float בין 0 ל-1, ומסדר את הערוצים ל-[C, H, W])
        transforms.ToTensor(),
        
        # נרמול מתמטי לפי הסטטיסטיקה של ImageNet
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    if split == 'train':
        # באימון, נוסיף אוגמנטציות לפני ההמרה לטנסור
        # המטרה: ליצור גיוון כדי שהמודל לא ישנן בעל פה
        aug_transforms = [
            # היפוך אופקי אקראי (כמו מראה) - 50% סיכוי
            transforms.RandomHorizontalFlip(p=0.5),
            
            # שינויי צבע עדינים (בהירות, ניגודיות) - עוזר להתמודד עם תאורה שונה
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            
            # סיבוב קל (עד 10 מעלות)
            transforms.RandomRotation(degrees=10)
        ]
        # מחברים את האוגמנטציות עם הבסיס
        final_transforms = transforms.Compose(aug_transforms + base_transforms)
    else:
        # בטסט/ולידציה אנחנו רוצים את התמונה "נקייה" ככל האפשר, רק בגודל הנכון
        final_transforms = transforms.Compose(base_transforms)

    return final_transforms