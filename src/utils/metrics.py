import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_accuracy(predictions, labels):
    """
    Calculates accuracy.
    predictions: Tensor of shape [Batch, 1] (probabilities 0-1)
    labels: Tensor of shape [Batch, 1] (0 or 1)
    """
    predicted_classes = (predictions > 0.5).float()
    correct = (predicted_classes == labels).float().sum()
    return correct / labels.size(0)

def calculate_precision_recall_f1(predictions, labels):
    """
    Calculates Precision, Recall, and F1 score.
    Returns a dictionary.
    """
    # Convert to numpy for sklearn
    preds_np = (predictions.detach().cpu().numpy() > 0.5).astype(int).flatten()
    labels_np = labels.detach().cpu().numpy().astype(int).flatten()
    
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
