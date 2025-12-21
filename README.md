# Siamese Network for One-Shot Face Verification (Assignment 2)

This project implements a **Siamese Neural Network** with an **Attention Mechanism** (ResNet18 + ResNet50 backbone) for one-shot face verification on the Labeled Faces in the Wild (LFW) dataset.

## Project Structure

```
DL_HW2/
├── config.yaml          # Hyperparameters and paths
├── requirements.txt     # Python dependencies
├── train.py             # Main training script
├── evaluate.py          # Evaluation script (Full Test Set)
├── predict.py           # Demo script for single pair
├── src/
│   ├── data/
│   │   ├── lfw_dataset.py   # Dataset loader with Subject-Independent Split
│   │   └── transforms.py    # Data augmentations
│   ├── models/
│   │   └── siamese_main.py  # Siamese Network Architecture
│   └── utils/
│       ├── metrics.py       # Accuracy, Precision, Recall, F1
│       └── visualization.py # Heatmap plotting
├── logs/                # TensorBoard logs and saved plots
└── checkpoints/         # Saved models
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    *   Download the **LFW-a** (aligned) dataset.
    *   Extract it so that images are in `data/lfw/Person_Name/image.jpg`.
    *   Ensure the path matches `dataset_path` in `config.yaml` (default: `data/lfw`).

## Running the Code

### 1. Training
The training script uses a **Subject-Independent Split** (80% of people for training, 20% for testing). It logs loss and accuracy to TensorBoard and saves the best model.

```bash
python train.py
```

*   **Outputs**:
    *   `checkpoints/best_model.pth`: Model with highest validation accuracy.
    *   `logs/loss_curve.png`: Training/Validation loss graph.
    *   `logs/vis/`: Visualization of attention maps during training.

### 2. Evaluation
To evaluate the model on the held-out test set and generate failure analysis:

```bash
python evaluate.py
```

*   **Outputs**:
    *   Terminal: Accuracy, Precision, Recall, F1 Score.
    *   `logs/failures/`: Images of False Positives and False Negatives for the report.

### 3. Prediction (Demo)
To test if two specific images belong to the same person:

```bash
python predict.py --img1 "path/to/img1.jpg" --img2 "path/to/img2.jpg"
```

## Architecture Details

*   **Attention Network**: ResNet18 (truncated) generates a spatial attention map.
*   **Feature Network**: ResNet50 (truncated) extracts deep features.
*   **Soft Gating**: The difference between features is weighted by the attention map.
*   **Loss**: Binary Cross Entropy (BCELoss).

## One-Shot Learning
The model is trained to learn a similarity metric. During testing (One-Shot), it verifies pairs of faces of people it has **never seen before** during training (disjoint identities).
