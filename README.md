# Siamese Network for One-Shot Face Verification (Assignment 2)

This project implements a **Siamese Neural Network** with an **Attention Mechanism** (ResNet18 + ResNet34 backbone) for one-shot face verification on the Labeled Faces in the Wild (LFW) dataset.

## Project Structure

```
DL_HW2/
├── config.yaml          # Hyperparameters and paths
├── requirements.txt     # Python dependencies
├── train.py             # Main training script
├── evaluate.py          # Evaluation script (Full Test Set)
├── predict.py           # Demo script for single pair
├── data/
│   ├── lfw-deepfunneled/        # LFW-a aligned images
│   ├── peopleDevTrain.csv       # Training identities (4,039 people)
│   ├── peopleDevTest.csv        # Test identities (1,712 people)
│   ├── matchpairsDevTrain.csv   # Positive training pairs
│   └── mismatchpairsDevTrain.csv # Negative training pairs
├── src/
│   ├── data/
│   │   ├── lfw_dataset.py   # Dataset loader with CSV-based splits
│   │   └── transforms.py    # Data augmentations
│   ├── models/
│   │   └── siamese_main.py  # Siamese Network Architecture
│   └── utils/
│       ├── analyze_dataset.py   # Dataset statistics generator
│       ├── document_setup.py    # Experimental setup documentation
│       ├── metrics.py           # Accuracy, Precision, Recall, F1
│       └── visualization.py     # Heatmap plotting
├── logs/                # TensorBoard logs and saved plots
└── checkpoints/         # Saved models
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    *   Download the **LFW-a** (deepfunneled) dataset from [official LFW site](http://vis-www.cs.umass.edu/lfw/).
    *   Extract it to `data/lfw-deepfunneled/lfw-deepfunneled/`.
    *   Ensure CSV files (`peopleDevTrain.csv`, `peopleDevTest.csv`, etc.) are in `data/`.

## Running the Code

### 1. Dataset Analysis
Generate comprehensive dataset statistics for your report:

```bash
python src/utils/analyze_dataset.py
```

*   **Outputs**: Train/test statistics, class distribution, histograms saved to `dataset_distribution.png`

### 2. Experimental Setup Documentation
Generate complete experimental setup for your report:

```bash
python src/utils/document_setup.py
```

*   **Outputs**: All hyperparameters, architecture details, training strategy

### 3. Training
The training script uses **CSV-based train/test splits** with 80/20 train/validation division. It logs to TensorBoard and saves the best model.

```bash
python scripts/train.py
```

*   **Outputs**:
    *   `checkpoints/best_model.pth`: Model with highest validation accuracy
    *   `logs/`: TensorBoard logs and visualizations

### 4. Evaluation
Evaluate the model on the held-out test set and generate failure analysis:

```bash
python scripts/evaluate.py
```

*   **Outputs**:
    *   Terminal: Accuracy, Precision, Recall, F1 Score
    *   `logs/failures/`: Images of False Positives and False Negatives

### 5. N-Way One-Shot Evaluation ⭐
**The core one-shot learning metric** from Koch et al. (2015):

```bash
python scripts/one_shot_evaluation.py --n_way 20 --n_trials 400
```

*   **Arguments**:
    *   `--n_way`: Number of classes (default: 20)
    *   `--n_trials`: Number of test episodes (default: 400)
    *   `--checkpoint`: Model path (default: `../checkpoints/best_model.pth`)
    *   `--output`: Results JSON (default: `../results/oneshot_results.json`)

*   **Outputs**:
    *   Terminal: N-way accuracy vs. random baseline
    *   `results/oneshot_results.json`: Detailed trial-by-trial results

**Example**: 5-way test with 50 trials (quick test):
```bash
python scripts/one_shot_evaluation.py --n_way 5 --n_trials 50
```

### 6. Prediction (Demo)
Test if two specific images belong to the same person:

```bash
python scripts/predict.py --img1 "path/to/img1.jpg" --img2 "path/to/img2.jpg"
```

## Architecture Details

*   **Attention Network**: ResNet18 (pretrained, truncated) generates a spatial attention map
*   **Feature Network**: ResNet34 (pretrained, truncated) extracts deep features
*   **Soft Gating**: Element-wise multiplication of L2 distance with attention map
*   **Loss**: Binary Cross Entropy with Logits (BCEWithLogitsLoss)
*   **Total Parameters**: ~32.4M

## Dataset Split Strategy

*   **Train identities**: 4,039 people (from `peopleDevTrain.csv`)
*   **Test identities**: 1,712 people (from `peopleDevTest.csv`)
*   **Validation**: 20% of training identities
*   **Zero overlap**: No person appears in both train and test sets

## One-Shot Learning
The model learns a similarity metric to verify pairs of faces from people it has **never seen during training** (disjoint identities).

