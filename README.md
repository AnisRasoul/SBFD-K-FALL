# SBFD-K-FALL 🎯

Sensor-Based Fall Detection using the **K-FALL Dataset** from Kaggle.

This repository contains:
- **train.py** — Training script for fall detection models (Random Forest, Simple NN, LSTM)
- **CI/CD Pipeline** — GitHub Actions workflow that automatically:
  - Downloads the K-FALL dataset from Kaggle on GitHub's servers
  - Processes it into the required format
  - Trains all models
  - Generates CML reports with metrics and visualizations

## Quick Start

### 1. Prerequisites
- Python 3.9+
- GitHub repository with Actions enabled

### 2. Create GitHub Repository

```bash
# Initialize git in this directory
git init
git add .
git commit -m "Initial commit: K-FALL fall detection pipeline"

# Create empty repository on GitHub (visit github.com/new)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/SBFD-K-FALL.git
git branch -M main
git push -u origin main
```

### 3. How It Works

When you push to any branch, GitHub Actions automatically:
1. **Downloads** the K-FALL dataset directly from Kaggle (no local upload needed!)
2. **Processes** the dataset into the required CSV structure
3. **Trains** three models:
   - Random Forest (200 trees, depth-15)
   - Simple Neural Network (MLP)
   - LSTM (sequence model)
4. **Posts** a detailed CML report to your PR/commit with:
   - Confusion matrices
   - Training curves
   - Feature importance
   - Performance metrics (accuracy, F1, recall, precision, ROC-AUC)

### 4. Monitor Training

Open the **Actions** tab in your GitHub repository to see real-time progress, or view the CML report in the PR/commit comments.

## Dataset

- **Source**: [K-FALL Dataset on Kaggle](https://www.kaggle.com/datasets/usmanabbasi2002/kfall-dataset)
- Downloaded automatically in CI/CD, no manual download needed
- Processed into trial-based structure expected by train.py

## Models

### 1. **Random Forest (200 trees)**
- Primary production model
- Feature-based (80 statistical features per window)
- Excellent interpretability via feature importance

### 2. **Simple Neural Network (MLP)**
- 3 hidden layers (128 → 64 → 32 units)
- Batch normalization + dropout for regularization
- Fast inference

### 3. **LSTM**
- Captures temporal dependencies
- Processes raw sensor sequences (100-sample windows)
- Best for detecting gradual fall patterns

## Configuration

Edit `train.py` constants to tune:
- `THRESHOLD` (0.35) — classification threshold (higher = fewer false positives)
- `WINDOW_SIZE` (100) — samples per inference window
- `STEP_SIZE` (50) — window hop size (50% overlap)
- Model hyperparameters (tree counts, layer sizes, etc.)

## Output Files Generated

- **metrics.txt** — Detailed performance report
- **metrics.json** — Machine-readable metrics
- **confusion_matrices.png** — Per-model confusion matrix heatmaps
- **training_history.png** — Loss/accuracy curves
- **feature_importance.png** — Top-20 important features
- **scaler_params.json** — Preprocessing parameters (for firmware)
- **scaler_params.h** — C header with scaler (for embedded devices)
- **fall_detector_*.pkl** — Trained models (pickle format)
- **fall_detector_*.h/.c** — Microcontroller-friendly exports

## Next Steps

1. Push to GitHub: `git push -u origin main`
2. Wait for Actions to complete (first run may take 30+ minutes for dataset download)
3. Check the CML report in the commit/PR comments
4. Iterate: adjust hyperparameters, push again, compare metrics in CML reports

---

**Note**: The K-FALL dataset is downloaded on GitHub's servers during CI, eliminating your local bandwidth limitations. Perfect for large datasets! 🚀
