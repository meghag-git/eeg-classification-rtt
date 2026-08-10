# Deep Learning EEG Binary Classification Pipeline

This repository contains the deep learning classification pipeline for classifying Rett Syndrome (RTT) vs. Healthy Control (HC) subjects using resting-state EEG recordings.

The pipeline utilizes nested, group-aware cross-validation to guarantee that subject data is strictly isolated across train/test splits, preventing data leakage and ensuring robust generalization performance.

---

## 📁 Repository Structure

* **`binary_classification_dl.py`**: The primary executable pipeline script that handles data loading, subject splitting, hyperparameter tuning, and cross-validation evaluation.
* **`utils_dl.py`**: Utility functions containing EEG signal scaling, processing, evaluation metrics, and the neural network architectures.
* **`classification_results.txt`**: Detailed logs and summaries of classification results from past runs.

---

## 🧠 Supported Deep Learning Architectures

All model architectures are optimized for processing 2D/3D EEG inputs `(batch, channels, samples)`:

1. **ShallowConvNet** (TensorFlow/Keras)[Schirrmeister et al., 2017]: A shallow convolutional network optimized for extracting temporal and spatial features.
2. **DeepConvNet** (TensorFlow/Keras)[Schirrmeister et al., 2017]: A deep network with multiple convolutional blocks followed by max-pooling. 
3. **EEGNet** (TensorFlow/Keras)[Lawhern et al., 2018]: A compact convolutional network specifically designed for EEG signals, utilizing depthwise and separable convolutions. 
4. **EEGConformer** (PyTorch/Skorch)[Song et al., 2023]: A state-of-the-art hybrid model combining shallow CNN patch embeddings with a Transformer Encoder to capture long-range temporal dependencies.

---

## ⚙️ Pipeline Overview

The pipeline executes the following 4 stages:

### Step 1: Data Loading & Preprocessing
* Loads resting-state eyes-closed EEG recordings (in `.set` format).
* Segments time-series data into **4-second windows** (1024 samples at 256Hz) across 20 channels.

### Step 2: Subject-Level DEV/EVAL Split
* To ensure zero group leakage, subjects are split into a **DEV set (30%)** for hyperparameter tuning and an **EVAL set (70%)** for final evaluation.
* This split is stratified by subject class labels to maintain a representative distribution.

### Step 3: Hyperparameter Tuning (Inner CV)
* Performs a `RandomizedSearchCV` on the DEV set using `StratifiedGroupKFold` (5-fold by default).
* Search parameters are optimized using **Balanced Accuracy** to counteract dataset class imbalance.

### Step 4: Outer Cross-Validation Evaluation
* Runs a 5-fold outer `StratifiedGroupKFold` on the EVAL set using the best hyperparameters found in Step 3.
* **Scaling Isolation**: Features are standardized on a per-channel basis using stats computed *only* on the training fold (`x_tr`) and applied to the test fold (`x_te`) to avoid scaling leakage.
* **Class Imbalance Mitigation**: Class weights are computed dynamically per fold based on training class distribution and applied to the loss function.
* **Evaluation**: Reports window-level metrics, including Accuracy, Precision, Sensitivity, Specificity, F1-Score, Balanced Accuracy, and ROC-AUC.

---

## 🚀 Usage

### Requirements
* Python 3.10+
* TensorFlow 2.x
* PyTorch 2.x
* MNE (MNE-Python)
* Scikit-Learn
* Scikeras (for Keras wrapping)
* Skorch (for PyTorch wrapping)

### Running a Classification Run
To run the classification pipeline using a specific model architecture:

```bash
python binary_classification_dl.py --model-choice DeepConvNet
```

### Configurable Command-Line Options
* `--hc-folder`: Path to Healthy Control (HC) `.set` files (default: `/home/megha/Data/EC_Segments/HC_Italy/`)
* `--rtt-folder`: Path to Rett Syndrome (RTT) `.set` files (default: `/home/megha/Data/EC_Segments/RTT_20Channels/`)
* `--model-choice`: Model to evaluate (`ShallowConvNet`, `DeepConvNet`, `EEGNet`, or `EEGConformer`).
* `--dev-frac`: Fraction of subjects reserved for hyperparameter tuning (default: `0.3`).
* `--n-iter`: Number of search iterations in `RandomizedSearchCV` (default: `10`).
* `--inner-splits`: Folds in hyperparameter tuning (default: `5`).
* `--outer-splits`: Folds in final evaluation (default: `5`).
* `--results-file`: Text file to append classifier performance summaries (default: `classification_results.txt`).

---

## References
- Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W. and Ball, T. (2017), Deep learning with convolutional neural networks for EEG decoding and visualization. Hum. Brain Mapp., 38: 5391-5420. https://doi.org/10.1002/hbm.23730
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018), EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of neural engineering, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c
- Song, Y., Zheng, Q., Liu, B., and Gao, X. (2023), EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization. IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 710-719, 2023, doi: https://doi.org/10.1109/TNSRE.2022.3230250
