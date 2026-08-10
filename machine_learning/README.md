# Machine Learning EEG Binary Classification Pipeline

This repository contains the machine learning classification pipeline for classifying Rett Syndrome (RTT) vs. Healthy Control (HC) subjects using power spectral density (PSD) features extracted from EEG recordings.

The pipeline utilizes nested, group-aware cross-validation to guarantee that subject data is strictly isolated across train/test splits, preventing data leakage and ensuring robust generalization performance.

---

## 📁 Repository Structure

* **`binary_classification.py`**: The primary executable pipeline script that handles data loading, subject splitting, hyperparameter tuning, and cross-validation evaluation.
* **`utils.py`**: Utility functions containing PSD feature extraction, subject grouping, evaluation metrics, and classifier configurations.
* **`classification_results.txt`**: Detailed logs and summaries of classification results from past runs.

---

## 🧠 Supported Machine Learning Models

The pipeline supports multiple traditional machine learning classifiers configured for hyperparameter tuning:

1. **SVM_linear**: Support Vector Machine with a linear kernel.
2. **SVM_rbf**: Support Vector Machine with a Radial Basis Function (RBF) kernel.
3. **kNN**: k-Nearest Neighbors classifier.
4. **Logistic_regression**: Logistic Regression with L2 regularization.
5. **Decision_tree**: Decision Tree classifier.
6. **Random_forest**: Random Forest ensemble classifier.
7. **Gradient_boosting**: Gradient Boosting classifier.
8. **XGBoost**: Extreme Gradient Boosting classifier.

---

## ⚙️ Pipeline Overview

The pipeline executes the following 4 stages:

### Step 1: Data Loading & Preprocessing
* Loads Power Spectral Density (PSD) features from MATLAB `.mat` files.
* Segments and extracts subject IDs and segment indexes from filenames (expects `FSM###_Seg#.mat` or `FSM###.mat`).
* Flattens each PSD spectrogram matrix into a 1-D feature vector.

### Step 2: Subject-Level DEV/EVAL Split
* To ensure zero group leakage, subjects are split into a **DEV set (30%)** for hyperparameter tuning and an **EVAL set (70%)** for final evaluation.
* This split is stratified by subject class labels to maintain a representative distribution.

### Step 3: Hyperparameter Tuning (Inner CV)
* Performs a `RandomizedSearchCV` on the DEV set using `StratifiedGroupKFold` (5-fold by default).
* Search parameters are optimized using **Balanced Accuracy** to counteract dataset class imbalance.

### Step 4: Outer Cross-Validation Evaluation
* Runs a 5-fold outer `StratifiedGroupKFold` on the EVAL set using the best hyperparameters found in Step 3.
* **Scaling Isolation**: For non-tree-based models, features are standardized using stats computed *only* on the training fold and applied to the test fold to avoid scaling leakage. Tree-based models (Random Forest, Gradient Boosting, XGBoost, Decision Tree) bypass this standard scaling.
* **Evaluation**: Reports window-level metrics, including Accuracy, Precision, Sensitivity (recall for RTT), Specificity (recall for HC), F1-Score, Balanced Accuracy, and ROC-AUC.

---

## 🚀 Usage

### Requirements
* Python 3.10+
* NumPy
* SciPy
* Scikit-Learn
* XGBoost

### Running a Classification Run
To run the classification pipeline using a specific model architecture:

```bash
python binary_classification.py binary_classification_dl.py --hc-folder /path/to/HC_folder --rtt-folder /path/to/RTT_folder --model-choice SVM_rbf
```

### Configurable Command-Line Options
* `--hc-folder`: Path to Healthy Control (HC) `.mat` files (default: `/home/megha/Data/PSD_Files/HC_Italy_2sec_EC/`)
* `--rtt-folder`: Path to Rett Syndrome (RTT) `.mat` files (default: `/home/megha/Data/PSD_Files/RTT_20Channels_2sec_EC/`)
* `--model-choice`: Model to evaluate (`SVM_linear`, `SVM_rbf`, `kNN`, `Logistic_regression`, `Decision_tree`, `Random_forest`, `Gradient_boosting`, or `XGBoost`). Default: `SVM_linear`.
* `--dev-frac`: Fraction of subjects reserved for hyperparameter tuning (default: `0.3`).
* `--n-iter`: Number of search iterations in `RandomizedSearchCV` (default: `20`).
* `--inner-splits`: Folds in hyperparameter tuning (default: `5`).
* `--outer-splits`: Folds in final evaluation (default: `5`).
* `--seed`: Random seed for reproducibility (default: `42`).
* `--results-file`: Text file to append classifier performance summaries (default: `classification_results.txt`).
