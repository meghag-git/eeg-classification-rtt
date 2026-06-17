import os
import re
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.io import loadmat
from scipy.stats import randint

# Model imports
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Metric imports
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score
)

def parse_subject_id(filename: str) -> str:
    """Normalize subject identifiers from a filename.

    Supports names such as "FSM001_Seg1.mat" and "FSM001.mat".
    """
    stem = Path(filename).stem
    if "_Seg" in stem:
        stem = stem.split("_Seg")[0]
    match = re.match(r"^(FSM\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return stem.strip()


def sort_key(filename: str) -> tuple[str, int]:
    """Return a tuple (subject_id, segment_number) usable as a sort key.

    Filenames without an explicit segment are sorted after those with a
    segment number (segment number becomes infinity).
    """
    subject = parse_subject_id(filename)
    segment_match = re.search(r"_Seg(\d+)", filename, flags=re.IGNORECASE)
    segment = int(segment_match.group(1)) if segment_match else float("inf")
    return subject, segment


def assign_groups(folder_path: Union[str, Path]) -> Tuple[List[int], List[str]]:
    """Assign group labels to files in a folder based on subject membership.

    Returns a label index for every .mat file and the sorted list of unique subject IDs.
    """
    folder_path = Path(folder_path)
    file_list = sorted(
        [path.name for path in folder_path.iterdir() if path.suffix.lower() == ".mat"],
        key=sort_key,
    )
    subject_ids = [parse_subject_id(filename) for filename in file_list]

    unique_ids: List[str] = []
    seen: set[str] = set()
    for subject_id in subject_ids:
        if subject_id not in seen:
            seen.add(subject_id)
            unique_ids.append(subject_id)

    subject_to_group: Dict[str, int] = {
        subject: index for index, subject in enumerate(unique_ids)
    }
    group_labels: List[int] = [subject_to_group[subject_id] for subject_id in subject_ids]
    return group_labels, unique_ids


def extract_data(folder_path: Union[str, Path]) -> np.ndarray:
    """Load PSD data from a folder and flatten each file into a feature vector."""
    folder_path = Path(folder_path)
    file_list = sorted(
        [path for path in folder_path.iterdir() if path.suffix.lower() == ".mat"],
        key=lambda path: sort_key(path.name),
    )
    features: List[np.ndarray] = []

    for file_path in file_list:
        # Load MATLAB file and extract the PSD array stored under the key
        # "psds". Flatten to a 1-D feature vector per file/segment.
        data = loadmat(file_path)
        psds = data["psds"]
        features.append(psds.flatten())

    return np.vstack(features) if features else np.empty((0, 0))


def get_model_and_params(model_choice: str) -> Tuple[object, dict]:
    """Return a base estimator and its hyperparameter grid for a model choice."""
    # Note: parameter names are formatted for use inside a sklearn `Pipeline`
    # where the estimator step is named 'clf' (hence the 'clf__' prefix).
    if model_choice == "SVM_linear":
        model = svm.SVC(kernel="linear", probability=True)
        model_params = {
            "clf__C": [0.1, 1, 10, 100],
            "clf__class_weight": [
                {0: 1, 1: 1},
                {0: 2, 1: 1},
                {0: 3, 1: 1},
                {0: 5, 1: 1},
                {0: 8, 1: 1},
                "balanced",
            ],
        }
    elif model_choice == "SVM_rbf":
        model = svm.SVC(kernel="rbf", probability=True)
        model_params = {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", 1e-3, 1e-2, 1e-1],
            "clf__class_weight": [
                {0:1, 1:1},
                {0:2, 1:1},
                {0:3, 1:1},
                {0:5, 1:1},
                {0:8, 1:1},
                "balanced"
            ]
        }
    # ── Model: Logistic Regression
    elif model_choice == 'Logistic_regression':
        model = LogisticRegression(max_iter=5000)
        model_params = {
            "clf__C": np.logspace(-4, 4, 9).tolist(),
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
            "clf__class_weight": [
                {0:1, 1:1},
                {0:2, 1:1},
                {0:3, 1:1},
                {0:5, 1:1},
                {0:8, 1:1},
                "balanced"
            ]
        }

    # ── Model: kNN
    elif model_choice == 'kNN':
        model = KNeighborsClassifier()
        model_params = {
            "clf__n_neighbors": randint(3, 16),   # 3 to 15
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],   # Manhattan / Euclidean
            "clf__metric": ["minkowski"]
        }
    # ── Model: Decision Tree
    elif model_choice == 'Decision_tree':
        model = DecisionTreeClassifier()
        model_params = {'clf__max_depth': randint(2, 10),
                      'clf__min_samples_split': randint(4, 40),
                      'clf__min_samples_leaf': randint(2, 15),
                      'clf__criterion': ['gini', 'entropy'],
                      "clf__class_weight": [
                        {0:1, 1:1},
                        {0:2, 1:1},
                        {0:0.25, 1:1},
                        {0:3, 1:1},
                        {0:5, 1:1},
                        {0:8, 1:1},
                        "balanced"
                    ]
                       }
    # ── Model: Random Forest
    elif model_choice == 'Random_forest':
        model = RandomForestClassifier()
        model_params = {'clf__n_estimators':randint(100, 1200),
                        'clf__max_depth': randint(2, 10),
                        'clf__min_samples_split': randint(4, 40),
                        'clf__min_samples_leaf': randint(2, 15),
                        'clf__criterion': ['gini', 'entropy'],
                        "clf__class_weight": [
                            {0:1, 1:1},
                            {0:2, 1:1},
                            {0:3, 1:1},
                            {0:5, 1:1},
                            {0:8, 1:1},
                            "balanced"
                        ]
                       }
    # ── Model: Gradient Boosting
    elif model_choice == 'Gradient_boosting':
        model = GradientBoostingClassifier(subsample=0.8, max_features='sqrt')
        model_params = {'clf__n_estimators': randint(100, 1200),  
                        'clf__max_depth': randint(2, 8),
                        'clf__min_samples_split': randint(4, 50),
                        'clf__min_samples_leaf': randint(2, 15),
                        'clf__learning_rate': [0.01, 0.1, 0.2],
                       }
    # ── Model: XGBoost
    elif model_choice == 'XGBoost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model_params = {'clf__n_estimators': randint(100, 1200),
                        'clf__max_depth': randint(2, 8),
                        "clf__min_child_weight": randint(1, 8),
                        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'clf__subsample': [0.5, 0.7, 0.9, 1],
                        "clf__colsample_bytree": [0.2, 0.4, 0.6],
                        "clf__gamma": [0.0, 0.1, 0.3],
                        "clf__reg_lambda": [1.0, 5.0, 10.0],
                        "clf__reg_alpha": [0.0, 0.01, 0.1],
                        "clf__scale_pos_weight": [0.25, 0.4, 0.4125, 0.5, 0.75, 1.0]}
    else:
        raise ValueError(
            f"Unknown model_choice: {model_choice}. "
            "Valid options are: SVM_linear, SVM_rbf, kNN, Logistic_regression, "
            "Decision_tree, Random_forest, Gradient_boosting, XGBoost."
        )
    return model, model_params


def set_all_seeds(seed=42):
    """Set RNG seeds for the python `random` module and numpy.

    This function ensures repeatability for operations that rely on these
    random sources (e.g., splits, randomized search sampling).
    """
    random.seed(seed)
    np.random.seed(seed)


def subject_level_labels(y: np.ndarray, groups: np.ndarray):
    """Return (unique_subject_ids, one_label_per_subject)."""
    subj_ids = np.unique(groups)
    subj_labels = []
    for sid in subj_ids:
        labels = np.unique(y[groups == sid])
        if len(labels) != 1:
            raise ValueError(f"Subject {sid} has mixed labels: {labels}.")
        subj_labels.append(int(labels[0]))
    return subj_ids, np.array(subj_labels, dtype=int)
    

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray) -> dict:
    """Compute evaluation metrics and return as a dict.

    Returns a dictionary with keys:
      - acc: overall accuracy
      - prec: macro-averaged precision
      - sens: recall for the positive class (RTT)
      - spec: recall for the negative class (HC)
      - f1: macro F1 score
      - bacc: balanced accuracy
      - auc: area under ROC (or np.nan if not computable)
    """
    metrics = {
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "sens": recall_score(y_true, y_pred, average=None, zero_division=0)[1],  # RTT recall
        "spec": recall_score(y_true, y_pred, average=None, zero_division=0)[0],  # HC recall
        "f1":   f1_score(y_true, y_pred, average="macro", zero_division=0),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "auc":  np.nan,
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
    return metrics


def get_proba(model, X_te: np.ndarray) -> np.ndarray:
    """Return (N, 2) probability array regardless of model type."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_te)
    # decision_function → convert scores to pseudo-probabilities via sigmoid
    scores = model.decision_function(X_te)
    p = 1 / (1 + np.exp(-scores))
    return np.column_stack([1 - p, p])


def log_elapsed(label: str, start: float, timings: dict) -> None:
    """Record elapsed time for `label` into `timings` and print a summary."""
    secs = time.time() - start
    timings[label] = secs
    m, s = divmod(secs, 60)
    print(f"  ⏱  {label}: {int(m)}m {s:.1f}s")


def print_metrics(label: str, m: dict) -> None:
    """Print a concise, human-readable metric summary line.

    `label` is a short descriptor (e.g., 'Segment' or 'Subject'). `m` is the
    metrics dict produced by `compute_metrics`.
    """
    print(f"  {label} | Acc: {m['acc']:.3f}  Prec: {m['prec']:.3f}  "
          f"Sens.: {m['sens']:.3f} Spec.: {m['spec']:.3f}  F1: {m['f1']:.3f}  "
          f"AUC: {m['auc']:.3f}  BAcc: {m['bacc']:.3f}")
