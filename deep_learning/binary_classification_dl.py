import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from collections import Counter

import tensorflow as tf
from tensorflow import keras

import torch
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils_dl import (
    set_all_seeds,
    get_safe_device,
    assign_groups,
    extract_data_SetFormat,
    log_elapsed,
    subject_level_labels,
    get_model_and_params,
    scale_per_channel,
    reshape_for_model,
    compute_metrics,
    print_metrics
)

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
# ── Data paths ────────────────────────────────────────────────────────────────
hc_folder_default  = "/home/megha/Data/EC_Segments/HC_Italy/"
rtt_folder_default = "/home/megha/Data/EC_Segments/RTT_20Channels/"

# ── Model & cross-validation ──────────────────────────────────────────────────
# Options: ShallowConvNet | DeepConvNet | EEGNet | EEGConformer
model_choice_default = "ShallowConvNet"
dev_frac_default = 0.3   # fraction of subjects reserved for hyperparameter tuning
n_iter_default = 10      # RandomizedSearchCV iterations
inner_splits_default = 5  # inner CV folds (tuning)
outer_splits_default = 5  # outer CV folds (evaluation)
MAX_EPOCHS    = 50
BATCH_SIZE    = 8
seed = 42

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run binary classification on PSD feature data with configurable model and CV settings."
    )
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments with attributes matching the script configuration
        options (folders, model choice, CV splits, seed, and results file).
    """
    parser.add_argument("--hc-folder", default=hc_folder_default, help="Path to HC .mat files")
    parser.add_argument("--rtt-folder", default=rtt_folder_default, help="Path to RTT .mat files")
    parser.add_argument(
        "--model-choice",
        default=model_choice_default,
        choices=[
            "ShallowConvNet",
            "DeepConvNet",
            "EEGNet",
            "EEGConformer",
        ],
        help="Model to evaluate",
    )
    parser.add_argument("--dev-frac", type=float, default=dev_frac_default, help="Fraction of subjects reserved for tuning")
    parser.add_argument("--n-iter", type=int, default=n_iter_default, help="RandomizedSearchCV iterations")
    parser.add_argument("--inner-splits", type=int, default=inner_splits_default, help="Inner CV folds for tuning")
    parser.add_argument("--outer-splits", type=int, default=outer_splits_default, help="Outer CV folds for evaluation")
    parser.add_argument("--max-epochs", type=int, default=seed, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=seed, help="Batch size")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed")
    parser.add_argument("--results-file", default="classification_results.txt", help="Results output file")
    return parser.parse_args()

def main():
    """Run the binary classification pipeline end to end."""
    args = parse_args()
    set_all_seeds(args.seed)
    device     = get_safe_device()

    hc_folder = Path(args.hc_folder)
    rtt_folder = Path(args.rtt_folder)
    model_choice = args.model_choice
    dev_frac = args.dev_frac
    n_iter = args.n_iter
    inner_splits = args.inner_splits
    outer_splits = args.outer_splits
    seed = args.seed
    import utils_dl
    utils_dl.MAX_EPOCHS = args.max_epochs
    utils_dl.BATCH_SIZE = args.batch_size

    results_path = Path(args.results_file)

    pipeline_start = time.time()
    _timings = {}
    _t = time.time()

    # ── Step 1 — Load data ───

    print("Loading data...")
    hc_groups, _ = assign_groups(hc_folder)
    rtt_groups, _ = assign_groups(rtt_folder)
    # subject ID per segment (HC first; RTT groups are offset to follow HC)
    rtt_groups = [x + 1 + max(hc_groups) for x in rtt_groups]

    hc_features, hc_groups  = extract_data_SetFormat(hc_folder, hc_groups)
    rtt_features, rtt_groups = extract_data_SetFormat(rtt_folder, rtt_groups)
    
    x      = np.vstack((hc_features, rtt_features))
    
    utils_dl.N_CHANS = x.shape[1]
    utils_dl.N_SAMPLES = x.shape[2]
    y      = np.hstack((np.zeros(len(hc_features)), np.ones(len(rtt_features))))
    groups = np.asarray(hc_groups + rtt_groups)
    
    log_elapsed("Step 1 — data loading", _t, _timings)
    print(f"Segments: {x.shape[0]}  |  features: {x.shape[1]}  |  "
          f"class dist: {Counter(y.astype(int))}  (HC:RTT = "
          f"{Counter(y.astype(int))[0]}:{Counter(y.astype(int))[1]})")
    print(f"Subjects: {len(np.unique(groups))}  "
          f"(HC: {len(np.unique(groups[y==0]))}, RTT: {len(np.unique(groups[y==1]))})")

    # ── Step 2 — Subject-level DEV / EVAL split (by subject IDs) ───
    subj_ids, subj_y = subject_level_labels(y, groups)
    dev_subj, eval_subj = train_test_split(
        subj_ids,
        test_size=1.0 - dev_frac,
        random_state=args.seed,
        stratify=subj_y,
    )

    dev_mask  = np.isin(groups, dev_subj)
    eval_mask = np.isin(groups, eval_subj)
    x_dev,  y_dev,  g_dev  = x[dev_mask],  y[dev_mask],  groups[dev_mask]
    x_eval, y_eval, g_eval = x[eval_mask], y[eval_mask], groups[eval_mask]

    log_elapsed("Step 2 — subject split", _t, _timings)
    print(f"DEV  — {len(np.unique(g_dev))} subjects, "
          f"{len(y_dev)} segments, class dist: {Counter(y_dev.astype(int))}")
    print(f"EVAL — {len(np.unique(g_eval))} subjects, "
          f"{len(y_eval)} segments, class dist: {Counter(y_eval.astype(int))}")

    # ── Step 3 — Hyperparameter tuning on DEV set ─────────────────────────────────
    _t = time.time()
    print(f"Tuning {model_choice} ({n_iter} iterations, {inner_splits}-fold inner CV)...")

    base_model, param_grid = get_model_and_params(model_choice)

    x_dev_scaled = scale_per_channel(x_dev)
    x_dev   = reshape_for_model(x_dev_scaled, model_choice)
    y_dev  = y_dev.astype(np.int64) if model_choice == "EEGConformer" else y_dev
    print(f"\nx_dev shape: {x_dev.shape}")

    if model_choice == "EEGConformer":
        device     = get_safe_device()
        # Criterion weights address loss imbalance during tuning inner folds.
        cw_arr    = compute_class_weight("balanced",
                        classes=np.unique(y_dev), y=y_dev)
        cw_tensor = torch.tensor(cw_arr, dtype=torch.float32).to(device)
        base_model.set_params(criterion__weight=cw_tensor)
        fit_kwargs = {}
    else:
        cw_arr     = compute_class_weight("balanced",
                        classes=np.unique(y_dev), y=y_dev)
        fit_kwargs = {"class_weight": dict(enumerate(cw_arr))}

    tuner = RandomizedSearchCV(
        base_model, 
        param_grid,
        cv=StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed),
        scoring="balanced_accuracy",
        n_iter=n_iter, 
        n_jobs=1, 
        random_state=seed, 
        verbose=2
    )
    
    _t = time.time()
    print(f"\nStarting hyperparameter tuning ({n_iter} iterations)...")
    tuner.fit(x_dev, y_dev, groups=g_dev, **fit_kwargs)
    best_params = tuner.best_params_
    log_elapsed("Step 3 — hyperparameter tuning", _t, _timings)
    print(f"Best params    : {best_params}")
    print(f"Best DEV score : {tuner.best_score_:.4f}")
    tf.keras.backend.clear_session()

    # ── Step 4 — Evaluation on EVAL set (outer CV by subject) ─────────────────────
    _t = time.time()
    _fold_times = []

    win_acc, win_prec, win_sens, win_spec, win_bacc, win_f1, win_auc = [], [], [], [], [], [], []

    # Store fitted models and fold information for later analysis
    fold_models = []
    fold_train_indices = []
    fold_test_indices = []

    print(f"Evaluating on EVAL set ({outer_splits}-fold outer CV)...")
    print(f"Reporting window-level metrics.\n")

    eval_cv = StratifiedGroupKFold(n_splits=outer_splits)

    for fold_idx, (tr_idx, te_idx) in enumerate(
        eval_cv.split(x_eval, y_eval, groups=g_eval), start=1
    ):
        _t_fold = time.time()

        x_tr, y_tr = x_eval[tr_idx], y_eval[tr_idx]
        x_te, y_te = x_eval[te_idx], y_eval[te_idx]
        g_te = g_eval[te_idx]

        x_tr, x_te  = scale_per_channel(x_tr, x_te)
        fold_cw_arr = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)

        if model_choice == "EEGConformer":
            x_tr = reshape_for_model(x_tr, model_choice)
            x_te = reshape_for_model(x_te, model_choice)
            y_tr_orig, y_te_orig = y_tr.copy(), y_te.copy()
            y_tr = y_tr.astype(np.int64)
            y_te = y_te.astype(np.int64)
    
            model, _ = get_model_and_params(model_choice, early_stop_monitor="train_loss")
    
            cw_tensor = torch.tensor(fold_cw_arr, dtype=torch.float32).to(device)
            model.set_params(criterion__weight=cw_tensor, **best_params)
            model.set_params(train_split=None)
            model.fit(x_tr, y_tr)
    
            # Predicted probabilities for test windows
            with torch.no_grad():
                logits = model.module_(torch.tensor(x_te).to(device))
                y_proba = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = y_proba.argmax(axis=1)
            y_te   = y_te_orig   # restore for metrics
    
        else:
            x_tr = reshape_for_model(x_tr, model_choice)
            x_te = reshape_for_model(x_te, model_choice)
            model, _ = get_model_and_params(model_choice, early_stop_monitor="loss")
            model.set_params(**best_params)
            model.fit(x_tr, y_tr,
                      class_weight=dict(enumerate(fold_cw_arr)))
            y_proba = model.predict_proba(x_te)
            y_pred  = y_proba.argmax(axis=1)
         
        # ── Window-level metrics ───────────────────────────────────────
        win_m = compute_metrics(y_te, y_pred, y_proba)
    
        _fold_elapsed = time.time() - _t_fold
        _fold_times.append(_fold_elapsed)
        fm, fs = divmod(_fold_elapsed, 60)

        print(f"Fold {fold_idx}  [{int(fm)}m {fs:.1f}s]")
        print_metrics("Window", win_m)
        print()

        win_acc.append(win_m["acc"])
        win_bacc.append(win_m["bacc"])
        win_auc.append(win_m["auc"])
        win_prec.append(win_m["prec"])
        win_sens.append(win_m["sens"])
        win_spec.append(win_m["spec"])
        win_f1.append(win_m["f1"])

    log_elapsed("Step 4 — classification folds", _t, _timings)

    # ── Summary ──────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    t_m, t_s = divmod(total, 60)

    print("=== Timing Summary ===")
    for stage, secs in _timings.items():
        m, s = divmod(secs, 60)
        print(f"  {stage:<42} {int(m)}m {s:.1f}s  ({100 * secs / total:.1f}%)")
    print(f"  {'Total':<42} {int(t_m)}m {t_s:.1f}s")
    if _fold_times:
        fm, fs = divmod(np.mean(_fold_times), 60)
        print(f"  Avg per eval fold: {int(fm)}m {fs:.1f}s")

    print()
    print("=== Evaluation Summary ===")
    print(f"    Accuracy     : {np.mean(win_acc):.3f} ± {np.std(win_acc):.3f}")
    print(f"    Precision     : {np.mean(win_prec):.3f} ± {np.std(win_prec):.3f}")
    print(f"    Sensitivity     : {np.mean(win_sens):.3f} ± {np.std(win_sens):.3f}")
    print(f"    Spec.     : {np.mean(win_spec):.3f} ± {np.std(win_spec):.3f}")
    print(f"    F1     : {np.mean(win_f1):.3f} ± {np.std(win_f1):.3f}")
    print(f"    Balanced Acc : {np.mean(win_bacc):.3f} ± {np.std(win_bacc):.3f}")
    if not np.all(np.isnan(win_auc)):
        print(f"    AUC          : {np.nanmean(win_auc):.3f} ± {np.nanstd(win_auc):.3f}")

    # ── Save results to text file ──────────────────────────────────────────────────
    with results_path.open("a") as f:
        f.write("=== Classifier Performance Results ===\n")
        f.write(f"Date Run            : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"HC Folder           : {hc_folder}\n")
        f.write(f"RTT Folder          : {rtt_folder}\n")
        f.write(f"Model Choice        : {model_choice}\n")
        f.write(f"Best Hyperparameters: {best_params}\n\n")
        
        f.write("=== Fold-wise Metrics ===\n")
        for fold_idx in range(outer_splits):
            f.write(f"Fold {fold_idx + 1}:\n")
            f.write(f"  Accuracy    : {win_acc[fold_idx]:.3f}\n")
            f.write(f"  Precision   : {win_prec[fold_idx]:.3f}\n")
            f.write(f"  Sensitivity : {win_sens[fold_idx]:.3f}\n")
            f.write(f"  Specificity : {win_spec[fold_idx]:.3f}\n")
            f.write(f"  F1 Score    : {win_f1[fold_idx]:.3f}\n")
            f.write(f"  Balanced Acc: {win_bacc[fold_idx]:.3f}\n")
            f.write(f"  AUC         : {win_auc[fold_idx]:.3f}\n\n")
            
        f.write("=== Final Evaluation Summary ===\n")
        f.write(f"  Accuracy    : {np.mean(win_acc):.3f} ± {np.std(win_acc):.3f}\n")
        # Write a concise, human-readable report. Appending mode is used so
        # multiple runs accumulate in the same file for later inspection.
        f.write(f"  Precision   : {np.mean(win_prec):.3f} ± {np.std(win_prec):.3f}\n")
        f.write(f"  Sensitivity : {np.mean(win_sens):.3f} ± {np.std(win_sens):.3f}\n")
        f.write(f"  Specificity : {np.mean(win_spec):.3f} ± {np.std(win_spec):.3f}\n")
        f.write(f"  F1 Score    : {np.mean(win_f1):.3f} ± {np.std(win_f1):.3f}\n")
        f.write(f"  Balanced Acc: {np.mean(win_bacc):.3f} ± {np.std(win_bacc):.3f}\n")
        if not np.all(np.isnan(win_auc)):
            f.write(f"  AUC         : {np.nanmean(win_auc):.3f} ± {np.nanstd(win_auc):.3f}\n")
        else:
            f.write("  AUC         : N/A\n")
            
        f.write("\n=== Timing Summary ===\n")
        total = time.time() - pipeline_start
        for stage, secs in _timings.items():
            m, s = divmod(secs, 60)
            f.write(f"  {stage:<42} {int(m)}m {s:.1f}s  ({100 * secs / total:.1f}%)\n")
        f.write(f"  {'Total':<42} {int(t_m)}m {t_s:.1f}s\n")
        if _fold_times:
            f.write(f"  Avg per eval fold: {int(fm)}m {fs:.1f}s\n\n")
    print(f"\nResults successfully saved to {results_path}")

if __name__ == "__main__":
    main()
