import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    assign_groups,
    extract_data,
    get_model_and_params,
    set_all_seeds,
    subject_level_labels,
    compute_metrics,
    get_proba,
    log_elapsed,
    print_metrics,
)

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Data paths ────────────────────────────────────────────────────────────────
hc_folder_default  = BASE_DIR / "Dataset" / "HC_Italy"
rtt_folder_default = BASE_DIR / "Dataset" / "RTT_Italy"

# ── Model & cross-validation ──────────────────────────────────────────────────
# Options: SVM_linear | SVM_rbf | kNN | Logistic_regression | Decision_tree | Random_forest | Gradient_boosting | XGBoost
model_choice_default = "SVM_linear"
dev_frac_default = 0.3   # fraction of subjects reserved for hyperparameter tuning
n_iter_default = 20      # RandomizedSearchCV iterations
inner_splits_default = 5  # inner CV folds (tuning)
outer_splits_default = 5  # outer CV folds (evaluation)

tree_models = {"Random_forest", "Gradient_boosting", "XGBoost", "Decision_tree"}
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
            "SVM_linear",
            "SVM_rbf",
            "kNN",
            "Logistic_regression",
            "Decision_tree",
            "Random_forest",
            "Gradient_boosting",
            "XGBoost",
        ],
        help="Model to evaluate",
    )
    parser.add_argument("--dev-frac", type=float, default=dev_frac_default, help="Fraction of subjects reserved for tuning")
    parser.add_argument("--n-iter", type=int, default=n_iter_default, help="RandomizedSearchCV iterations")
    parser.add_argument("--inner-splits", type=int, default=inner_splits_default, help="Inner CV folds for tuning")
    parser.add_argument("--outer-splits", type=int, default=outer_splits_default, help="Outer CV folds for evaluation")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed")
    parser.add_argument("--results-file", default="classification_results.txt", help="Results output file")
    return parser.parse_args()


def build_pipeline(model_choice: str, estimator) -> Pipeline:
    """Build a sklearn pipeline for the chosen model.

    Non-tree models require standard scaling, while tree-based models do not.
    """
    if model_choice in tree_models:
        return Pipeline([("clf", estimator)])
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def main():
    """Run the binary classification pipeline end to end."""
    args = parse_args()
    set_all_seeds(args.seed)

    hc_folder = Path(args.hc_folder)
    rtt_folder = Path(args.rtt_folder)
    model_choice = args.model_choice
    dev_frac = args.dev_frac
    n_iter = args.n_iter
    inner_splits = args.inner_splits
    outer_splits = args.outer_splits
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

    hc_features = extract_data(hc_folder)
    rtt_features = extract_data(rtt_folder)

    x = np.vstack((hc_features, rtt_features))
    y = np.hstack((np.zeros(len(hc_features)), np.ones(len(rtt_features))))
    groups = np.asarray(hc_groups + rtt_groups)

    log_elapsed("Step 1 — data loading", _t, _timings)
    print(f"Segments: {x.shape[0]}  |  features: {x.shape[1]}  |  "
          f"class dist: {Counter(y.astype(int))}  (HC:RTT = "
          f"{Counter(y.astype(int))[0]}:{Counter(y.astype(int))[1]})")
    print(f"Subjects: {len(np.unique(groups))}  "
          f"(HC: {len(np.unique(groups[y==0]))}, RTT: {len(np.unique(groups[y==1]))})")

    # ── Step 2 — Subject-level DEV / EVAL split (by subject IDs) ───
    # Start timer for this step (subject-level operations)
    _t = time.time()

    # Create subject-level labels (unique subject IDs and one label per subject)
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

    clf, param_grid = get_model_and_params(model_choice)
    pipe = build_pipeline(model_choice, clf)

    tuner = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=args.seed),
        n_jobs=1,
        refit=True,
        random_state=args.seed,
    )
    tuner.fit(x_dev, y_dev, groups=g_dev)

    best_params = tuner.best_params_
    log_elapsed("Step 3 — hyperparameter tuning", _t, _timings)
    print(f"Best params    : {best_params}")
    print(f"Best DEV score : {tuner.best_score_:.4f}")

    # ── Step 4 — Evaluation on EVAL set (outer CV by subject) ─────────────────────

    _t = time.time()
    _fold_times = []

    seg_acc, seg_prec, seg_sens, seg_spec, seg_bacc, seg_f1, seg_auc = [], [], [], [], [], [], []

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

        # fresh model per fold (recreate estimator so folds are independent)
        clf, _ = get_model_and_params(model_choice)
        model = build_pipeline(model_choice, clf)
        model.set_params(**best_params)
        model.fit(x_tr, y_tr)

        # Save fitted model and indices for later use
        fold_models.append(model)
        fold_train_indices.append(tr_idx)
        fold_test_indices.append(te_idx)

        # Segment-level metrics
        y_pred = model.predict(x_te)
        y_proba = get_proba(model, x_te)
        seg_m = compute_metrics(y_te, y_pred, y_proba)

        _fold_elapsed = time.time() - _t_fold
        _fold_times.append(_fold_elapsed)
        fm, fs = divmod(_fold_elapsed, 60)

        print(f"Fold {fold_idx}  [{int(fm)}m {fs:.1f}s]")
        print_metrics("  Segment", seg_m)
        print()

        seg_acc.append(seg_m["acc"])
        seg_bacc.append(seg_m["bacc"])
        seg_auc.append(seg_m["auc"])
        seg_prec.append(seg_m["prec"])
        seg_sens.append(seg_m["sens"])
        seg_spec.append(seg_m["spec"])
        seg_f1.append(seg_m["f1"])

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
    print(f"    Accuracy     : {np.mean(seg_acc):.3f} ± {np.std(seg_acc):.3f}")
    print(f"    Precision     : {np.mean(seg_prec):.3f} ± {np.std(seg_prec):.3f}")
    print(f"    Sensitivity     : {np.mean(seg_sens):.3f} ± {np.std(seg_sens):.3f}")
    print(f"    Spec.     : {np.mean(seg_spec):.3f} ± {np.std(seg_spec):.3f}")
    print(f"    F1     : {np.mean(seg_f1):.3f} ± {np.std(seg_f1):.3f}")
    print(f"    Balanced Acc : {np.mean(seg_bacc):.3f} ± {np.std(seg_bacc):.3f}")
    if not np.all(np.isnan(seg_auc)):
        print(f"    AUC          : {np.nanmean(seg_auc):.3f} ± {np.nanstd(seg_auc):.3f}")

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
            f.write(f"  Accuracy    : {seg_acc[fold_idx]:.3f}\n")
            f.write(f"  Precision   : {seg_prec[fold_idx]:.3f}\n")
            f.write(f"  Sensitivity : {seg_sens[fold_idx]:.3f}\n")
            f.write(f"  Specificity : {seg_spec[fold_idx]:.3f}\n")
            f.write(f"  F1 Score    : {seg_f1[fold_idx]:.3f}\n")
            f.write(f"  Balanced Acc: {seg_bacc[fold_idx]:.3f}\n")
            f.write(f"  AUC         : {seg_auc[fold_idx]:.3f}\n\n")
            
        f.write("=== Final Evaluation Summary ===\n")
        f.write(f"  Accuracy    : {np.mean(seg_acc):.3f} ± {np.std(seg_acc):.3f}\n")
        # Write a concise, human-readable report. Appending mode is used so
        # multiple runs accumulate in the same file for later inspection.
        f.write(f"  Precision   : {np.mean(seg_prec):.3f} ± {np.std(seg_prec):.3f}\n")
        f.write(f"  Sensitivity : {np.mean(seg_sens):.3f} ± {np.std(seg_sens):.3f}\n")
        f.write(f"  Specificity : {np.mean(seg_spec):.3f} ± {np.std(seg_spec):.3f}\n")
        f.write(f"  F1 Score    : {np.mean(seg_f1):.3f} ± {np.std(seg_f1):.3f}\n")
        f.write(f"  Balanced Acc: {np.mean(seg_bacc):.3f} ± {np.std(seg_bacc):.3f}\n")
        if not np.all(np.isnan(seg_auc)):
            f.write(f"  AUC         : {np.nanmean(seg_auc):.3f} ± {np.nanstd(seg_auc):.3f}\n")
        else:
            f.write("  AUC         : N/A\n")
            
        f.write("\n=== Timing Summary ===\n")
        total = time.time() - pipeline_start
        for stage, secs in _timings.items():
            m, s = divmod(secs, 60)
            f.write(f"  {stage:<42} {int(m)}m {s:.1f}s  ({100 * secs / total:.1f}%)\n")
        f.write(f"  {'Total':<42} {int(t_m)}m {t_s:.1f}s\n")
        if _fold_times:
            f.write(f"  Avg per eval fold: {int(fm)}m {fs:.1f}s\n")
    print(f"\nResults successfully saved to {results_path}")

if __name__ == "__main__":
    main()
