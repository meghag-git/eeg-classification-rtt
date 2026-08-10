import argparse
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from utils import (
    assign_groups,
    extract_data,
    set_all_seeds,
    subject_level_labels,
    log_elapsed,
    run_hyperparameter_tuning,
    run_outer_cv_evaluation,
    compute_shap_explanations,
    generate_and_save_plots,
)

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
# ── Model & cross-validation ──────────────────────────────────────────────────
# Options: SVM_linear | SVM_rbf | kNN | Logistic_regression | Decision_tree | Random_forest | Gradient_boosting | XGBoost
model_choice_default = "Decision_tree"
dev_frac_default = 0.3   # fraction of subjects reserved for hyperparameter tuning
n_iter_default = 20      # RandomizedSearchCV iterations
inner_splits_default = 5  # inner CV folds (tuning)
outer_splits_default = 5  # outer CV folds (evaluation)

seed = 42

# ── SHAP ──────────────────────────────────────────────────
# 20 channel names 
CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", 
    "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"    
]
 
N_CHANS = 20
SFREQ   = 256    # Hz — used only for the frequency axis label

BANDS = [
    ("Delta",  0,  4.0),
    ("Theta",  4.0,  8.0),
    ("Alpha",  8.0, 13.0),
    ("Beta",  13.0, 30.0),
    ("Gamma", 30.0, 45.0),
]
BAND_COLORS = ["#378ADD", "#7F77DD", "#1D9E75", "#EF9F27", "#D85A30"]
FREQ_PLOT_MAX = 45   # clip display above gamma
OUTPUT_DIR = Path(__file__).parent.resolve() / "feature_importance_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    parser.add_argument("--hc-folder", default=None, help="Path to HC .mat files")
    parser.add_argument("--rtt-folder", default=None, help="Path to RTT .mat files")
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


def main():
    """Run the binary classification pipeline end to end."""
    global model_choice_default
    args = parse_args()
    set_all_seeds(args.seed)

    hc_folder = Path(args.hc_folder)
    rtt_folder = Path(args.rtt_folder)
    model_choice = args.model_choice
    model_choice_default = model_choice
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
    # subject ID per window (HC first; RTT groups are offset to follow HC)
    rtt_groups = [x + 1 + max(hc_groups) for x in rtt_groups]

    hc_features = extract_data(hc_folder)
    rtt_features = extract_data(rtt_folder)

    x = np.vstack((hc_features, rtt_features))
    y = np.hstack((np.zeros(len(hc_features)), np.ones(len(rtt_features))))
    groups = np.asarray(hc_groups + rtt_groups)

    log_elapsed("Step 1 — data loading", _t, _timings)
    print(f"Windows: {x.shape[0]}  |  features: {x.shape[1]}  |  "
          f"class dist: {Counter(y.astype(int))}  (HC:RTT = "
          f"{Counter(y.astype(int))[0]}:{Counter(y.astype(int))[1]})")
    print(f"Windows: {len(np.unique(groups))}  "
          f"(HC: {len(np.unique(groups[y==0]))}, RTT: {len(np.unique(groups[y==1]))})")

    # ── Step 2 — Subject-level DEV / EVAL split (by subject IDs) ───
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
          f"{len(y_dev)} windows, class dist: {Counter(y_dev.astype(int))}")
    print(f"EVAL — {len(np.unique(g_eval))} subjects, "
          f"{len(y_eval)} windows, class dist: {Counter(y_eval.astype(int))}")

    # ── Step 3 — Hyperparameter tuning on DEV set ─────────────────────────────────
    _t = time.time()
    best_params, best_score = run_hyperparameter_tuning(
        x_dev, y_dev, g_dev, model_choice, n_iter, inner_splits, args.seed
    )
    log_elapsed("Step 3 — hyperparameter tuning", _t, _timings)
    print(f"Best params    : {best_params}")
    print(f"Best DEV score : {best_score:.4f}")

    # ── Step 4 — Evaluation on EVAL set (outer CV by subject) ─────────────────────
    _t = time.time()
    (
        fold_models,
        fold_train_indices,
        fold_test_indices,
        win_acc,
        win_prec,
        win_sens,
        win_spec,
        win_bacc,
        win_f1,
        win_auc,
    ) = run_outer_cv_evaluation(
        x_eval,
        y_eval,
        g_eval,
        model_choice,
        best_params,
        outer_splits,
        args.seed,
        results_path,
        hc_folder,
        rtt_folder,
    )
    log_elapsed("Step 4 — classification folds", _t, _timings)

    # ── Step 5 — SHAP Explanation ─────────────────────
    _t = time.time()
    N_FREQS = x_eval.shape[1] // N_CHANS
    FREQS = np.linspace(0, SFREQ / 2, N_FREQS)

    (
        all_shap,
        fold_mean_abs,
        mean_abs_across_folds,
        new_index,
        global_importance_matrix,
        channel_importance_mean,
    ) = compute_shap_explanations(
        fold_models,
        fold_train_indices,
        fold_test_indices,
        x_eval,
        y_eval,
        model_choice,
        N_CHANS,
        N_FREQS,
    )
    log_elapsed("Step 5 — SHAP feature importance", _t, _timings)

    # ── Step 6 — Generate and Save Plots ─────────────────────
    generate_and_save_plots(
        all_shap=all_shap,
        fold_mean_abs=fold_mean_abs,
        mean_abs_across_folds=mean_abs_across_folds,
        x_eval=x_eval,
        y_eval=y_eval,
        fold_test_indices=fold_test_indices,
        FREQS=FREQS,
        N_CHANS=N_CHANS,
        N_FREQS=N_FREQS,
        SFREQ=SFREQ,
        CH_NAMES=CH_NAMES,
        BANDS=BANDS,
        OUTPUT_DIR=OUTPUT_DIR,
        model_choice=model_choice,
        channel_importance_mean=channel_importance_mean,
        new_index=new_index,
        FREQ_PLOT_MAX=FREQ_PLOT_MAX,
    )

if __name__ == "__main__":
    main()
