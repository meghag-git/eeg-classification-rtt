# Machine Learning- Binary Classification

This repository contains a binary classification pipeline for PSD feature data from HC and RTT subjects.

## Files

- `binary_classification.py` — main script for training, tuning, and evaluating models
- `utils.py` — reusable helper functions for data loading, grouping, metrics, and model configuration
- `requirements.txt` — package dependencies

## Requirements

Create and activate your environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script from the repository root:

```bash
python binary_classification.py
```

### Common options

```bash
python binary_classification.py \
  --hc-folder /path/to/HC \
  --rtt-folder /path/to/RTT \
  --model-choice SVM_rbf \
  --dev-frac 0.3 \
  --n-iter 20 \
  --inner-splits 5 \
  --outer-splits 5 \
  --results-file classification_results.txt
```

## Expected inputs

- `--hc-folder`: folder containing HC `.mat` files
- `--rtt-folder`: folder containing RTT `.mat` files

## Output

- Default result file: `classification_results.txt`
- The script appends each run with a time summary, fold metrics, and final evaluation scores.

## Notes

- The script performs subject-level DEV/EVAL splitting and nested cross-validation.
- If your filename format differs, update `Assign_Groups()` in `utils.py`.

