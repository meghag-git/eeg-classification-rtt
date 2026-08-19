# eeg-classification-ml-dl
A collection of reproducible EEG classification pipelines implementing classical machine learning on power spectral density (PSD) features and deep learning on preprocessed EEG time-series data for neurological diagnosis and EEG research. 

### EEG Preprocessing

EEG preprocessing was performed in **MATLAB R2024b** using custom scripts and **EEGLAB (v2025.0.0)**, with the same pipeline applied to both Rett syndrome (RTT) and healthy control (HC) recordings.

The preprocessing pipeline consisted of:

* Importing EEG recordings in `.edf` format.
* Retaining the **20 EEG channels** common to all RTT and HC recordings.
* Applying a **1–45 Hz zero-phase FIR band-pass filter** using EEGLAB `pop_eegfiltnew`.
* Re-referencing signals to the **common average reference**.
* Visually inspecting recordings for signal quality and unusable channels.
* Extracting resting-state **eyes-closed (EC)** segments using the `"Eyes Closed"` and `"Eyes Opened"` event annotations in the original recordings.
* Keeping multiple EC segments separate to avoid artificial discontinuities caused by concatenation.
* Dividing EC segments into **non-overlapping 4-second epochs** for subsequent machine learning and deep learning analyses.

### Feature Extraction for Machine Learning

For the classical machine learning pipeline, **power spectral density** features were extracted from each 4-second eyes-closed EEG epoch using **Welch’s method**.

* PSD was computed separately for each of the **20 EEG channels**.
* Each epoch was analysed using **512-sample (2-second) windows** with **50% overlap**.
* A **Hamming window** was applied to reduce spectral leakage.
* Spectral estimates were computed using a **1024-point FFT** and averaged across windows.
* The full PSD representation was retained rather than aggregating power into predefined frequency bands.
* Each channel produced **513 non-redundant frequency bins**.
* Features from all channels were concatenated to produce a **10,260-dimensional feature vector (20 channels × 513 frequency bins)** for each EEG epoch.

These PSD feature vectors were subsequently used as input to the classical machine learning classification models.

This repository contains reproducible pipelines for EEG-based classification, implementing classical machine learning on PSD features and deep learning on preprocessed EEG time-series data for neurological diagnosis and EEG research. It provides two top-level folders — machine_learning/ (scikit-learn, XGBoost, classical pipelines) and deep_learning/ (PyTorch/TensorFlow models and training scripts) — plus readme, utilities, and evaluation scripts.
