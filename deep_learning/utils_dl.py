import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler

import numpy as np
import mne
from scipy.io import loadmat
from scipy.stats import randint

# Model imports
import tensorflow as tf
from tensorflow.keras import backend as K

# Global configuration defaults (can be overridden dynamically by imports/callers)
MAX_EPOCHS = 50
BATCH_SIZE = 8
NB_CLASSES = 2
N_CHANS = 20
N_SAMPLES = 1024

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, BatchNormalization,
    Conv2D, Dense, DepthwiseConv2D, Dropout, Flatten,
    GaussianNoise, Input, MaxPooling2D, SeparableConv2D, SpatialDropout2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping as SkorchEarlyStopping
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.utils.data import WeightedRandomSampler

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

def set_all_seeds(seed=42):
    """Set RNG seeds for random, numpy, torch, and tensorflow to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # TensorFlow seeding
    tf.random.set_seed(seed)

def get_safe_device() -> str:
    """Return the best available device, falling back to CPU if CUDA is incompatible."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except Exception:
            print("WARNING: CUDA incompatible with this PyTorch build — using CPU.")
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def log_elapsed(label: str, start: float, timings: dict) -> None:
    """Record elapsed time for `label` into `timings` and print a summary."""
    secs = time.time() - start
    timings[label] = secs
    m, s = divmod(secs, 60)
    print(f"  ⏱  {label}: {int(m)}m {s:.1f}s")

def scale_per_channel(x_train: np.ndarray, x_test: np.ndarray = None):
    """Standardise per channel using training statistics only."""
    n_chans = x_train.shape[1]
    scaler  = StandardScaler()
    x_train_scaled = scaler.fit_transform(
        x_train.reshape(-1, n_chans)).reshape(x_train.shape)
    if x_test is not None:
        x_test_scaled = scaler.transform(
            x_test.reshape(-1, n_chans)).reshape(x_test.shape)
        return x_train_scaled, x_test_scaled
    return x_train_scaled

def reshape_for_model(X: np.ndarray, model_choice: str) -> np.ndarray:
    """Reshape X to the format expected by the chosen model backend."""
    n, chans, samples = X.shape
    if model_choice == "EEGConformer":
        return X.reshape(n, 1, chans, samples).astype(np.float32)
    return X.reshape(n, chans, samples, 1)

def subject_level_labels(y: np.ndarray, groups: np.ndarray):
    """Return (unique_subject_ids, one_label_per_subject).

    Raises ValueError if any subject has inconsistent labels.
    """
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
    """Compute all evaluation metrics and return as a dict."""
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


def print_metrics(label: str, m: dict) -> None:
    print(f"  {label} | Acc: {m['acc']:.3f}  Prec: {m['prec']:.3f}  "
          f"Sensitivity: {m['sens']:.3f} Specif.: {m['spec']:.3f}  F1: {m['f1']:.3f}  "
          f"AUC: {m['auc']:.3f}  BAcc: {m['bacc']:.3f}")

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

    Returns a label index for every matched file (.set or .mat) and the sorted list of unique subject IDs.
    """
    folder_path = Path(folder_path)
    # Auto-detect suffix based on what files exist in the folder (defaulting to .set if present, else .mat)
    has_set = any(path.suffix.lower() == ".set" for path in folder_path.iterdir() if path.is_file())
    suffix = ".set" if has_set else ".mat"

    file_list = sorted(
        [path.name for path in folder_path.iterdir() if path.suffix.lower() == suffix],
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

def extract_data_SetFormat(folder_path, Group):
    features = []
    New_Group = []
    # Make sure we use the same sort_key and extension check to align indices
    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.set')],
        key=sort_key
    )
    num_files = len(file_list)
    for i in range(num_files):
        raw = mne.io.read_raw_eeglab(os.path.join(folder_path,file_list[i]), preload=True)
        # Get data as NumPy array
        data, _ = raw.get_data(return_times=True)
        sfreq = raw.info['sfreq']
        win_length = int(4 * sfreq)
        
        group_id = Group[i]
        
        # Segment into windows
        for start in range(0, data.shape[1] - win_length, win_length):
            end = start + win_length
            features.append(data[:, start:end])
            New_Group.append(group_id)  # assign same group ID to all windows from this recording
            
    features_array = np.array(features)
    
    return features_array, New_Group 

def get_model_and_params(model_choice: str, early_stop_monitor: str = None) -> Tuple[object, dict]:
    """Return a base estimator and its hyperparameter grid for a model choice."""
    if model_choice == "ShallowConvNet":
        monitor = early_stop_monitor if early_stop_monitor is not None else "loss"
        model = build_keras_wrapper(build_shallow_convnet, early_stop_monitor=monitor, patience=10)
        model_params = {
            "model__filters_1":       [20, 40, 60, 80],
            "model__filters_2":       [20, 40, 60, 80],
            "model__temporal_kernel": [10, 15, 25, 40],
            "model__bn_epsilon":      [1e-3, 1e-4, 1e-5],
            "model__bn_momentum":     [0.8, 0.9, 0.99],
            "model__pool_size":       [35, 50, 75],
            "model__pool_stride":     [5, 10, 15],
            "model__dropout":         [0.2, 0.3, 0.4, 0.5, 0.6],
            "model__dense_max_norm":  [0.25, 0.5, 1.0],
            "model__learning_rate":   [1e-2, 1e-3, 1e-4],
        }
    elif model_choice == "DeepConvNet":
        monitor = early_stop_monitor if early_stop_monitor is not None else "loss"
        model = build_keras_wrapper(build_deep_convnet, early_stop_monitor=monitor, patience=10)
        model_params = {
            "model__filters_1":       [25, 30, 40],
            "model__filters_2":       [50, 60],
            "model__filters_3":       [100, 120],
            "model__filters_4":       [200, 250],
            "model__temporal_kernel": [10, 15, 20, 25, 30],
            "model__bn_epsilon":      [1e-4, 1e-5],
            "model__bn_momentum":     [0.8, 0.9, 0.99],
            "model__pool_size":       [2, 3],
            "model__pool_stride":     [2, 3],
            "model__dropout":         [0.4, 0.5, 0.6, 0.7],
            "model__dense_max_norm":  [0.5, 1.0],
            "model__learning_rate":   [1e-2, 1e-3, 1e-4],
            "model__conv_max_norm":   [0.5, 1.0, 2.0],
            "model__l2_reg":          [0.01, 0.001, 0.0001],
        }
    elif model_choice == "EEGNet":
        monitor = early_stop_monitor if early_stop_monitor is not None else "loss"
        model = build_keras_wrapper(build_eegnet, early_stop_monitor=monitor, patience=10)
        model_params = {
            "model__dropoutRate":   [0.25, 0.3, 0.4, 0.5],
            "model__kernLength":    [16, 32, 64],
            "model__F1":            [4, 8, 16],
            "model__D":             [2, 4],
            "model__F2":            [8, 16, 32],
            "model__norm_rate":     [0.25, 0.5, 1.0],
            "model__dropoutType":   ["Dropout", "SpatialDropout2D"],
            "model__learning_rate": [1e-2, 1e-3, 1e-4],
        }
    elif model_choice == "EEGConformer":
        device     = get_safe_device()
        chans      = 20
        monitor = early_stop_monitor if early_stop_monitor is not None else "train_loss"
        model = build_skorch_conformer(chans, device, early_stop_monitor=monitor, patience=10)
        model_params = {
            "module__temporal_filters": [20, 40, 60, 80],
            "module__temporal_kernel": [10, 15, 20, 25, 40],
            "module__pool_kernel": [35, 50, 75],
            "module__pool_stride":     [5, 10, 15],
            "module__emb_size":  [20, 30, 40, 50, 60],
            "module__depth":     [4, 6, 8],
            "module__drop_prob": [0.2, 0.3, 0.4, 0.5],
            "lr":                [1e-2, 1e-3, 1e-4],
        }

    return model, model_params
        
def build_keras_wrapper(model_builder, early_stop_monitor, patience):
    """Instantiate a scikeras KerasClassifier with EarlyStopping."""
    return KerasClassifier(
        model=model_builder,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[EarlyStopping(monitor=early_stop_monitor,
                                  patience=patience, restore_best_weights=True)],
    )

def build_skorch_conformer(chans, device, early_stop_monitor, patience):
    """Instantiate a skorch NeuralNetClassifier wrapping the Conformer."""
    return NeuralNetClassifier(
        module=Conformer,
        module__n_chans=chans,
        module__n_classes=NB_CLASSES,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        device=device,
        callbacks=[SkorchEarlyStopping(monitor=early_stop_monitor,
                                        patience=patience, lower_is_better=True)],
    )

def square(x):
    return K.square(x)

def safe_log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=1e4))

def build_shallow_convnet(filters_1=40, filters_2=40, temporal_kernel=25,
    bn_epsilon=1e-5, bn_momentum=0.1,
    pool_size=75, pool_stride=15,
    dropout=0.5, dense_max_norm=0.5, learning_rate=1e-3,):
    
    inputs = Input((N_CHANS, N_SAMPLES, 1))
    x = Conv2D(filters_1, (1, temporal_kernel),
               kernel_constraint=max_norm(2., axis=(0, 1, 2)))(inputs)
    x = Conv2D(filters_2, (N_CHANS, 1), use_bias=False,
               kernel_constraint=max_norm(2., axis=(0, 1, 2)))(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation(square)(x)
    x = AveragePooling2D((1, pool_size), strides=(1, pool_stride))(x)
    x = Activation(safe_log)(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(NB_CLASSES, kernel_constraint=max_norm(dense_max_norm))(x)
    outputs = Activation("softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    return model

def build_deep_convnet(
    filters_1=25, filters_2=50, filters_3=100, filters_4=200,
    temporal_kernel=5, pool_size=2, pool_stride=2,
    bn_epsilon=1e-5, bn_momentum=0.9, dropout=0.5,
    dense_max_norm=0.5, conv_max_norm=2.0, l2_reg=0.01, learning_rate=1e-3,
):
    def conv_block(x, filters, kernel_size):
        x = Conv2D(filters, kernel_size,
                   kernel_constraint=max_norm(conv_max_norm, axis=(0, 1, 2)),
                   kernel_regularizer=l2(l2_reg))(x)
        return BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)

    inputs = Input((N_CHANS, N_SAMPLES, 1))
    x = GaussianNoise(0.01)(inputs)
    x = conv_block(x, filters_1, (1, temporal_kernel))
    x = conv_block(x, filters_1, (N_CHANS, 1))
    x = Activation("elu")(x)
    x = MaxPooling2D((1, pool_size), strides=(1, pool_stride))(x)
    x = Dropout(dropout)(x)
    for filters in (filters_2, filters_3):
        x = conv_block(x, filters, (1, temporal_kernel))
        x = Activation("elu")(x)
        x = MaxPooling2D((1, pool_size), strides=(1, pool_stride))(x)
        x = Dropout(dropout)(x)
    x = conv_block(x, filters_4, (1, temporal_kernel))
    x = Activation(square)(x)
    x = AveragePooling2D((1, pool_size), strides=(1, pool_stride))(x)
    x = Activation(safe_log)(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(NB_CLASSES, kernel_constraint=max_norm(dense_max_norm),
              kernel_regularizer=l2(l2_reg))(x)
    outputs = Activation("softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    return model

def build_eegnet(
    dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
    norm_rate=0.25, dropoutType="Dropout", learning_rate=1e-3,
):
    DropoutLayer = SpatialDropout2D if dropoutType == "SpatialDropout2D" else Dropout
    inputs = Input((N_CHANS, N_SAMPLES, 1))
    x = Conv2D(F1, (1, kernLength), padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((N_CHANS, 1), use_bias=False, depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 4))(x)
    x = DropoutLayer(dropoutRate)(x)
    x = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 8))(x)
    x = DropoutLayer(dropoutRate)(x)
    x = Flatten(name="flatten")(x)
    x = Dense(NB_CLASSES, kernel_constraint=max_norm(norm_rate), name="dense")(x)
    outputs = Activation("softmax", name="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    return model

class PatchEmbedding(nn.Module):
    """Shallow CNN front-end that projects raw EEG into patch embeddings."""
    def __init__(
        self,
        n_chans: int,
        emb_size: int = 40,
        drop_p: float = 0.5,
        temporal_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 75,
        pool_stride: int = 15,
    ):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_kernel)),
            nn.Conv2d(temporal_filters, temporal_filters, kernel_size=(n_chans, 1)),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride)),
            nn.Dropout(drop_p),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(temporal_filters, emb_size, kernel_size=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.shallownet(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads  = num_heads
        self.emb_size   = emb_size
        self.queries    = nn.Linear(emb_size, emb_size)
        self.keys       = nn.Linear(emb_size, emb_size)
        self.values     = nn.Linear(emb_size, emb_size)
        self.att_drop   = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", q, k) / (self.emb_size ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)
        att = self.att_drop(F.softmax(energy, dim=-1))
        out = rearrange(torch.einsum("bhqk, bhkd -> bhqd", att, v), "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x, **kwargs): return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(), nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, att_drop=0.5, ff_expansion=4, ff_drop=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, att_drop),
                nn.Dropout(att_drop),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, ff_expansion, ff_drop),
                nn.Dropout(ff_drop),
            )),
        )


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )
    def forward(self, x): return self.head(x)


class Conformer(nn.Module):
    """EEG Conformer: shallow CNN patch embedding + Transformer encoder."""
    def __init__(
        self,
        n_chans,
        n_classes,
        emb_size=40,
        depth=6,
        drop_prob=0.5,
        temporal_filters=40,
        temporal_kernel=25,
        pool_kernel=75,
        pool_stride=15,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            n_chans=n_chans,
            emb_size=emb_size,
            drop_p=drop_prob,
            temporal_filters=temporal_filters,
            temporal_kernel=temporal_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )

        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(emb_size) for _ in range(depth)]
        )

        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x