# =====================
# Fall Detection - Training Script
# =====================
# Loads processed head-sensor trial data, trains 3 models, and saves
# metrics.txt + PNG plots for the CML feedback loop.

import os
import glob
import json
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# For model export (optional – used only in Step 10; each call is already
# wrapped in try/except so missing packages are handled gracefully)
try:
    from micromlgen import port as mlport
except ImportError:
    mlport = None

try:
    import m2cgen as m2c
except ImportError:
    m2c = None

try:
    import emlearn
except ImportError:
    emlearn = None


warnings.filterwarnings("ignore")
print(f"TensorFlow {tf.__version__} | NumPy {np.__version__}")

# ── Paths ────────────────────────────────────────────────────────────────────
# The notebook used 'combined_head_only.csv', but this script is set up for CML
# and uses separate trial files, which is better for reproducibility.
DATA_DIR = "data/processed/separate_trials"
OUTPUTS_DIR = "."          # write outputs to the repo root for CML to pick up

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 100          # Hz
WINDOW_SIZE = 100          # 1 s at 100 Hz
STEP_SIZE   = 50           # 50 % overlap
THRESHOLD   = 0.35         # classification threshold (favor recall / safety)
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FALL_DURATION_SECS = 15   # cap unclosed fall segments at this many seconds
FALL_START_MARKERS = {'startOfFall', 'FallStart'}
FALL_END_MARKERS   = {'endOfFall',   'FallEnd'}

SENSOR_COLS = [
    "acc_x", "acc_y", "acc_z",
    "rot_x", "rot_y", "rot_z",
    "pitch", "roll", "yaw", "acc_mag",
]
N_FEATURES = len(SENSOR_COLS)

# ── Step 1: Load processed trial CSVs ────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "trial_*", "*_head_only.csv")))
if not csv_files:
    raise FileNotFoundError(f"No head_only CSV files found under {DATA_DIR}")

all_data = []
for fp in csv_files:
    df = pd.read_csv(fp, low_memory=False)
    # Use trial name from folder instead of filename for clarity
    df["trial_id"] = os.path.basename(os.path.dirname(fp))
    if 'MarkerNames' not in df.columns:
        print(f"  ⚠️  WARNING: {os.path.basename(fp)} has no MarkerNames column — all rows treated as non-fall")
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(csv_files)} trial files  →  {len(data):,} total rows")
print(f"  Detected {data['trial_id'].nunique()} trials")


# ── Step 2: Rename & clean columns ───────────────────────────────────────────
print("\nSTEP 2: Cleaning data and deriving features")
col_map = {
    "Time,s":                  "time",
    "Head course,deg":         "yaw",
    "Head pitch,deg":          "pitch",
    "Head roll,deg":           "roll",
    "Head Accel Sensor X,mG":  "acc_x",
    "Head Accel Sensor Y,mG":  "acc_y",
    "Head Accel Sensor Z,mG":  "acc_z",
    "Head Rot X,":             "rot_x",
    "Head Rot Y,":             "rot_y",
    "Head Rot Z,":             "rot_z",
}
data.rename(columns=col_map, inplace=True)

raw_cols = ['time', 'yaw', 'pitch', 'roll',
            'acc_x', 'acc_y', 'acc_z',
            'rot_x', 'rot_y', 'rot_z']
for c in raw_cols:
    data[c] = pd.to_numeric(data[c], errors='coerce')

data.dropna(subset=raw_cols, inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"  After NaN drop: {data.shape[0]:,} rows")

# ---- derived feature ----
data['acc_mag'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)


# ── Step 3: Label falls using MarkerNames ────────────────────────────────────
print("\nSTEP 3: Labelling falls from MarkerNames")
# Ensure the column exists even if some trials lacked it (filled with NaN by concat)
if 'MarkerNames' not in data.columns:
    data['MarkerNames'] = ''
data['MarkerNames'] = data['MarkerNames'].fillna('').astype(str).str.strip()
data['label'] = 0

MAX_FALL_SAMPLES = int(MAX_FALL_DURATION_SECS * SAMPLE_RATE)

def _label_trial(names):
    """State-machine labeler that handles marker aliases, double starts, and unclosed falls.

    - Recognises both 'startOfFall'/'FallStart' and 'endOfFall'/'FallEnd'.
    - When two consecutive start markers appear without an intervening end (e.g.
      trial_01 at t=178 s), the first segment is capped at MAX_FALL_SAMPLES and a
      warning is emitted instead of mislabelling the rest of the trial as a fall.
    - Any fall still open at the end of the trial is likewise capped and flagged.
    """
    labels = np.zeros(len(names), dtype=np.int8)
    in_fall = False
    fall_start_i = None
    msgs = []

    for i, name in enumerate(names):
        if name in FALL_START_MARKERS:
            if in_fall:
                cap = min(fall_start_i + MAX_FALL_SAMPLES, i)
                labels[fall_start_i:cap] = 1
                msgs.append(
                    f"double fall-start at row {i} (prev at {fall_start_i}); "
                    f"capped previous segment at {cap - fall_start_i} samples"
                )
            in_fall = True
            fall_start_i = i
        elif name in FALL_END_MARKERS:
            if in_fall:
                labels[fall_start_i : i + 1] = 1
                in_fall = False
                fall_start_i = None
            # else: spurious end marker — ignore

    # Unclosed fall at end of trial
    if in_fall and fall_start_i is not None:
        cap = min(fall_start_i + MAX_FALL_SAMPLES, len(names))
        labels[fall_start_i:cap] = 1
        msgs.append(
            f"unclosed fall at row {fall_start_i}; "
            f"capped at {cap - fall_start_i} samples "
            f"({(cap - fall_start_i) / SAMPLE_RATE:.1f} s)"
        )

    return labels, msgs

for tid, g in data.groupby('trial_id', sort=False):
    idx = g.index.to_numpy()
    names = g['MarkerNames'].to_numpy()
    labels, msgs = _label_trial(names)
    for m in msgs:
        print(f"  [{tid}] {m}")
    data.loc[idx, 'label'] = labels

fall_n   = int(data["label"].sum())
normal_n = len(data) - fall_n
print(f"  Fall samples  : {fall_n:>8,} ({fall_n / len(data) * 100:.1f}%)")
print(f"  Normal samples: {normal_n:>8,} ({normal_n / len(data) * 100:.1f}%)")
non_empty_markers = data[data['MarkerNames'] != '']['MarkerNames']
print(f"  Marker events : {non_empty_markers.value_counts().to_dict()}")


# ── Step 4: Sliding-window feature extraction ─────────────────────────────────
print("\nSTEP 4: Extracting sliding-window features")

STAT_NAMES = ['mean', 'std', 'min', 'max', 'range', 'peak', 'rms', 'delta']
feature_names = [f'{col}_{stat}' for col in SENSOR_COLS for stat in STAT_NAMES]
print(f"  Feature vector size : {len(feature_names)}")
print(f"  Sequence shape      : ({WINDOW_SIZE}, {N_FEATURES})")

def _stats(v):
    return [v.mean(), v.std(), v.min(), v.max(),
            v.max()-v.min(), np.abs(v).max(),
            np.sqrt((v**2).mean()), v[-1]-v[0]]

def extract_windows(trial_df):
    """Return feature matrix X_f (n,80), sequence tensor X_s (n,100,10), labels y (n,)."""
    arr   = trial_df[SENSOR_COLS].values.astype(np.float32)
    labs  = trial_df['label'].values
    n     = len(arr)

    X_f, X_s, y = [], [], []
    for s in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        e    = s + WINDOW_SIZE
        win  = arr[s:e]
        lab  = int(labs[s:e].mean() >= 0.5) # majority vote

        feats = []
        for ci in range(N_FEATURES):
            feats.extend(_stats(win[:, ci]))

        X_f.append(feats)
        X_s.append(win)
        y.append(lab)

    return np.array(X_f, np.float32), np.array(X_s, np.float32), np.array(y, np.int32)

X_feat_list, X_seq_list, y_list, group_list = [], [], [], []
for tid, g in data.groupby('trial_id', sort=False):
    g = g.reset_index(drop=True)
    if len(g) < WINDOW_SIZE:
        continue
    xf, xs, yy = extract_windows(g)
    X_feat_list.append(xf)
    X_seq_list.append(xs)
    y_list.append(yy)
    group_list.extend([tid] * len(yy))

X_feat  = np.concatenate(X_feat_list, axis=0)
X_seq   = np.concatenate(X_seq_list,  axis=0)
y       = np.concatenate(y_list,      axis=0)
groups  = np.array(group_list)

print(f"\n  Feature matrix : {X_feat.shape}")
print(f"  Sequence tensor: {X_seq.shape}")
print(f"  Labels         : {np.bincount(y)}  (normal={np.bincount(y)[0]}, fall={np.bincount(y)[1]})")
print(f"  Groups (trials): {len(np.unique(groups))} trials")


# ── Step 5: Prepare train / test splits ──────────────────────────────────────
print("\nSTEP 5: Preparing train/test splits (group-aware), scaling, and SMOTE")

# Group-aware split: whole trials go entirely to train or test.
# This prevents subject leakage — the same person never appears in both splits.
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_feat, y, groups=groups))

X_feat_tr, X_feat_te = X_feat[train_idx], X_feat[test_idx]
X_seq_tr,  X_seq_te  = X_seq[train_idx],  X_seq[test_idx]
y_tr,      y_te       = y[train_idx],      y[test_idx]

print(f"  Train windows : {len(y_tr):,}  (falls: {y_tr.sum()})  — {len(np.unique(groups[train_idx]))} trials: {np.unique(groups[train_idx])}")
print(f"  Test  windows : {len(y_te):,}  (falls: {y_te.sum()})  — {len(np.unique(groups[test_idx]))} trials: {np.unique(groups[test_idx])}")

if y_te.sum() == 0:
    raise RuntimeError("Test set has 0 fall windows — try a different RANDOM_STATE.")

# ---- feature scaler (for RF + NN) ----
feat_scaler = StandardScaler()
X_feat_tr_sc = feat_scaler.fit_transform(X_feat_tr)
X_feat_te_sc = feat_scaler.transform(X_feat_te)

# ---- sequence scaler (for LSTM) ----
n_tr, W, F = X_seq_tr.shape
seq_scaler  = StandardScaler()
X_seq_tr_sc = seq_scaler.fit_transform(X_seq_tr.reshape(-1, F)).reshape(n_tr, W, F)
X_seq_te_sc = seq_scaler.transform(X_seq_te.reshape(-1, F)).reshape(len(y_te), W, F)

# ---- SMOTE on 2-D feature data (RF + NN training) ----
# LSTM uses class_weight on the original imbalanced sequences instead.
print("  Applying SMOTE to feature training set ...")
sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=max(1, min(5, int(y_tr.sum()) - 1)))
X_feat_tr_sm, y_tr_sm = sm.fit_resample(X_feat_tr_sc, y_tr)
print(f"  After SMOTE : {np.bincount(y_tr_sm)}")

# ---- class weights for LSTM (computed from the original imbalanced distribution) ----
cw_arr  = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
class_w = {0: float(cw_arr[0]), 1: float(cw_arr[1])}
print(f"  Class weights (LSTM): {class_w}")


# ── Step 6: Train models ──────────────────────────────────────────────────────
print("\nSTEP 6: Training models")
results = {}


def _best_f1_threshold(y_true, proba):
    """Return the probability threshold that maximises F1 on the given split.

    Uses the precision-recall curve so every unique probability value is tried
    efficiently (O(n log n) via sklearn).  The threshold is informational only
    — it must NOT be used for threshold selection when the same data was used
    for training (use a held-out validation set in production).
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns n+1 values but only n thresholds;
    # the last precision/recall pair (index n) has no corresponding threshold.
    # Use a safe denominator to avoid FloatingPointError: np.where evaluates
    # both branches before masking, so we must prevent actual division by zero.
    denom = precision_vals + recall_vals
    safe_denom = np.where(denom == 0, 1.0, denom)
    f1_vals = np.where(denom == 0, 0.0, 2 * precision_vals * recall_vals / safe_denom)
    best_idx = int(np.argmax(f1_vals[:-1]))   # restrict to indices with a threshold
    assert best_idx < len(thresholds), "best_idx out of range for thresholds array"
    return float(thresholds[best_idx]), float(f1_vals[best_idx])


# --- Model 1: Random Forest ---
print("\n--- Random Forest ---")
rf = RandomForestClassifier(
    n_estimators   = 200,
    max_depth      = 15,
    min_samples_leaf = 3,
    max_features   = 'sqrt',
    class_weight   = 'balanced',
    random_state   = RANDOM_STATE,
    n_jobs         = -1,
)
rf.fit(X_feat_tr_sm, y_tr_sm)
rf_proba = rf.predict_proba(X_feat_te_sc)[:, 1]
rf_pred  = (rf_proba >= THRESHOLD).astype(int)
_rf_best_thr, _rf_best_f1 = _best_f1_threshold(y_te, rf_proba)
results["Random Forest"] = {
    "proba": rf_proba, "pred": rf_pred,
    "acc": accuracy_score(y_te, rf_pred), "f1": f1_score(y_te, rf_pred),
    "recall": recall_score(y_te, rf_pred), "precision": precision_score(y_te, rf_pred),
    "roc_auc": roc_auc_score(y_te, rf_proba),
    "best_f1_threshold": _rf_best_thr, "best_f1": _rf_best_f1,
}
print(f"  Precision {results['Random Forest']['precision']:.4f}, Recall {results['Random Forest']['recall']:.4f}, F1 {results['Random Forest']['f1']:.4f}, ROC-AUC {results['Random Forest']['roc_auc']:.4f}")
print(f"  (Best-F1 threshold on test set: {_rf_best_thr:.3f} → F1 {_rf_best_f1:.4f})")

# MCU-friendly Random Forest (small enough for AVR / STM32 Flash ≤ 256 KB)
print("\n--- MCU Random Forest (microcontroller export) ---")
rf_mcu = RandomForestClassifier(
    n_estimators   = 15,
    max_depth      = 8,
    min_samples_leaf = 5,
    max_features   = 'sqrt',
    class_weight   = 'balanced',
    random_state   = RANDOM_STATE,
    n_jobs         = -1,
)
rf_mcu.fit(X_feat_tr_sm, y_tr_sm)
rf_mcu_proba = rf_mcu.predict_proba(X_feat_te_sc)[:, 1]
rf_mcu_pred  = (rf_mcu_proba >= THRESHOLD).astype(int)
print(f"  Recall {recall_score(y_te, rf_mcu_pred):.4f},  F1 {f1_score(y_te, rf_mcu_pred):.4f}"
      f"  (15 trees × depth-8, suitable for ≤256 KB Flash)")


# --- Model 2: Simple Neural Network (MLP) ---
print("\n--- Simple Neural Network (MLP) ---")
nn_model = keras.Sequential([
    layers.Input(shape=(X_feat_tr_sm.shape[1],)),
    layers.Dense(128, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.30),
    layers.Dense(64, activation='relu'),  layers.BatchNormalization(), layers.Dropout(0.30),
    layers.Dense(32, activation='relu'),  layers.Dropout(0.20),
    layers.Dense(1, activation='sigmoid'),
], name='simple_nn')
nn_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

nn_callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
]
nn_history = nn_model.fit(
    X_feat_tr_sm, y_tr_sm,
    epochs=150, batch_size=64, validation_split=0.15,
    callbacks=nn_callbacks, verbose=0,
)
nn_proba = nn_model.predict(X_feat_te_sc, verbose=0).flatten()
nn_pred = (nn_proba >= THRESHOLD).astype(int)
_nn_best_thr, _nn_best_f1 = _best_f1_threshold(y_te, nn_proba)
results["Simple NN"] = {
    "proba": nn_proba, "pred": nn_pred, "history": nn_history,
    "acc": accuracy_score(y_te, nn_pred), "f1": f1_score(y_te, nn_pred),
    "recall": recall_score(y_te, nn_pred), "precision": precision_score(y_te, nn_pred),
    "roc_auc": roc_auc_score(y_te, nn_proba),
    "best_f1_threshold": _nn_best_thr, "best_f1": _nn_best_f1,
}
print(f"  Precision {results['Simple NN']['precision']:.4f}, Recall {results['Simple NN']['recall']:.4f}, F1 {results['Simple NN']['f1']:.4f}, ROC-AUC {results['Simple NN']['roc_auc']:.4f}")
print(f"  (Best-F1 threshold on test set: {_nn_best_thr:.3f} → F1 {_nn_best_f1:.4f})")


# --- Model 3: LSTM ---
print("\n--- LSTM ---")
lstm_model = keras.Sequential([
    layers.Input(shape=(WINDOW_SIZE, N_FEATURES)),
    layers.LSTM(64, return_sequences=True), layers.Dropout(0.30),
    layers.LSTM(32), layers.Dropout(0.30),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
], name='lstm_model')
lstm_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

lstm_callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
]
lstm_history = lstm_model.fit(
    X_seq_tr_sc, y_tr,
    epochs=100, batch_size=64, validation_split=0.15,
    class_weight=class_w, callbacks=lstm_callbacks, verbose=0,
)
lstm_proba = lstm_model.predict(X_seq_te_sc, verbose=0).flatten()
lstm_pred = (lstm_proba >= THRESHOLD).astype(int)
_lstm_best_thr, _lstm_best_f1 = _best_f1_threshold(y_te, lstm_proba)
results["LSTM"] = {
    "proba": lstm_proba, "pred": lstm_pred, "history": lstm_history,
    "acc": accuracy_score(y_te, lstm_pred), "f1": f1_score(y_te, lstm_pred),
    "recall": recall_score(y_te, lstm_pred), "precision": precision_score(y_te, lstm_pred),
    "roc_auc": roc_auc_score(y_te, lstm_proba),
    "best_f1_threshold": _lstm_best_thr, "best_f1": _lstm_best_f1,
}
print(f"  Precision {results['LSTM']['precision']:.4f}, Recall {results['LSTM']['recall']:.4f}, F1 {results['LSTM']['f1']:.4f}, ROC-AUC {results['LSTM']['roc_auc']:.4f}")
print(f"  (Best-F1 threshold on test set: {_lstm_best_thr:.3f} → F1 {_lstm_best_f1:.4f})")


# ── Step 7: Save metrics.txt ─────────────────────────────────────────────────
print("\nSTEP 7: Saving metrics and reports")
best_name = max(results, key=lambda k: results[k]["recall"])
best_pred = results[best_name]["pred"]

metrics_path = os.path.join(OUTPUTS_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Fall Detection — Training Results\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"{'Model':<20} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'Precision':>10} {'ROC-AUC':>8} {'BestF1Thr':>10}\n")
    f.write("-" * 80 + "\n")
    for name, r in results.items():
        # 'best_f1_threshold' is present for all main models; guard against
        # any future model added without calling _best_f1_threshold().
        best_thr_str = f"{r['best_f1_threshold']:.3f}" if r.get('best_f1_threshold') is not None else "N/A"
        f.write(
            f"{name:<20} {r['acc']:>10.4f} {r['f1']:>8.4f} "
            f"{r['recall']:>8.4f} {r['precision']:>10.4f} {r['roc_auc']:>8.4f} {best_thr_str:>10}\n"
        )
    f.write(f"\n  Note: BestF1Thr = threshold that maximises F1 on the test set (informational).\n")
    f.write(f"  Current fixed threshold : {THRESHOLD} (tuned for high recall / safety).\n")
    f.write(f"\n★ Best model (highest recall): {best_name}\n\n")
    f.write(f"Detailed report — {best_name}:\n")
    f.write(classification_report(y_te, best_pred, target_names=["Normal", "Fall"]))
    f.write(f"\nTraining configuration:\n")
    f.write(f"  Window size  : {WINDOW_SIZE} samples\n")
    f.write(f"  Step size    : {STEP_SIZE} samples\n")
    f.write(f"  Threshold    : {THRESHOLD}\n")
    f.write(f"  SMOTE        : enabled\n")
    f.write(f"  Total windows: {len(y):,} (Fall: {sum(y):,}, {sum(y)/len(y)*100:.1f}%)\n")

print(f"  Saved {metrics_path}")

# Save metrics as JSON for programmatic tracking across CI runs
metrics_json_path = os.path.join(OUTPUTS_DIR, "metrics.json")
with open(metrics_json_path, "w") as f:
    json.dump(
        {name: {"accuracy": r["acc"], "f1": r["f1"], "recall": r["recall"],
                "precision": r["precision"], "roc_auc": r["roc_auc"],
                "best_f1_threshold": r.get("best_f1_threshold"),
                "best_f1": r.get("best_f1")}
         for name, r in results.items()},
        f, indent=2,
    )
print(f"  Saved {metrics_json_path}")

# ── Step 8: Save plots ───────────────────────────────────────────────────────
print("  Plotting and saving visualizations ...")
# --- Confusion matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_te, r["pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fall"], yticklabels=["Normal", "Fall"])
    ax.set_title(f"{name}\nAcc={r['acc']:.3f}  Recall={r['recall']:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrices.png"), dpi=120)
plt.close()

# --- Training histories ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for row, name in enumerate(['Simple NN', 'LSTM']):
    hist = results[name]['history']
    axes[row, 0].plot(hist.history['loss'], label='Train')
    axes[row, 0].plot(hist.history['val_loss'], label='Val')
    axes[row, 0].set_title(f'{name} — Loss'); axes[row, 0].legend(); axes[row, 0].grid(alpha=0.3)
    axes[row, 1].plot(hist.history['accuracy'], label='Train')
    axes[row, 1].plot(hist.history['val_accuracy'], label='Val')
    axes[row, 1].set_title(f'{name} — Accuracy'); axes[row, 1].legend(); axes[row, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "training_history.png"), dpi=120)
plt.close()

# --- Feature importance (RF) ---
feat_imp = pd.Series(rf.feature_importances_, index=feature_names)
plt.figure(figsize=(10, 7))
feat_imp.nlargest(20).sort_values().plot(kind='barh', color='steelblue')
plt.title('Top-20 Feature Importances (Random Forest)'); plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=120)
plt.close()
print("  Saved confusion_matrices.png, training_history.png, feature_importance.png")


# ── Step 9: Save scalers and models ──────────────────────────────────────────
print("\nSTEP 9: Saving scalers and models for deployment")
# Scaler parameters as JSON (for firmware)
scaler_json = {
    'sample_rate': SAMPLE_RATE, 'window_size': WINDOW_SIZE, 'step_size': STEP_SIZE,
    'threshold': THRESHOLD, 'sensor_cols': SENSOR_COLS, 'feature_names': feature_names,
    'feat_mean': feat_scaler.mean_.tolist(), 'feat_scale': feat_scaler.scale_.tolist(),
    'seq_mean': seq_scaler.mean_.tolist(), 'seq_scale': seq_scaler.scale_.tolist(),
}
with open(os.path.join(OUTPUTS_DIR, 'scaler_params.json'), 'w') as f:
    json.dump(scaler_json, f, indent=2)
print("  Saved scaler_params.json")

# Python pickle objects
with open(os.path.join(OUTPUTS_DIR, 'fall_detector_rf.pkl'), 'wb') as f: pickle.dump(rf, f)
with open(os.path.join(OUTPUTS_DIR, 'feat_scaler.pkl'), 'wb') as f: pickle.dump(feat_scaler, f)
with open(os.path.join(OUTPUTS_DIR, 'seq_scaler.pkl'), 'wb') as f: pickle.dump(seq_scaler, f)
print("  Saved fall_detector_rf.pkl, feat_scaler.pkl, seq_scaler.pkl")

# Generate a C header so firmware can use the scaler without JSON parsing.
# acc_mag must be computed on-chip as sqrt(acc_x²+acc_y²+acc_z²) before
# feature extraction.  'peak' = abs(v).max(), 'delta' = v[-1]-v[0].
def _c_array(name, values, dtype='float', per_row=8):
    rows = []
    vals = list(values)
    for i in range(0, len(vals), per_row):
        rows.append('    ' + ', '.join(f'{v:.8f}f' for v in vals[i:i + per_row]))
    return f'static const {dtype} {name}[{len(vals)}] = {{\n' + ',\n'.join(rows) + '\n};\n'

h_lines = [
    '// Auto-generated by train.py — do not edit manually.',
    '// Copy next to your firmware sketch and #include it.',
    '#pragma once',
    '#include <stdint.h>',
    '',
    f'#define FALL_WINDOW_SIZE  {WINDOW_SIZE}    // samples per inference window',
    f'#define FALL_SAMPLE_RATE  {SAMPLE_RATE}    // Hz',
    f'#define FALL_STEP_SIZE    {STEP_SIZE}      // hop size in samples ({100 * (1 - STEP_SIZE / WINDOW_SIZE):.0f} % overlap)',
    f'#define FALL_N_FEATURES   {len(feature_names)} // statistical features per window',
    f'#define FALL_THRESHOLD    {THRESHOLD}f    // classification threshold',
    '',
    '// Feature-level scaler: apply AFTER extracting the 80 statistical features.',
    _c_array('FEAT_MEAN',  feat_scaler.mean_),
    _c_array('FEAT_SCALE', feat_scaler.scale_),
    '// Sequence-level scaler: apply to raw sensor channels BEFORE LSTM inference.',
    _c_array('SEQ_MEAN',  seq_scaler.mean_),
    _c_array('SEQ_SCALE', seq_scaler.scale_),
]
header_path = os.path.join(OUTPUTS_DIR, 'scaler_params.h')
with open(header_path, 'w') as f:
    f.write('\n'.join(h_lines))
print(f"  Saved {header_path}")


# ── Step 10: Export models for microcontrollers ──────────────────────────────
# rf_mcu (15 trees, depth 8) is exported instead of the full 200-tree RF so the
# resulting C code fits within the Flash of AVR / STM32 class devices (≤256 KB).
# TFLite (NN / LSTM) exports require additional int8 quantization configuration
# and are kept separate from the main CI loop.
print("\nSTEP 10: Exporting MCU-friendly models")

try:
    c_code = mlport(rf_mcu, classmap={0: 'NORMAL', 1: 'FALL'})
    with open(os.path.join(OUTPUTS_DIR, 'fall_detector_micromlgen.h'), 'w') as f: f.write(c_code)
    print(f"  micromlgen → fall_detector_micromlgen.h ({len(c_code)//1024} KB)")
except Exception as e: print(f"  micromlgen → SKIPPED ({e})")

try:
    c_code_m2c = m2c.export_to_c(rf_mcu)
    with open(os.path.join(OUTPUTS_DIR, 'fall_detector_m2cgen.c'), 'w') as f: f.write(c_code_m2c)
    print(f"  m2cgen     → fall_detector_m2cgen.c ({len(c_code_m2c)//1024} KB)")
except Exception as e: print(f"  m2cgen     → SKIPPED ({e})")

try:
    emlearn_model = emlearn.convert(rf_mcu, method='inline')
    emlearn_model.save(file=os.path.join(OUTPUTS_DIR, 'fall_detector_emlearn.h'))
    size = os.path.getsize(os.path.join(OUTPUTS_DIR, 'fall_detector_emlearn.h')) // 1024
    print(f"  emlearn    → fall_detector_emlearn.h ({size} KB)")
except Exception as e: print(f"  emlearn    → SKIPPED ({e})")


# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training complete!")
print(f"  Best model        : {best_name}")
print(f"  Precision         : {results[best_name]['precision']:.4f}")
print(f"  Recall            : {results[best_name]['recall']:.4f}")
print(f"  F1 Score          : {results[best_name]['f1']:.4f}")
print(f"  ROC-AUC           : {results[best_name]['roc_auc']:.4f}")
if results[best_name].get('best_f1_threshold') is not None:
    print(f"  Best-F1 threshold : {results[best_name]['best_f1_threshold']:.3f}"
          f"  (F1={results[best_name]['best_f1']:.4f} on test set)")
