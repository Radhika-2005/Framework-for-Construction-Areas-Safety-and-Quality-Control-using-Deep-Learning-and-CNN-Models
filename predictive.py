# predictive_cmaps_lstm.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score)
from sklearn.utils import class_weight
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)
plt.ion()

start_time = time.time()

print("=" * 90)
print("⚡ NASA C-MAPS PREDICTIVE MAINTENANCE LSTM")
print("=" * 90)

# ==================== CONFIG ====================
dataset_dir = 'Predictive Maintenance Dataset'  # change if needed
train_file = 'train_FD001.txt'
test_file = 'test_FD001.txt'
rul_file = 'RUL_FD001.txt'

# Column names (26 columns: engine_id, cycle, 24 sensor/OS)
col_names = ['Engine_ID', 'Cycle', 'OS1', 'OS2', 'OS3',
             'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
             'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21']

# Sequence length and RUL threshold
SEQ_LENGTH = 50
RUL_THRESHOLD = 30

# ==================== STEP 1: LOAD DATA ====================
print("\n📥 Loading dataset...")

train_path = os.path.join(dataset_dir, train_file)
test_path = os.path.join(dataset_dir, test_file)
rul_path = os.path.join(dataset_dir, rul_file)

# basic file existence check
for p in [train_path, test_path, rul_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}\nPlease download/extract NASA C-MAPs (FD001) files into '{dataset_dir}'.")

# Load (allow variable whitespace)
train_data = pd.read_csv(train_path, sep='\s+', header=None)
test_data = pd.read_csv(test_path, sep='\s+', header=None)
rul_data = pd.read_csv(rul_path, sep='\s+', header=None)

# Some files may include extra trailing columns; keep first 26
train_data = train_data.iloc[:, :26]
test_data = test_data.iloc[:, :26]

train_data.columns = col_names
test_data.columns = col_names

print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}, RUL rows: {len(rul_data)}")

# ==================== STEP 2: PREPROCESS & BUILD RUL / FAILURE ====================
print("\n🔧 Preprocessing and RUL/Failure label creation...")

# Create RUL for training: for each engine, RUL = cycles - current_cycle + 1
train_processed = []
for eng in train_data['Engine_ID'].unique():
    df_e = train_data[train_data['Engine_ID'] == eng].reset_index(drop=True)
    cycles = len(df_e)
    for i in range(cycles):
        rul = cycles - i
        row = df_e.loc[i, :].to_dict()
        row['RUL'] = rul
        train_processed.append(row)
train_df = pd.DataFrame(train_processed)

# For test set, RUL file gives remaining life at last cycle for each engine
# rul_data is one value per engine in order
rul_vals = [int(x) for x in rul_data.iloc[:, 0].values]
rul_dict = {i+1: rul_vals[i] for i in range(len(rul_vals))}

test_processed = []
for eng in sorted(test_data['Engine_ID'].unique()):
    df_e = test_data[test_data['Engine_ID'] == eng].reset_index(drop=True)
    cycles = len(df_e)
    remaining = rul_dict.get(eng, cycles)
    # RUL at each cycle = remaining + (cycles - current_index - 1)
    for i in range(cycles):
        rul = remaining + (cycles - i - 1)
        row = df_e.loc[i, :].to_dict()
        row['RUL'] = rul
        test_processed.append(row)
test_df = pd.DataFrame(test_processed)

# Sensor columns (S1..S21)
sensor_cols = [c for c in col_names if c.startswith('S')]
print(f"Using sensor columns: {sensor_cols}")

# Combine to compute failure label then split again (fit scaler on train only later)
combined = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')], ignore_index=True)

# Binary failure label (RUL < threshold)
combined['Failure'] = (combined['RUL'] < RUL_THRESHOLD).astype(int)
print("Failure distribution (combined):")
print(combined['Failure'].value_counts().to_dict())

# Split back
train_df = combined[combined['dataset'] == 'train'].drop(columns=['dataset']).reset_index(drop=True)
test_df = combined[combined['dataset'] == 'test'].drop(columns=['dataset']).reset_index(drop=True)

# ==================== STEP 3: NORMALIZE (fit scaler on train only) ====================
print("\n🔁 Scaling sensor features (fit on train only)...")
scaler = StandardScaler()
scaler.fit(train_df[sensor_cols].values)          # fit only on training sensor data
train_df[sensor_cols] = scaler.transform(train_df[sensor_cols].values)
test_df[sensor_cols] = scaler.transform(test_df[sensor_cols].values)

# save scaler for later
with open('scaler_cmaps.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ==================== STEP 4: CREATE SEQUENCES PER ENGINE ====================
print("\n⏱ Creating sequences per engine (seq_length=%d)..." % SEQ_LENGTH)

def create_sequences_from_df(df, sensor_columns, seq_length):
    X_seqs = []
    y_seqs = []
    engine_ids = []
    # iterate per engine to avoid sequences crossing engine boundaries
    for eng in sorted(df['Engine_ID'].unique()):
        eng_df = df[df['Engine_ID'] == eng].reset_index(drop=True)
        n = len(eng_df)
        if n < seq_length:
            continue
        sensors = eng_df[sensor_columns].values
        labels = eng_df['Failure'].values
        # sliding window: label for a sequence = label at last timestep of sequence
        for start in range(0, n - seq_length + 1):
            X_seqs.append(sensors[start:start + seq_length])
            y_seqs.append(labels[start + seq_length - 1])
            engine_ids.append(eng)
    return np.array(X_seqs), np.array(y_seqs), np.array(engine_ids)

X_train_seq, y_train_seq, train_eng_ids = create_sequences_from_df(train_df, sensor_cols, SEQ_LENGTH)
X_test_seq, y_test_seq, test_eng_ids = create_sequences_from_df(test_df, sensor_cols, SEQ_LENGTH)

print(f"Train sequences: {X_train_seq.shape}, Train positives: {np.sum(y_train_seq)}")
print(f"Test sequences:  {X_test_seq.shape}, Test positives:  {np.sum(y_test_seq)}")

# If extremely imbalanced (no failures), warn but continue
if np.sum(y_train_seq) == 0 or np.sum(y_test_seq) == 0:
    print("WARNING: No positive failure samples in train or test sequences. "
          "Consider lowering RUL_THRESHOLD or using a different FD dataset (FD002/FD003/FD004).")

# ==================== STEP 5: CLASS WEIGHTS (handle imbalance) ====================
print("\n⚖️ Computing class weights to handle imbalance...")
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
class_weights = {i: cw[i] for i in range(len(cw))}
print("Class weights:", class_weights)

# ==================== STEP 6: BUILD LSTM MODEL ====================
print("\n🧠 Building LSTM model...")
n_features = X_train_seq.shape[2]

model = models.Sequential([
    layers.Input(shape=(SEQ_LENGTH, n_features)),
    layers.LSTM(256, return_sequences=True, name='lstm_1'),
    layers.BatchNormalization(),
    layers.SpatialDropout1D(0.25),

    layers.LSTM(128, return_sequences=True, name='lstm_2'),
    layers.BatchNormalization(),
    layers.SpatialDropout1D(0.25),

    layers.LSTM(64, return_sequences=False, name='lstm_3'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu', name='dense_1'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(64, activation='relu', name='dense_2'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu', name='dense_3'),
    layers.Dropout(0.2),

    layers.Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=8e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
)

print(model.summary())
print("Total params:", model.count_params())

# ==================== STEP 7: TRAIN ====================
print("\n🚀 Training model...")

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, min_delta=1e-4, verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=50,
    batch_size=128,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# ==================== STEP 8: EVALUATE ====================
print("\n📈 Evaluating model on test sequences...")
y_pred_prob = model.predict(X_test_seq, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test_seq, y_pred)
precision = precision_score(y_test_seq, y_pred, zero_division=0)
recall = recall_score(y_test_seq, y_pred, zero_division=0)
f1 = f1_score(y_test_seq, y_pred, zero_division=0)
auc = roc_auc_score(y_test_seq, y_pred_prob) if len(np.unique(y_test_seq)) > 1 else float('nan')

print(f"\n✓ TEST ACCURACY:  {accuracy*100:.2f}%")
print(f"✓ PRECISION:      {precision:.4f}")
print(f"✓ RECALL:         {recall:.4f}")
print(f"✓ F1-SCORE:       {f1:.4f}")
print(f"✓ AUC-ROC:        {auc:.4f}")

cm = confusion_matrix(y_test_seq, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\nConfusion matrix:")
print(cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Sensitivity (Recall)={sensitivity:.4f}, Specificity={specificity:.4f}")

print("\nClassification report:")
print(classification_report(y_test_seq, y_pred, target_names=['Normal', 'Failure'], zero_division=0))

# ==================== STEP 9: SAVE ARTIFACTS (use UTF-8 encoding) ====================
print("\n💾 Saving model and artifacts...")
model.save('nasa_cmaps_lstm_model.keras')  # recommended Keras format
model.save_weights('nasa_cmaps_lstm.weights.h5')

with open('scaler_cmaps.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('training_history_cmaps.json', 'w', encoding='utf-8') as f:
    json.dump({
        'accuracy': [float(x) for x in history.history.get('accuracy',[])],
        'val_accuracy': [float(x) for x in history.history.get('val_accuracy',[])],
        'loss': [float(x) for x in history.history.get('loss',[])],
        'val_loss': [float(x) for x in history.history.get('val_loss',[])]
    }, f, indent=2)

model_info = f"""NASA C-MAPS PREDICTIVE MAINTENANCE LSTM MODEL - FINAL REPORT
{'='*80}

MODEL PERFORMANCE:
  Accuracy:     {accuracy*100:.2f}%
  Precision:    {precision:.4f}
  Recall:       {recall:.4f}
  F1-Score:     {f1:.4f}
  AUC-ROC:      {auc:.4f}
  Sensitivity:  {sensitivity:.4f}
  Specificity:  {specificity:.4f}

CONFUSION MATRIX:
  TN: {tn}
  FP: {fp}
  FN: {fn}
  TP: {tp}

DATA INFO:
  Train engines: {train_df['Engine_ID'].nunique()}
  Test engines:  {test_df['Engine_ID'].nunique()}
  Train sequences: {len(X_train_seq)}
  Test sequences:  {len(X_test_seq)}
  Sequence length:  {SEQ_LENGTH}
  Sensor features:  {len(sensor_cols)}
  RUL threshold:    {RUL_THRESHOLD}

Model params: {model.count_params():,}
Execution time: {time.time()-start_time:.1f} seconds
"""
with open('model_info_cmaps.txt', 'w', encoding='utf-8') as f:
    f.write(model_info)

print("Saved: nasa_cmaps_lstm_model.keras, nasa_cmaps_lstm.weights.h5, scaler_cmaps.pkl, training_history_cmaps.json, model_info_cmaps.txt")

# ==================== STEP 10: VISUALIZE RESULTS ====================
print("\n📊 Generating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('NASA C-MAPS LSTM Predictive Maintenance', fontsize=16, fontweight='bold')

# Accuracy
axes[0, 0].plot(history.history.get('accuracy', []), 'o-', label='Train')
axes[0, 0].plot(history.history.get('val_accuracy', []), 's-', label='Val')
axes[0, 0].set_title('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.2)

# Loss
axes[0, 1].plot(history.history.get('loss', []), 'o-', label='Train')
axes[0, 1].plot(history.history.get('val_loss', []), 's-', label='Val')
axes[0, 1].set_title('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.2)

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 2], xticklabels=['Normal','Failure'], yticklabels=['Normal','Failure'])
axes[0, 2].set_title('Confusion Matrix')

metrics = ['Accuracy','Precision','Recall','F1','AUC','Sensitivity','Specificity']
vals = [accuracy, precision, recall, f1, auc if not np.isnan(auc) else 0, sensitivity, specificity]
axes[1, 0].bar(range(len(metrics)), vals)
axes[1, 0].set_xticks(range(len(metrics)))
axes[1, 0].set_xticklabels(metrics, rotation=45)
axes[1, 0].set_ylim([0,1.05])
axes[1, 0].set_title('Metrics')

# Probability distribution
axes[1, 1].hist(y_pred_prob[y_test_seq==0], bins=40, alpha=0.7, label='Normal')
axes[1, 1].hist(y_pred_prob[y_test_seq==1], bins=40, alpha=0.7, label='Failure')
axes[1, 1].axvline(0.5, linestyle='--')
axes[1, 1].legend()
axes[1, 1].set_title('Prediction probability distribution')

# summary text
axes[1, 2].axis('off')
summary = f"Accuracy: {accuracy*100:.2f}%\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nAUC: {auc:.4f}"
axes[1, 2].text(0.02, 0.95, summary, fontsize=12, va='top', family='monospace')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('nasa_cmaps_model_performance.png', dpi=150, bbox_inches='tight')
plt.show()

elapsed = time.time() - start_time
print("\n" + "=" * 90)
print(f"Training complete in {elapsed:.1f} s")
print("=" * 90)
