import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import cv2
import time
import json

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
plt.ion()

start_time = time.time()

print("=" * 70)
print("⚡⚡ LIGHTNING FAST CNN - 85%+ ACCURACY IN SECONDS")
print("=" * 70)

# ==================== LOAD DATA ====================
print("\nSTEP 1: LOADING DATA")
print("-" * 70)

base_path = "Concrete"
negative_path = os.path.join(base_path, "Negative")
positive_path = os.path.join(base_path, "Positive")

all_neg = [f for f in os.listdir(negative_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
all_pos = [f for f in os.listdir(positive_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 🔥 MINIMAL DATASET FOR SPEED
DATASET_SIZE = 400  # Only 400 images
neg_images = all_neg[:DATASET_SIZE//2]
pos_images = all_pos[:DATASET_SIZE//2]

print(f"⚡ Using {len(neg_images) + len(pos_images)} images")
print(f"   Negative: {len(neg_images)}, Positive: {len(pos_images)}")

# Load images
print("\n⏳ Loading images...")
IMG_SIZE = 96  # Even faster processing
x_data = []
y_data = []

for img_name in neg_images:
    img = cv2.imread(os.path.join(negative_path, img_name))
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        x_data.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        y_data.append(0)

for img_name in pos_images:
    img = cv2.imread(os.path.join(positive_path, img_name))
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        x_data.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        y_data.append(1)

x_data = np.array(x_data, dtype='float32') / 255.0
y_data = np.array(y_data)

print(f"✓ Loaded {len(x_data)} images ({x_data.nbytes / 1024 / 1024:.1f} MB)")

# Shuffle and split
idx = np.random.permutation(len(x_data))
x_data = x_data[idx]
y_data = y_data[idx]

split_idx = int(0.8 * len(x_data))
x_train, x_test = x_data[:split_idx], x_data[split_idx:]
y_train, y_test = y_data[:split_idx], y_data[split_idx:]

print(f"✓ Train: {len(x_train)}, Test: {len(x_test)}")

# ==================== BUILD FAST MODEL ====================
print("\nSTEP 2: BUILDING MODEL")
print("-" * 70)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"✓ Model compiled - Parameters: {model.count_params():,}")

# ==================== TRAIN ====================
print("\nSTEP 3: TRAINING (FAST!)")
print("-" * 70)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

print("⏳ Training...\n")

history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=32),
    epochs=8,
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // 32,
    verbose=1
)

# ==================== EVALUATE ====================
print("\n" + "=" * 70)
print("STEP 4: EVALUATION")
print("-" * 70)

y_pred_prob = model.predict(x_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype('int32').flatten()

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n✓ TEST ACCURACY: {accuracy*100:.2f}%")
print(f"✓ AUC-ROC: {auc:.4f}")

if accuracy >= 0.85:
    print("\n✅ MEETS 85%+ TARGET!")
else:
    print(f"\n⚠ Current: {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print(f"\nCONFUSION MATRIX:")
print(f"TN: {cm[0,0]}  FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}  TP: {cm[1,1]}")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['No Crack', 'Crack']))

sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

print(f"\n✓ Sensitivity: {sensitivity:.4f}")
print(f"✓ Specificity: {specificity:.4f}")

# ==================== SAVE MODEL ====================
print("\n" + "=" * 70)
print("STEP 5: SAVING MODEL")
print("-" * 70)

model.save('concrete_crack_model.h5')
print("✅ Model saved: concrete_crack_model.h5")

model.save_weights('concrete_crack_model.weights.h5')
print("✅ Weights saved: concrete_crack_model.weights.h5")

model.export('concrete_crack_saved_model')
print("✅ SavedModel saved: concrete_crack_saved_model/")

with open('training_history.json', 'w') as f:
    json.dump({
        'accuracy': list(map(float, history.history['accuracy'])),
        'val_accuracy': list(map(float, history.history['val_accuracy'])),
        'loss': list(map(float, history.history['loss'])),
        'val_loss': list(map(float, history.history['val_loss']))
    }, f)
print("✅ History saved: training_history.json")

with open('model_info.txt', 'w') as f:
    f.write(f"""CONCRETE CRACK DETECTION MODEL
====================================
Test Accuracy: {accuracy*100:.2f}%
AUC-ROC: {auc:.4f}

Sensitivity: {sensitivity:.4f}
Specificity: {specificity:.4f}

Dataset: {len(x_train)} train, {len(x_test)} test
Image Size: {IMG_SIZE}x{IMG_SIZE}
Model Parameters: {model.count_params():,}
Training Epochs: {len(history.history['accuracy'])}

Confusion Matrix:
TN: {cm[0,0]}  FP: {cm[0,1]}
FN: {cm[1,0]}  TP: {cm[1,1]}
""")
print("✅ Info saved: model_info.txt")

# ==================== VISUALIZE ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fast Training Results", fontsize=16, fontweight='bold')

axes[0, 0].plot(history.history['accuracy'], 'o-', label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], 's-', label='Test', linewidth=2)
axes[0, 0].axhline(y=0.85, color='g', linestyle='--', alpha=0.5, label='85% Target')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(history.history['loss'], 'o-', label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], 's-', label='Test', linewidth=2)
axes[0, 1].set_title('Loss')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Crack', 'Crack'],
            yticklabels=['No Crack', 'Crack'],
            ax=axes[1, 0], cbar_kws={"label": "Count"})
axes[1, 0].set_title('Confusion Matrix')

axes[1, 1].axis('off')
summary = f"""RESULTS SUMMARY

✓ Accuracy: {accuracy*100:.2f}%
✓ AUC-ROC: {auc:.4f}

✓ TP: {cm[1,1]}  TN: {cm[0,0]}
✓ FP: {cm[0,1]}  FN: {cm[1,0]}

✓ Sensitivity: {sensitivity:.4f}
✓ Specificity: {specificity:.4f}

⏱ Training Time: {time.time()-start_time:.1f}s
📊 Dataset: {len(x_data)} images
"""
axes[1, 1].text(0.1, 0.95, summary, fontsize=12, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.show()

# ==================== FINAL SUMMARY ====================
elapsed = time.time() - start_time
print("\n" + "=" * 70)
print(f"✅ COMPLETE IN {elapsed:.1f} SECONDS!")
print("=" * 70)
print(f"📊 Accuracy: {accuracy*100:.2f}%")
print(f"⏱ Time: {elapsed:.1f} seconds")
print(f"📁 Files: All saved in current directory")
print("=" * 70)