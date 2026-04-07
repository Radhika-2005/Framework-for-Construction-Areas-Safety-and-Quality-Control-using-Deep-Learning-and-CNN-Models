import json
import time
from ultralytics import YOLO

print("=" * 80)
print("📊 PPE DETECTION – SAVE METRICS ONLY (NO TRAINING)")
print("=" * 80)

start_time = time.time()

# ------------------------------
# 1. Load best trained model
# ------------------------------
best_model_path = "runs/detect/train/weights/best.pt"

try:
    model = YOLO(best_model_path)
    print(f"✓ Loaded model: {best_model_path}")
except Exception as e:
    print(f"❌ Could not load best.pt, trying last.pt")
    best_model_path = "runs/detect/train/weights/last.pt"
    model = YOLO(best_model_path)
    print(f"✓ Loaded model: {best_model_path}")

# ------------------------------
# 2. Run validation quickly
# ------------------------------
print("\n🔍 Running quick validation...")
results = model.val(verbose=False)

val_map50 = results.box.map50
val_map5095 = results.box.map
precision = results.box.mp
recall = results.box.mr

accuracy = val_map50 * 100  # Primary accuracy

print("\n📊 METRICS")
print(f"mAP@0.5:      {accuracy:.2f}%")
print(f"mAP@0.5:0.95: {val_map5095*100:.2f}%")
print(f"Precision:    {precision*100:.2f}%")
print(f"Recall:       {recall*100:.2f}%")

# ------------------------------
# 3. Save JSON safely
# ------------------------------
output_data = {
    "accuracy_map50": float(accuracy),
    "map50": float(val_map50 * 100),
    "map50_95": float(val_map5095 * 100),
    "precision": float(precision * 100),
    "recall": float(recall * 100),
    "target_achieved": bool(accuracy >= 85)
}

with open("ppe_metrics.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("\n💾 JSON saved: ppe_metrics.json")

elapsed = time.time() - start_time
print(f"\n⏱ Completed in {elapsed:.2f} seconds")
print("=" * 80)
