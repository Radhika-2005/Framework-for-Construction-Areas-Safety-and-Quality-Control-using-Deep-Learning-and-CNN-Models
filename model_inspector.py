import os
from ultralytics import YOLO
import yaml

print("=" * 80)
print("🔍 PPE MODEL CLASS INSPECTOR")
print("=" * 80)

# ==================== STEP 1: FIND MODEL ====================
print("\n📁 STEP 1: Looking for trained model...")
print("-" * 80)

model_paths = [
    'runs/detect/train/weights/best.pt',
    'runs/detect/train/weights/last.pt',
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        print(f"✅ Found: {path}")
        model_path = path
        break

if model_path is None:
    print("❌ Model not found!")
    print("Please train first: python src/ppe_detection.py")
    exit()

# ==================== STEP 2: LOAD MODEL ====================
print("\n🧠 STEP 2: Loading Model...")
print("-" * 80)

try:
    model = YOLO(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==================== STEP 3: CHECK MODEL CLASSES ====================
print("\n📊 STEP 3: Checking Model Classes...")
print("-" * 80)

try:
    # Get model names (classes)
    model_names = model.names
    num_classes = model.model.nc
    
    print(f"Number of classes: {num_classes}")
    print(f"\nClasses the model can detect:")
    
    for class_id, class_name in model_names.items():
        print(f"  Class {class_id}: {class_name}")
    
except Exception as e:
    print(f"Error: {e}")

# ==================== STEP 4: CHECK TRAINING DATA ====================
print("\n📋 STEP 4: Checking Training Configuration...")
print("-" * 80)

yaml_path = 'PPE_YOLO_Dataset/data.yaml'

if os.path.exists(yaml_path):
    print(f"✅ Found data.yaml")
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nTraining Data Configuration:")
    print(f"  Path: {data_config.get('path', 'N/A')}")
    print(f"  Number of classes: {data_config.get('nc', 'N/A')}")
    print(f"  Classes: {data_config.get('names', [])}")
else:
    print(f"⚠️  data.yaml not found at {yaml_path}")

# ==================== STEP 5: TEST WITH SAMPLE IMAGE ====================
print("\n🧪 STEP 5: Testing Model on Sample Images...")
print("-" * 80)

val_dir = 'PPE_YOLO_Dataset/images/val'

if os.path.exists(val_dir):
    val_images = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if val_images:
        print(f"Found {len(val_images)} validation images")
        
        # Test on first 3 images
        test_images = val_images[:3]
        
        all_detections = {}
        
        for img_file in test_images:
            img_path = os.path.join(val_dir, img_file)
            print(f"\n📸 Testing: {img_file}")
            
            results = model.predict(img_path, conf=0.5, verbose=False)
            
            if len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            if class_name not in all_detections:
                                all_detections[class_name] = []
                            
                            all_detections[class_name].append(confidence)
                            print(f"   ✓ {class_name}: {confidence*100:.2f}%")
                    else:
                        print(f"   ⚠️  No detections found")
            else:
                print(f"   ⚠️  No results returned")
        
        if all_detections:
            print(f"\n📊 Classes Detected Across All Test Images:")
            for class_name, confidences in sorted(all_detections.items()):
                avg_conf = sum(confidences) / len(confidences)
                print(f"  {class_name}: {len(confidences)} detections, avg confidence: {avg_conf*100:.2f}%")
        else:
            print(f"\n⚠️  No objects detected in any test image!")
    else:
        print(f"❌ No validation images found in {val_dir}")
else:
    print(f"❌ Validation directory not found: {val_dir}")

# ==================== STEP 6: SUMMARY ====================
print("\n" + "=" * 80)
print("📝 SUMMARY")
print("=" * 80)

print(f"\n✅ Model Information:")
print(f"  Model Path: {model_path}")
print(f"  Number of Classes: {num_classes}")
print(f"  Classes: {list(model.names.values())}")

print(f"\n❓ What the model detects:")

if 'helmet' in str(model.names.values()).lower():
    print(f"  ✅ Helmet")
if 'vest' in str(model.names.values()).lower():
    print(f"  ✅ Vest")
if 'gloves' in str(model.names.values()).lower():
    print(f"  ✅ Gloves")
if 'boots' in str(model.names.values()).lower():
    print(f"  ✅ Boots")

if len(all_detections) == 1:
    print(f"\n⚠️  WARNING: Model only detected ONE class!")
    print(f"  Only detected: {list(all_detections.keys())[0]}")
    print(f"\n  This means:")
    print(f"    • Training data likely only had that class")
    print(f"    • OR annotations only labeled that class")
    print(f"    • OR model wasn't trained on all 4 classes")
    
    print(f"\n  To fix:")
    print(f"    1. Add images with vest, gloves, boots")
    print(f"    2. Annotate all 4 classes properly")
    print(f"    3. Retrain: python src/ppe_detection.py")
else:
    print(f"\n✅ Model can detect {len(all_detections)} classes!")

print("\n" + "=" * 80)