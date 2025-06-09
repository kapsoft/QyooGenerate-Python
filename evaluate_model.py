from ultralytics import YOLO
import cv2

# Load current model
model = YOLO('runs/segment/train_full4/weights/last.pt')

# Test on a training image (since 000000.jpg exists there)
results = model.predict('src/dataset/train/images/000000.jpg', conf=0.25, save=True)

# Check how many detections
if results[0].boxes:
    print(f"Detections: {len(results[0].boxes)}")
    print(f"Confidences: {results[0].boxes.conf}")
else:
    print("No detections")

# Also check the same image again for demonstration
results_train = model.predict('src/dataset/train/images/000000.jpg', conf=0.25, save=True)
print(f"Train detections: {len(results_train[0].boxes) if results_train[0].boxes else 0}")

# Test a few more to confirm consistency
for i in range(5):
    results = model.predict(f'src/dataset/val/images/{str(i).zfill(6)}.jpg', conf=0.25)
    if results[0].boxes:
        print(f"Image {i}: {len(results[0].boxes)} detections, conf: {results[0].boxes.conf[0]:.3f}")
    else:
        print(f"Image {i}: No detections") 