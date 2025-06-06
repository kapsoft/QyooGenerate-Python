# Test CoreML model conversion and inference
from ultralytics import YOLO
import coremltools as ct
import numpy as np
from PIL import Image

# 1. Test PyTorch model
model = YOLO('runs/segment/train_full4/weights/best.pt')
results = model.predict('dataset/val/images/000000.jpg', save=True)
print(f"PyTorch detections: {len(results[0].boxes) if results[0].boxes else 0}")

# 2. Export to CoreML
model.export(format='coreml', imgsz=512, nms=False)  # NMS=False for raw outputs

# 3. Test CoreML model
ml_model = ct.models.MLModel('runs/segment/train_full4/weights/best.mlpackage')

# Prepare test image
img = Image.open('dataset/val/images/000000.jpg').resize((512, 512))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
prediction = ml_model.predict({'image': img_array})
print("CoreML outputs:", prediction.keys())

# Check detection confidence
detections = prediction['var_1052']
max_conf = 0
for i in range(detections.shape[2]):
    conf = detections[0, 4, i]
    if conf > max_conf:
        max_conf = conf
print(f"Max confidence: {max_conf}") 