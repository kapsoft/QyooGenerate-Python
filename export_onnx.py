from ultralytics import YOLO

# Load the model
model = YOLO('runs/segment/train_full4/weights/best.pt')

# Export the model to CoreML format
model.export(format='coreml', imgsz=512) 