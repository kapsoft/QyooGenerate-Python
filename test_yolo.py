from ultralytics import YOLO

model = YOLO('runs/segment/train_full4/weights/best.pt')
results = model.predict('src/dataset/train/test/000000_vis.jpg', conf=0.3, save=True, save_txt=True)

# Check mask quality
if results[0].masks:
    print(f"Mask shape: {results[0].masks.data.shape}")
    print(f"Mask confidence: {results[0].boxes.conf[0]:.3f}") 