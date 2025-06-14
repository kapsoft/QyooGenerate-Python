# test_with_ground_truth_overlay.py
from ultralytics import YOLO
import random
import os
import cv2
import numpy as np

def calculate_iou(pred_box, gt_polygon, img_w, img_h):
    """Calculate IoU between predicted bounding box and ground truth polygon"""
    if not gt_polygon:
        return 0.0
    
    # Create masks
    pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # Fill prediction box
    x1, y1, x2, y2 = pred_box
    pred_mask[y1:y2, x1:x2] = 1
    
    # Fill ground truth polygon
    pts = np.array(gt_polygon, np.int32)
    cv2.fillPoly(gt_mask, [pts], 1)
    
    # Calculate IoU
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO segmentation format labels and convert to pixel coordinates"""
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 9:  # class + bbox (4) + at least 4 polygon coords
                class_id = int(parts[0])
                
                # Bounding box (for reference)
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Segmentation polygon points (starting from index 5)
                poly_coords = []
                for i in range(5, len(parts), 2):
                    if i + 1 < len(parts):
                        x = float(parts[i]) * img_width
                        y = float(parts[i + 1]) * img_height
                        poly_coords.append((int(x), int(y)))
                
                labels.append((class_id, poly_coords, (x_center, y_center, width, height)))
    return labels

def draw_comparison(img_path, label_path, model_results, save_path):
    """Draw ground truth (green polygon) and predictions (red boxes) on image"""
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # Draw ground truth segmentation in GREEN
    gt_labels = load_yolo_labels(label_path, w, h)
    for class_id, poly_coords, bbox in gt_labels:
        if poly_coords:
            # Draw the actual Qyoo shape outline
            pts = np.array(poly_coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 3)  # Green outline
            cv2.putText(image, f'GT: qyoo', (poly_coords[0][0], poly_coords[0][1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw predictions in RED
    if model_results[0].boxes:
        for box in model_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
            cv2.putText(image, f'PRED: {conf:.2f}', (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imwrite(save_path, image)
    return len(gt_labels), len(model_results[0].boxes) if model_results[0].boxes else 0

# Load the trained model
model = YOLO('runs/segment/train_quick_test4/weights/best.pt')

# Test on 100 random images with ground truth overlay
print("Testing 100 random images with ground truth comparison...\n")
test_indices = random.sample(range(5000), 100)

os.makedirs('validation_output', exist_ok=True)

# Track statistics
detections = 0
no_detections = 0
confidences = []
matches = 0
mismatches = 0
ious = []
good_detections_30 = 0  # IoU > 0.3
good_detections_50 = 0  # IoU > 0.5
good_detections_70 = 0  # IoU > 0.7

for i, idx in enumerate(test_indices):
    img_path = f'src/dataset_test/train/images/{idx:06d}.jpg'
    label_path = f'src/dataset_test/train/labels/{idx:06d}.txt'
    
    results = model.predict(img_path, conf=0.25, save=False, verbose=False)
    
    # Only save comparison images for first 10 to avoid clutter
    if i < 10:
        save_path = f'validation_output/{idx:06d}_comparison.jpg'
        gt_count, pred_count = draw_comparison(img_path, label_path, results, save_path)
    else:
        # Still count ground truth for statistics
        gt_labels = load_yolo_labels(label_path, 640, 640)  # Assuming 640x640 images
        gt_count = len(gt_labels)
        pred_count = len(results[0].boxes) if results[0].boxes else 0
    
    # Calculate IoU if we have both prediction and ground truth
    best_iou = 0.0
    if results[0].boxes:
        gt_labels = load_yolo_labels(label_path, 640, 640)
        if gt_labels:
            # Get best IoU (in case of multiple predictions/ground truths)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                pred_box = [x1, y1, x2, y2]
                
                for class_id, poly_coords, bbox in gt_labels:
                    iou = calculate_iou(pred_box, poly_coords, 640, 640)
                    best_iou = max(best_iou, iou)
            
            ious.append(best_iou)
            
            # Count good detections at different thresholds
            if best_iou > 0.3:
                good_detections_30 += 1
            if best_iou > 0.5:
                good_detections_50 += 1  
            if best_iou > 0.7:
                good_detections_70 += 1
    
    # Track statistics
    if results[0].boxes:
        detections += 1
        conf = results[0].boxes.conf[0].item()
        confidences.append(conf)
    else:
        no_detections += 1
    
    # Track matches vs mismatches
    if (gt_count > 0 and pred_count > 0):
        matches += 1
        status = "✅ MATCH"
    else:
        mismatches += 1
        status = "❌ MISMATCH"
    
    # Print detailed results for first 10
    if i < 10:
        iou_str = f" IoU: {best_iou:.3f}" if best_iou > 0 else ""
        if results[0].boxes:
            conf = results[0].boxes.conf[0].item()
            print(f"Image {idx:06d}: {status} | GT: {gt_count} | PRED: {pred_count} ({conf:.2%}){iou_str}")
        else:
            print(f"Image {idx:06d}: {status} | GT: {gt_count} | PRED: {pred_count}{iou_str}")

# Summary statistics
print(f"\n📊 SUMMARY (100 test images):")
print(f"✅ Detections: {detections} ({detections}%)")
print(f"❌ No detections: {no_detections} ({no_detections}%)")
print(f"🎯 Matches (GT + PRED): {matches} ({matches}%)")
print(f"💥 Mismatches: {mismatches} ({mismatches}%)")

print(f"\n🎯 IoU-BASED ACCURACY:")
print(f"📈 Good detections (IoU > 0.3): {good_detections_30} ({good_detections_30}%)")
print(f"📈 Good detections (IoU > 0.5): {good_detections_50} ({good_detections_50}%)")  
print(f"📈 Good detections (IoU > 0.7): {good_detections_70} ({good_detections_70}%)")

if confidences:
    avg_conf = sum(confidences) / len(confidences)
    print(f"\n📈 Average confidence: {avg_conf:.2%}")
    print(f"📈 Min confidence: {min(confidences):.2%}")
    print(f"📈 Max confidence: {max(confidences):.2%}")

if ious:
    avg_iou = sum(ious) / len(ious)
    print(f"📈 Average IoU: {avg_iou:.3f}")
    print(f"📈 Min IoU: {min(ious):.3f}")
    print(f"📈 Max IoU: {max(ious):.3f}")

print(f"\n💾 First 10 comparison images saved to: validation_output/")
print(f"🟢 Green boxes = Ground Truth")
print(f"🔴 Red boxes = Model Predictions") 