#!/usr/bin/env python3
"""
Test validation using actual segmentation masks instead of just bounding boxes
"""

from ultralytics import YOLO
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_mask_iou(pred_mask, gt_polygon, img_w, img_h):
    """Calculate IoU between predicted mask and ground truth polygon"""
    if not gt_polygon:
        return 0.0
    
    # Create ground truth mask
    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts = np.array(gt_polygon, np.int32)
    cv2.fillPoly(gt_mask, [pts], 1)
    
    # Resize prediction mask to image size
    pred_mask_resized = cv2.resize(pred_mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    pred_mask_binary = (pred_mask_resized > 0.5).astype(np.uint8)
    
    # Calculate IoU
    intersection = np.logical_and(pred_mask_binary, gt_mask).sum()
    union = np.logical_or(pred_mask_binary, gt_mask).sum()
    
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
                
                # Segmentation polygon points (starting from index 5)
                poly_coords = []
                for i in range(5, len(parts), 2):
                    if i + 1 < len(parts):
                        x = float(parts[i]) * img_width
                        y = float(parts[i + 1]) * img_height
                        poly_coords.append((int(x), int(y)))
                
                labels.append((class_id, poly_coords))
    return labels


def visualize_segmentation_comparison(img_path, label_path, model_results, save_path):
    """Visualize ground truth polygon vs predicted segmentation mask"""
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    gt_img = img_rgb.copy()
    gt_labels = load_yolo_labels(label_path, w, h)
    for class_id, poly_coords in gt_labels:
        if poly_coords:
            pts = np.array(poly_coords, np.int32)
            cv2.polylines(gt_img, [pts], True, (0, 255, 0), 3)
    
    axes[1].imshow(gt_img)
    axes[1].set_title('Ground Truth (Green)')
    axes[1].axis('off')
    
    # Prediction with mask
    pred_img = img_rgb.copy()
    has_mask = False
    
    if model_results[0].masks is not None and len(model_results[0].masks.data) > 0:
        # Get best detection
        best_idx = model_results[0].boxes.conf.argmax()
        mask = model_results[0].masks.data[best_idx].cpu().numpy()
        
        # Resize mask to image size
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Create colored overlay
        mask_colored = np.zeros_like(img_rgb)
        mask_colored[:, :, 0] = mask_binary * 255  # Red channel
        
        # Overlay on image
        pred_img = cv2.addWeighted(pred_img, 0.7, mask_colored, 0.3, 0)
        
        # Also draw bounding box
        box = model_results[0].boxes[best_idx]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        conf = box.conf[0].cpu().numpy()
        cv2.putText(pred_img, f'{conf:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        has_mask = True
    
    axes[2].imshow(pred_img)
    axes[2].set_title('Prediction (Red Mask + Blue Box)' if has_mask else 'No Detection')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return len(gt_labels), has_mask


# Load the trained model
print("Loading model...")
model = YOLO('runs/segment/train_quick_test4/weights/best.pt')

# Test on 10 random images
print("Testing on random images with segmentation masks...\n")
test_indices = random.sample(range(5000), 10)

os.makedirs('segmentation_validation_output', exist_ok=True)

# Track statistics
total_images = len(test_indices)
detections_with_masks = 0
ious = []

for i, idx in enumerate(test_indices):
    img_path = f'src/dataset_test/train/images/{idx:06d}.jpg'
    label_path = f'src/dataset_test/train/labels/{idx:06d}.txt'
    
    print(f"Processing image {idx:06d}...", end='')
    
    # Run prediction
    results = model.predict(img_path, conf=0.25, save=False, verbose=False)
    
    # Save visualization
    save_path = f'segmentation_validation_output/{idx:06d}_segmentation.png'
    gt_count, has_pred = visualize_segmentation_comparison(img_path, label_path, results, save_path)
    
    # Calculate IoU if we have both prediction and ground truth
    if results[0].masks is not None and len(results[0].masks.data) > 0:
        detections_with_masks += 1
        
        # Get best mask
        best_idx = results[0].boxes.conf.argmax()
        mask = results[0].masks.data[best_idx].cpu().numpy()
        
        # Load ground truth
        gt_labels = load_yolo_labels(label_path, 640, 640)
        if gt_labels:
            # Calculate IoU with first ground truth (assuming single object)
            iou = calculate_mask_iou(mask, gt_labels[0][1], 640, 640)
            ious.append(iou)
            print(f" ✓ IoU: {iou:.3f}")
        else:
            print(f" ✓ No GT")
    else:
        print(f" ✗ No mask detected")

# Summary
print(f"\n📊 SEGMENTATION SUMMARY:")
print(f"✅ Images with mask detections: {detections_with_masks}/{total_images} ({detections_with_masks/total_images*100:.1f}%)")

if ious:
    print(f"\n🎯 MASK IoU METRICS:")
    print(f"📈 Average IoU: {np.mean(ious):.3f}")
    print(f"📈 Min IoU: {np.min(ious):.3f}")
    print(f"📈 Max IoU: {np.max(ious):.3f}")
    print(f"📈 IoU > 0.5: {sum(1 for iou in ious if iou > 0.5)}/{len(ious)} ({sum(1 for iou in ious if iou > 0.5)/len(ious)*100:.1f}%)")
    print(f"📈 IoU > 0.7: {sum(1 for iou in ious if iou > 0.7)}/{len(ious)} ({sum(1 for iou in ious if iou > 0.7)/len(ious)*100:.1f}%)")

print(f"\n💾 Visualizations saved to: segmentation_validation_output/")
print("Check the images to see if masks are being produced!")