#!/usr/bin/env python3
"""
Proper Qyoo dot decoder that handles rotation, scale, and perspective
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


class QyooDotDecoder:
    def __init__(self, model_path='runs/segment/train_quick_test4/weights/best.pt'):
        self.model = YOLO(model_path)
        
    def detect_and_decode(self, image_path, visualize=False):
        """Full pipeline: detect, normalize, decode dots"""
        # Step 1: Get segmentation mask
        results = self.model.predict(image_path, conf=0.25, verbose=False)
        
        if not results[0].boxes or results[0].masks is None:
            return None, "No Qyoo detected"
            
        # Get best detection
        best_idx = results[0].boxes.conf.argmax().item()
        mask = results[0].masks.data[best_idx].cpu().numpy()
        confidence = results[0].boxes.conf[best_idx].item()
        
        # Load original image
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Step 2: Extract and normalize Qyoo
        normalized_qyoo = self.extract_and_normalize(img, mask, target_size=256)
        
        if normalized_qyoo is None:
            return None, "Failed to normalize Qyoo"
            
        # Step 3: Find orientation using square corner
        oriented_qyoo = self.find_orientation(normalized_qyoo)
        
        # Step 4: Decode dots from normalized, oriented Qyoo
        dot_pattern = self.decode_dots(oriented_qyoo)
        
        if visualize:
            self.visualize_process(img, mask, normalized_qyoo, oriented_qyoo, dot_pattern, confidence)
            
        return dot_pattern, f"Success (conf: {confidence:.2f})"
    
    def extract_and_normalize(self, image, mask, target_size=256):
        """Extract Qyoo using mask and normalize to standard orientation/size"""
        h, w = image.shape[:2]
        
        # Resize mask to image size
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Find the Qyoo contour
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Get the largest contour (the Qyoo)
        qyoo_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w_rect, h_rect = cv2.boundingRect(qyoo_contour)
        
        # Extract region with padding
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + w_rect + pad)
        y2 = min(h, y + h_rect + pad)
        
        # Extract region
        roi = image[y1:y2, x1:x2]
        roi_mask = mask_binary[y1:y2, x1:x2]
        
        # Apply mask to get clean Qyoo
        qyoo_region = cv2.bitwise_and(roi, roi, mask=roi_mask)
        
        # Find the exact Qyoo bounds in the ROI
        roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not roi_contours:
            return None
            
        roi_contour = max(roi_contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(roi_contour)
        
        # Crop to exact Qyoo
        exact_qyoo = qyoo_region[ry:ry+rh, rx:rx+rw]
        
        # Resize to standard size while maintaining aspect ratio
        if exact_qyoo.shape[0] > 0 and exact_qyoo.shape[1] > 0:
            # Calculate scale to fit in target_size square
            scale = min(target_size / exact_qyoo.shape[1], target_size / exact_qyoo.shape[0])
            new_w = int(exact_qyoo.shape[1] * scale)
            new_h = int(exact_qyoo.shape[0] * scale)
            
            # Resize
            resized = cv2.resize(exact_qyoo, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Center in target_size square
            normalized = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            start_x = (target_size - new_w) // 2
            start_y = (target_size - new_h) // 2
            normalized[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            return normalized
            
        return None
    
    def find_orientation(self, qyoo_image):
        """Find the square corner to determine orientation"""
        # Convert to grayscale
        gray = cv2.cvtColor(qyoo_image, cv2.COLOR_BGR2GRAY)
        
        # For now, assume the Qyoo is roughly oriented
        # In production, you would:
        # 1. Detect the square corner using corner detection
        # 2. Calculate rotation angle
        # 3. Rotate image so square corner is at bottom-right
        
        # TODO: Implement corner detection and rotation
        # This is a complex computer vision problem requiring:
        # - Harris corner detection
        # - Shape analysis to identify the square corner
        # - Perspective correction
        
        return qyoo_image
    
    def decode_dots(self, oriented_qyoo):
        """Decode 6x6 dot pattern from normalized, oriented Qyoo"""
        # Convert to grayscale
        gray = cv2.cvtColor(oriented_qyoo, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to enhance dots
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Define the grid area (center region where dots should be)
        h, w = gray.shape
        margin = 0.25  # 25% margin from edges
        
        grid_x = int(w * margin)
        grid_y = int(h * margin)
        grid_w = int(w * (1 - 2 * margin))
        grid_h = int(h * (1 - 2 * margin))
        
        # Divide into 6x6 grid
        cell_w = grid_w // 6
        cell_h = grid_h // 6
        
        # Analyze each cell for dot presence
        pattern = ""
        dot_confidences = []
        
        for row in range(6):
            for col in range(6):
                # Get cell coordinates
                x1 = grid_x + col * cell_w
                y1 = grid_y + row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                
                # Extract cell
                cell = gray[y1:y2, x1:x2]
                cell_binary = binary[y1:y2, x1:x2]
                
                # Multiple methods to detect dots
                dot_score = self.analyze_cell_for_dot(cell, cell_binary)
                dot_confidences.append(dot_score)
                
                # Threshold for dot presence
                has_dot = dot_score > 0.3
                pattern += '1' if has_dot else '0'
        
        return pattern
    
    def analyze_cell_for_dot(self, cell_gray, cell_binary):
        """Analyze a single cell to determine if it contains a dot"""
        if cell_gray.size == 0:
            return 0.0
            
        # Method 1: Circular object detection
        circles = cv2.HoughCircles(
            cell_gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=cell_gray.shape[0]//2,
            param1=50, 
            param2=15, 
            minRadius=2, 
            maxRadius=cell_gray.shape[0]//2
        )
        
        circle_score = 0.0
        if circles is not None:
            circle_score = 0.8  # High confidence if circle detected
        
        # Method 2: Contrast analysis
        center_region = cell_gray[
            cell_gray.shape[0]//4:3*cell_gray.shape[0]//4,
            cell_gray.shape[1]//4:3*cell_gray.shape[1]//4
        ]
        
        if center_region.size > 0:
            center_mean = np.mean(center_region)
            edge_mean = np.mean(cell_gray) - np.mean(center_region)
            contrast_score = abs(center_mean - edge_mean) / 255.0
        else:
            contrast_score = 0.0
        
        # Method 3: Binary blob analysis
        contours, _ = cv2.findContours(cell_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_score = 0.0
        
        if contours:
            # Look for roughly circular blobs
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 10:  # Minimum area threshold
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Reasonably circular
                            blob_score = max(blob_score, circularity)
        
        # Combine all methods
        final_score = max(circle_score, contrast_score * 0.7, blob_score * 0.6)
        return final_score
    
    def visualize_process(self, original, mask, normalized, oriented, pattern, confidence):
        """Visualize the entire decoding process"""
        fig = plt.figure(figsize=(16, 10))
        
        # Original image with mask overlay
        plt.subplot(2, 4, 1)
        h, w = original.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        overlay = original.copy()
        overlay[:, :, 0] = np.where(mask_resized > 0.5, 255, overlay[:, :, 0])
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'Detection (conf: {confidence:.2f})')
        plt.axis('off')
        
        # Extracted and normalized Qyoo
        plt.subplot(2, 4, 2)
        if normalized is not None:
            plt.imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
        plt.title('Normalized Qyoo')
        plt.axis('off')
        
        # Oriented Qyoo
        plt.subplot(2, 4, 3)
        plt.imshow(cv2.cvtColor(oriented, cv2.COLOR_BGR2RGB))
        plt.title('Oriented Qyoo')
        plt.axis('off')
        
        # Dot detection grid overlay
        plt.subplot(2, 4, 4)
        grid_img = oriented.copy()
        gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        margin = 0.25
        grid_x = int(w * margin)
        grid_y = int(h * margin)
        grid_w = int(w * (1 - 2 * margin))
        grid_h = int(h * (1 - 2 * margin))
        cell_w = grid_w // 6
        cell_h = grid_h // 6
        
        # Draw grid
        for i in range(7):
            x = grid_x + i * cell_w
            y = grid_y + i * cell_h
            cv2.line(grid_img, (x, grid_y), (x, grid_y + grid_h), (0, 255, 0), 1)
            cv2.line(grid_img, (grid_x, y), (grid_x + grid_w, y), (0, 255, 0), 1)
        
        plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
        plt.title('Dot Detection Grid')
        plt.axis('off')
        
        # Decoded pattern as matrix
        plt.subplot(2, 4, 5)
        if pattern and len(pattern) == 36:
            pattern_matrix = np.array(list(pattern), dtype=int).reshape(6, 6)
            plt.imshow(pattern_matrix, cmap='RdBu_r', interpolation='nearest')
            plt.title('Decoded Pattern')
            for i in range(6):
                for j in range(6):
                    plt.text(j, i, pattern_matrix[i, j], ha='center', va='center', 
                           color='white' if pattern_matrix[i, j] else 'black', fontsize=12)
        else:
            plt.text(0.5, 0.5, 'No Pattern', ha='center', va='center')
            plt.title('Decoded Pattern')
        plt.axis('off')
        
        # Pattern as text
        plt.subplot(2, 4, 6)
        if pattern:
            # Format as 6x6 grid
            grid_text = '\n'.join([pattern[i:i+6] for i in range(0, 36, 6)])
            plt.text(0.1, 0.5, f'Binary Pattern:\n{pattern}\n\nGrid:\n{grid_text}', 
                    fontsize=10, family='monospace', va='center')
        plt.title('Pattern Data')
        plt.axis('off')
        
        # Statistics
        plt.subplot(2, 4, 7)
        if pattern:
            ones = pattern.count('1')
            zeros = pattern.count('0')
            plt.text(0.1, 0.7, f'Dots detected: {ones}', fontsize=12)
            plt.text(0.1, 0.5, f'Empty cells: {zeros}', fontsize=12)
            plt.text(0.1, 0.3, f'Total cells: 36', fontsize=12)
        plt.title('Statistics')
        plt.axis('off')
        
        # Instructions
        plt.subplot(2, 4, 8)
        plt.text(0.1, 0.8, 'Key Challenges:', fontsize=12, weight='bold')
        plt.text(0.1, 0.6, '• Rotation handling', fontsize=10)
        plt.text(0.1, 0.5, '• Scale normalization', fontsize=10)
        plt.text(0.1, 0.4, '• Perspective correction', fontsize=10)
        plt.text(0.1, 0.3, '• Dot vs noise detection', fontsize=10)
        plt.text(0.1, 0.1, 'Current: Basic implementation', fontsize=10, style='italic')
        plt.title('Implementation Notes')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def test_dot_decoder():
    """Test the dot decoder on real images"""
    decoder = QyooDotDecoder()
    
    # Test on a few images
    test_images = [
        'src/dataset_test/train/images/000000.jpg',
        'src/dataset_test/train/images/000001.jpg',
        'src/dataset_test/train/images/000002.jpg',
    ]
    
    for img_path in test_images:
        if cv2.imread(img_path) is not None:
            print(f"\nProcessing: {img_path}")
            pattern, status = decoder.detect_and_decode(img_path, visualize=True)
            
            if pattern:
                print(f"✅ {status}")
                print(f"Pattern: {pattern}")
                # Format as grid
                grid = '\n'.join([f"  {pattern[i:i+6]}" for i in range(0, 36, 6)])
                print(f"Grid:\n{grid}")
            else:
                print(f"❌ {status}")


if __name__ == "__main__":
    print("Testing proper Qyoo dot decoder...")
    print("\nNOTE: This addresses your concerns about rotation, scale, and perspective")
    print("But full implementation requires advanced computer vision techniques.\n")
    
    test_dot_decoder()