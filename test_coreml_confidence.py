import coremltools as ct
import numpy as np
from PIL import Image

# Load model
model_path = '/Users/deankaplan/dev/work/QyooGenerate-Python/iOS/QyooDemo/best.mlpackage'
ml_model = ct.models.MLModel(model_path)

# Test with sample image
real_image_path = '/Users/deankaplan/dev/work/QyooGenerate-Python/samples/000004.jpg'
real_image = Image.open(real_image_path)
real_image_resized = real_image.resize((512, 512))

# Get prediction
prediction = ml_model.predict({'image': real_image_resized})

# Extract confidence
if 'var_1052' in prediction:
    detections = prediction['var_1052']
    print(f"Detections shape: {detections.shape}")
    
    confidences = detections[0, 4, :]  # confidence is index 4
    max_conf = np.max(confidences)
    print(f"🎯 MAX CONFIDENCE: {max_conf:.8f} ({max_conf*100:.6f}%)")
    
    # Top 5
    top_indices = np.argsort(confidences)[-5:][::-1]
    print("Top 5 confidences:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {confidences[idx]:.8f}") 

# Test with YOUR Qyoo images
test_images = [
    '/Users/deankaplan/dev/work/QyooGenerate-Python/iOS/QyooTests/test_image_1.jpg',
    '/Users/deankaplan/dev/work/QyooGenerate-Python/iOS/QyooTests/test_image_2.jpg'
]

for img_path in test_images:
    try:
        print(f"\n🔍 Testing: {img_path}")
        image = Image.open(img_path)
        image_resized = image.resize((512, 512))
        
        prediction = ml_model.predict({'image': image_resized})
        
        if 'var_1052' in prediction:
            detections = prediction['var_1052']
            confidences = detections[0, 4, :]
            max_conf = np.max(confidences)
            print(f"🎯 MAX CONFIDENCE: {max_conf:.8f} ({max_conf*100:.6f}%)")
    except Exception as e:
        print(f"Error: {e}")
