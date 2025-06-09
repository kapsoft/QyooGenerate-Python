import coremltools as ct
import numpy as np
from PIL import Image

# Load your test image
img_path = '/Users/deankaplan/dev/work/QyooGenerate-Python/iOS/QyooTests/test_image_2.jpg'
image = Image.open(img_path)

print(f"Original image:")
print(f"  Size: {image.size}")
print(f"  Mode: {image.mode}")

# Resize (like CoreML expects)
image_resized = image.resize((512, 512))
print(f"\nAfter resize:")
print(f"  Size: {image_resized.size}")
print(f"  Mode: {image_resized.mode}")

# Convert to array to see pixel values
img_array = np.array(image_resized)
print(f"\nArray properties:")
print(f"  Shape: {img_array.shape}")
print(f"  Dtype: {img_array.dtype}")
print(f"  Value range: {img_array.min()} to {img_array.max()}")
print(f"  Sample pixels (top-left corner): {img_array[0, 0]}")
print(f"  Sample pixels (center): {img_array[256, 256]}")

# Test with CoreML
ml_model = ct.models.MLModel('/Users/deankaplan/dev/work/QyooGenerate-Python/iOS/QyooDemo/best.mlpackage')
prediction = ml_model.predict({'image': image_resized})
confidences = prediction['var_1052'][0, 4, :]
max_conf = np.max(confidences)
print(f"\n🎯 Python confidence: {max_conf:.8f} ({max_conf*100:.6f}%)") 