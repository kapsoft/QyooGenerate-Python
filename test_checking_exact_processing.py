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

# After loading the image...
img_array = np.array(image_resized)
print(f"\nArray stats:")
print(f"  Shape: {img_array.shape}")
print(f"  Data type: {img_array.dtype}")
print(f"  Min/Max: {img_array.min()} / {img_array.max()}")



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
spec = ml_model.get_spec()
print("\nFull model description:")
print(f"Model version: {spec.specificationVersion}")

# Check preprocessing
input_spec = spec.description.input[0].type.imageType
print(f"\nImage input details:")
print(f"  Color space: {input_spec.colorSpace}")
print(f"  Size: {input_spec.width}x{input_spec.height}")

# Check for preprocessing layers
if len(spec.neuralNetwork.layers) > 0:
    print(f"\nFirst few layers:")
    for i, layer in enumerate(spec.neuralNetwork.layers[:3]):
        print(f"  Layer {i}: {layer.WhichOneof('layer')}")
        
# Check for preprocessing in the model
if hasattr(spec, 'neuralNetwork') and hasattr(spec.neuralNetwork, 'preprocessing'):
    print(f"\nPreprocessing found: {spec.neuralNetwork.preprocessing}")