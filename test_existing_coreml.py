# test_existing_coreml.py
import coremltools as ct
import numpy as np
from PIL import Image

# Check if the model was already exported
model_path = '/Users/deankaplan/dev/work/QyooGenerate-Python/iOS/QyooDemo/best.mlpackage'

try:
    # Load existing CoreML model
    ml_model = ct.models.MLModel(model_path)
    print("✅ CoreML model loaded successfully!")
    
    # Check the model specs
    spec = ml_model.get_spec()
    print("\n📊 Model Information:")
    print(f"Inputs: {[i.name for i in spec.description.input]}")
    print(f"Outputs: {[i.name for i in spec.description.output]}")
    
    # Test with dummy input
    dummy_input = np.random.rand(1, 3, 512, 512).astype(np.float32)
    dummy_image = Image.fromarray((dummy_input[0].transpose(1, 2, 0) * 255).astype(np.uint8))
    
    try:
        prediction = ml_model.predict({'image': dummy_image})
        print("\n✅ Model inference successful with dummy input!")
        print(f"Output keys: {list(prediction.keys())}")
    except Exception as e:
        print(f"\n❌ Inference failed with dummy input: {e}")
        
    # Test with a real image file
    real_image_path = '/Users/deankaplan/dev/work/QyooGenerate-Python/samples/000004.jpg'
    print(f"\n🔍 Attempting to load real image from: {real_image_path}")
    try:
        real_image = Image.open(real_image_path)
        print("✅ Real image loaded successfully!")
        # Resize to model's expected input size
        real_image_resized = real_image.resize((512, 512))
        print("🔍 Resized real image to 512x512 for inference...")
        print("🔍 Attempting inference with real image...")
        prediction = ml_model.predict({'image': real_image_resized})
        print("\n✅ Model inference successful with real image!")
        print(f"Output keys: {list(prediction.keys())}")
        print(f"Output values: {prediction}")
    except Exception as e:
        print(f"\n❌ Error during real image test: {e}")
        
except FileNotFoundError:
    print(f"❌ No CoreML model found at {model_path}")
    print("Need to export first!")
except Exception as e:
    print(f"❌ Error loading model: {e}") 