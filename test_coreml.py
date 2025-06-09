# Test CoreML model conversion and inference
import torch
import coremltools as ct
import numpy as np
from PIL import Image
import os


def test_model():
    try:
        # 1. Create a simple PyTorch model for testing
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 128 * 128, 10)
        )
        model.eval()

        # 2. Export to CoreML
        print("Exporting to CoreML...")
        example_input = torch.randn(1, 3, 512, 512)
        traced_model = torch.jit.trace(model, example_input)
        ml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="image", shape=example_input.shape)]
        )
        ml_model.save("simple_model.mlpackage")
        print("CoreML model saved as simple_model.mlpackage")

        # 3. Test CoreML model
        ml_model = ct.models.MLModel("simple_model.mlpackage")
        img_array = np.random.rand(1, 3, 512, 512).astype(np.float32)
        prediction = ml_model.predict({"image": img_array})
        print("CoreML outputs:", prediction.keys())

    except Exception as e:
        print(f"Error during testing: {str(e)}")


if __name__ == "__main__":
    test_model() 