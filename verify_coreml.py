import coremltools as ct
import numpy as np

# Load the exported model
model = ct.models.MLModel('runs/segment/train_full4/weights/best.mlpackage')

# Get the exact output names and shapes
print("\n=== CORE ML MODEL OUTPUTS ===")
for output in model.get_spec().description.output:
    print(f"\nOutput name: '{output.name}'")
    print(f"Type: {output.type.WhichOneof('Type')}")
    
    if hasattr(output.type, 'multiArrayType'):
        shape = output.type.multiArrayType.shape
        print(f"Shape: {list(shape)}") 