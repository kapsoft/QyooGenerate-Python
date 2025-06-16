# modal_working.py
import modal

app = modal.App("qyoo-final")

# Add ALL dependencies  
image = (
    modal.Image.debian_slim()
    .apt_install("wget", "curl", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("opencv-python-headless")  # Use headless version first
    .pip_install("ultralytics")
)

volume = modal.Volume.from_name("qyoo-results-v2", create_if_missing=True)

@app.function(
    image=image,
    gpu="a10g",
    timeout=3600,
    volumes={"/results": volume}
)
def train_yolo(dataset_url: str):
    """Actually train the model"""
    import subprocess
    import os
    
    print(f"🚀 Training with: {dataset_url[:50]}...")
    
    # Simple commands that WORK
    cmds = [
        f"cd /tmp && curl -L '{dataset_url}' -o dataset.tar.gz",
        "cd /tmp && tar -xzf dataset.tar.gz",
        "cd /tmp && ls -la dataset/train/",
        """cd /tmp && echo 'path: /tmp/dataset
train: train/images  
val: train/images
nc: 1
names: ["qyoo"]' > data.yaml""",
        "cd /tmp && yolo train model=yolov8s-seg.pt data=data.yaml epochs=25 batch=8 imgsz=512 device=0 project=/results name=run1"
    ]
    
    for cmd in cmds:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise Exception(f"Command failed: {cmd}")
        print(result.stdout)
    
    return "✅ Training complete!"

@app.local_entrypoint()
def main():
    # Read URL WITHOUT dotenv
    with open("dataset_url.txt", "r") as f:
        url = f.read().strip()
    
    # Run training
    result = train_yolo.remote(url)
    print(result)
    
    print("\n📥 Get your model:")
    print("modal volume get qyoo-results-v2 /run1/weights/best.pt ./my_trained_model.pt")

if __name__ == "__main__":
    # Modal will handle this
    pass