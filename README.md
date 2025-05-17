# QyooGenerate-Python

## TL;DR

I’m building an iOS app that uses the camera to detect a custom “Qyoo” barcode symbol, regardless of rotation, and extract a clean mask of its shape. The end goal is to crop out that region and decode an internal dot pattern unique to each barcode. To do this, I’ve trained a YOLOv8 segmentation model on synthetic images with annotated masks and converted it to Core ML format. However, I’m stuck integrating the model into Vision: the expected pixel mask output isn’t working properly, and I need a reliable, standards-compliant way to get the masked region into a usable format (e.g. CGImage) for further analysis.

## Problem:

I need an iOS app that can scan a barcode like ![photo of sample qyoo flyer](samples/1.jpg?raw=true) (I'd like it to be at any orientation - it can be rotated at any degree). I have currently trained a model against this with multiple angles.

I would like the ML model to identify the mask around the desired shape and give me a CGImage or whatever to work with that is masked, something like ![masked version of sample qyoo flyer](samples/2.jpg?raw=true).

I then need to crop the shape and read the dot pattern, something like ![photo of qyoo with grid](samples/3.jpg?raw=true).

I have taken the following steps to do this, but got stuck:

Steps:

- `src/generate_qyoo_synthetic.py`: I have a script that generated 50k sample images (as if they were coming from a camera) with label text describing the mask for the shape. (I also generated an additional 5k as validation images)
- `src/validate_label.py`: I have a script to help test that the generated images and mask are correct by drawing the outline of the mask on the image and saving it to confirm that it's correct.
- `src/run_qyoo_train.zsh`: I have generated a YOLO model based on this data, ran it for 25 epochs, and am trying to work with it
- `iOS/QyooDemo.xcodeproj`: I have a sample iOS app to take that model and detect the qyoo shape, but it's not working because either the model is not giving me a mask properly, or there's something I'm doing wrong - I need help with either getting a model that does what it needs, or integrating it so that it will properly show the mask around the detected qyoo shape.

# Further details:

### GENERATING SAMPLE IMAGES

You must edit dataset.yaml and fix the main path for your system.

to run:

`python3 generate_qyoo_synthetic.py --count 5 --bg-dir ./backgrounds`

This will generate 5 images in dataset/train

You will need 50,000 images for training, and 5,000 for validation (I generated 5000 in the dataset/val folder).

You can generate a few and then run this to confirm they are working:

`python3 validate_label.py 0`

(to validate the 000000.jpg file, if you look in /dataset/test you will see the merge of the label text values with the training image)



### TRAINING ML MODEL

```
yolo task=segment mode=train \
     model=yolov8n-seg.pt \
     data=dataset.yaml \
     imgsz=512 batch=64 epochs=120 \
     device=mps workers=0 cache=disk \
     nms=True \
     save=True save_period=5 patience=20 \
     name=train_full
```

or use the `src/run_qyoo_train.zsh` to continue training if it crashes (do at least one epoch of the above first, and then run this script).

### GENERATING COREML MODEL

```
# still inside your project folder
python3.11 -m venv .coremlenv
source .coremlenv/bin/activate
pip install --upgrade pip

# install compatible toolchain
pip install torch==2.2.1
pip install coremltools==8.2
pip install ultralytics==8.3.124

# CoreML for iOS
yolo export model=../runs/detect/train/weights/best.pt format=coreml imgsz=512 nms=True

```