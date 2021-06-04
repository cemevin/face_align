# Selfie Time Lapse in OpenCV
Face align Selfies in Open CV

This Python code take a series of selfies taken, aligns and centers them, and creates a time lapse video. <br/>
The code uses OpenCV, PIL and imageio for face detection and video generation.

![Example](example.gif)

## Installation
```
pip install opencv-contrib-python 
pip install imageio 
pip install --upgrade Pillow
```

## Usage
Put your images into `/images` folder. <br/>
Create an empty `/croppe`d folder. <br/>
run `python face_align.py` <br/>
The output movie will be exported as `output.avi`.
