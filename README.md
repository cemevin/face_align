# Selfie Time Lapse in OpenCV
Face align Selfies in Open CV

This Python code take a series of selfies taken, aligns and centers them, and creates a time lapse video. <br/>
The code uses OpenCV, PIL and imageio for face detection and video generation.

![Example](example.gif)

## Installation
pip install opencv-contrib-python <br/>
pip install imageio <br/>
pip install --upgrade Pillow <br/>

Add pictures to ./images <br/>
run face_align.py <br/>

## Usage
Put your images in order to /images folder. Empty /cropped folder but don't remove the folder itself, as it's used as cache by the program. The output movie will be exported as output.avi/