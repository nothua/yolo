# Object Detection with YOLOv5 and Audio Description

## Overview

This script combines real-time object detection using YOLOv5 and audio description generation for the detected objects. It can be used with both live webcam feed and pre-recorded video files.

## Requirements

*Weights auto-downloads when a script is executed*

- torch
- numpy
- opencv-python (cv2)
- pydub
- gtts
- ffmpeg

Install the required libraries using the following command:
```bash
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
pip install ffmpeg gtts pydub 
```

## Usage

### Real-time Object Detection with Webcam Feed

Run the following script to perform real-time object detection using your webcam:
```bash
python real_time_detection.py
```
The script captures 300 frames, performs object detection, and generates audio descriptions for the detected objects.

### Object Detection in a Video File
- Place your video file (e.g., "in.mp4") in the same directory as the script.
- Run the following script to perform object detection on the video file:
```bash
python video_detection.py
```
The script annotates the video frames with bounding boxes around detected objects and generates an audio description for each frame.
- The final output is saved as "output.mp4" in the same directory.

