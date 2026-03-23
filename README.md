# Automatic-Measurement-of-Rice-Tiller-Angle-from-UAV-Images

This repository contains code and data for automatic measurement of rice tiller angle from UAV Images.

Folder "ultralytics" contains package for YOLO models (https://www.ultralytics.com/).
Folder "TillersAngle" contains code and data for tiller angle estimation.
File "YOLO11P312.yaml" defines Anaconda environment for tiller agle estimation.

"RiceTillerAngles.yaml": Settings for tiller angle model training
"yolo11n-pose.pt": Initial trained model for pose detection from Roboflow (https://roboflow.com/).
"TillerAnglev5.pt": Final trained model for keypont detection.
"SegmentationTransformation.py": Support Python code for segmentation and perspective transformaiton.
"YOLOv11 - Tiller Angle.py": Code for model taining and evaluation.

You will nened to modify data directories accordingly to run the program on your computer.

*Due to file size restrictions, only a subset of the full image data is uploaded into this repository (The Compressed Zip File).

