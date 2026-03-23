# Automatic-Measurement-of-Rice-Tiller-Angle-from-UAV-Images

This repository contains code and data for automatic measurement of rice tiller angle from UAV Images.

Folder "ultralytics" contains package for YOLO models (https://www.ultralytics.com/).
Folder "TillersAngle" contains code and data for tiller angle estimation.

"RiceTillerAngles.yaml": Anaconda environment for tiller agle estimation.
"yolo11n-pose.pt": Initial trained model for pose detection from Roboflow (https://roboflow.com/).
"TillerAnglev5.pt": Final trained model for keypont detection.
"SegmentationTransformation.py": Support Python code for segmentation and perspective transformaiton.
"YOLOv11 - Tiller Angle.py": Code for model taining and evaluation.

You will nened to modify data directories accordingly to run the program on your computer.
