# Baseball Player Main Character Tracking (YOLOv11n + SAM 2)

![Demo](demo.gif)

## Overview
This project demonstrates an automated visual analytics pipeline designed to dynamically isolate and track a specific main character in real-world, dynamic sports footage. By integrating **YOLOv11n** for robust object detection and the **Segment Anything Model (SAM 2)** for precise instance segmentation, the system effectively tracks targeted athletes while filtering out background noise.

## Key Features & Engineering Highlights

* **Target-Specific Isolation (`detect_top_one.py`):** Automatically identifies and locks onto the primary subject of interest, generating highly accurate segmentation masks frame by frame.
* **Scalable Multi-Target Tracking (`detect_top_n.py`):** An extended implementation that allows users to dynamically input the number of top-confidence targets ($N$) they wish to track simultaneously.
* **Dynamic Bounding Box Padding:** To solve the common issue of limbs being cropped out during fast-moving sports actions (e.g., running or swinging a bat), a custom algorithm dynamically applies a 10% padding to the YOLO bounding boxes before passing them to SAM.
* **Hardware Auto-Detection:** The pipeline includes built-in environment checks to automatically utilize GPU (CUDA) acceleration when available, gracefully falling back to CPU computation otherwise.

## Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/baseball-player-video-main-character-tracking-yolov11n-sam.git](https://github.com/YourUsername/baseball-player-video-main-character-tracking-yolov11n-sam.git)
   cd baseball-player-video-main-character-tracking-yolov11n-sam
