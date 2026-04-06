# Dynamic Athlete Tracking (YOLOv11n + SAM 2)

![Demo](demo.gif)

## Overview
This project demonstrates an automated visual analytics pipeline designed to dynamically isolate and track specific main characters in real-world sports footage. By integrating **YOLOv11n** for robust object detection and the **Segment Anything Model (SAM 2)** for precise instance segmentation, the system effectively tracks targeted athletes while filtering out background noise.

## Key Features & Engineering Highlights

* **Target-Specific Isolation (`detect_top_one.py`):** Automatically identifies and locks onto the primary subject of interest, generating highly accurate segmentation masks frame by frame.
* **Scalable Multi-Target Tracking (`detect_top_n.py`):** An extended implementation that allows users to dynamically input the number of top-confidence targets they wish to track simultaneously.
* **Dynamic Bounding Box Padding:** To solve the common issue of limbs being cropped out during fast-moving sports actions, the algorithm dynamically applies a 10% padding to the YOLO bounding boxes before passing them to SAM.
* **Hardware Auto-Detection:** The pipeline includes built-in environment checks using `torch.cuda.is_available()` to automatically utilize GPU (CUDA) acceleration when available, gracefully falling back to CPU computation otherwise.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/PL-32003/Dynamic-Athlete-Tracking-YOLO-SAM.git
   cd Dynamic-Athlete-Tracking-YOLO-SAM
   ```
2.Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained weights (yolo11n.pt and sam2_b.pt) and place them in the root directory.

## Usage
To track a single main character (Demo Version):

```Bash
python detect_top_one.py
```
To track a custom number of top targets:

```Bash
python detect_top_n.py
```
(The script will prompt you to input the desired number of targets to track).
