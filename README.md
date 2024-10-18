# Unattended Luggage Detection
This project aims to detect unattended luggage in airports and other public places.
The project is implemented using YOLOv8 object and MiDaS for depth estimation. 

## Requirements
- Python 3.10 or later
- Torch 2.4.0
- Deepsort
- YoloV8

## Installation
1. Clone the repository
2. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
3. If you have Nvidia GPU, make sure to install [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and a supported [Torch](https://pytorch.org/get-started/locally/) version

## Usage
Run the `main.py` file to start the application. Make sure to modify the `main.py` file to give it a path to a video to process.
You can do this by modifiying the following line:
```python
video_path = "path/to/video.mp4"
```

# My Model
This model is a custom YOLO and DeepSORT tracker model for object tracking and unattended luggage detection.

## Files Included
- `yolov8x_custom_weights.pt`: YOLOv8 custom weights.
- `midas_weights.pth`: Weights for the MiDaS depth estimation model.

## Usage
You can use this model by loading the weights and running the provided script.

```python
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the model
model = YOLO('yolov8x_custom_weights.pt')

