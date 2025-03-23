# Blackjack Card Detection and Tracking System

This repository contains an advanced card detection and tracking system for blackjack games using YOLO-based object detection and multi-object tracking. The system can maintain consistent card identities across video frames, even when cards are temporarily occluded or overlap with each other.

## Features

- **Real-time card detection** using YOLOv8

https://github.com/user-attachments/assets/2a1f8b0f-c364-4158-83e4-0f0f5ff53ae3

 model trained on playing cards
- **Consistent card tracking** with DeepSORT and/or custom tracking algorithms
- **Re-identification (Re-ID)** to maintain card identity even after temporary disappearance
- **Occlusion handling** using Mask R-CNN (in GPU version) or motion prediction (in CPU version)
- **GPU and CPU optimized versions** for different hardware capabilities

## Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU (for GPU version) or decent CPU (for CPU version)

## Installation

### GPU Version

```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### CPU-Only Version

```bash
# Create and activate a virtual environment
python -m venv venv_cpu
.\venv_cpu\Scripts\activate

# Install dependencies (CPU optimized)
pip install -r requirements_cpu.txt
```

## Usage

### Live Detection with GPU

```bash
python live.py
```

### Live Detection with CPU-Optimized Code

```bash
python live_cpu.py
```

### View Multiple Camera Feeds

```bash
python video.py
```

## Tuning Parameters

You can modify various parameters to optimize the system for your specific use case. Here's how to modify the main parameters:

### 1. Tracker Parameters

#### DeepSORT Parameters (in `tracker.py` or `tracker_cpu.py`)

- **max_age**: Maximum frames to keep a lost track alive (default: 30)
  - Higher values: Tracks persist longer when temporarily occluded
  - Lower values: Tracks are removed more quickly, faster but less memory of cards

- **n_init**: Number of consecutive frames a detection should appear before generating a track (default: 3)
  - Higher values: More stable tracks, less false positives
  - Lower values: Faster tracking initialization, might include more false detections

- **max_cosine_distance**: Threshold for appearance similarity (default: 0.5 for GPU, 0.7 for CPU)
  - Higher values: More permissive matching (different cards might be considered the same)
  - Lower values: Stricter matching (same card might be considered different)

- **max_iou_distance**: Threshold for spatial proximity (default: 0.7 for GPU, 0.8 for CPU)
  - Higher values: Better for fast moving objects
  - Lower values: Better for slow moving objects

Example modification:
```python
# In tracker.py or tracker_cpu.py
self.tracker = DeepSort(
    max_age=45,          # Increasing from 30 to 45
    n_init=2,            # Decreasing from 3 to 2
    max_cosine_distance=0.4,  # Decreasing from 0.5 to 0.4 for more strict appearance matching
    max_iou_distance=0.6,     # Decreasing from 0.7 to 0.6
    # ...other parameters...
)
```

### 2. YOLO Detection Parameters (in `live.py` or `live_cpu.py`)

- **conf**: Confidence threshold for detection (default: 0.25 for GPU, 0.3 for CPU)
  - Higher values: Fewer detections but more confident
  - Lower values: More detections but possibly more false positives

- **iou**: IoU threshold for Non-Maximum Suppression (default: 0.45 for GPU, 0.5 for CPU)
  - Higher values: More overlapping boxes
  - Lower values: Fewer overlapping boxes

Example modification:
```python
# In process_frame function of live.py or live_cpu.py
results = model(img_proc, conf=0.35, iou=0.4, verbose=False)
```

### 3. Re-ID Parameters (in `tracker.py` or `tracker_cpu.py`)

- **similarity_threshold**: Threshold for considering Re-ID embeddings similar (default: 0.75 for GPU, 0.6 for CPU)
  - Higher values: Stricter Re-ID matching
  - Lower values: More permissive Re-ID matching

Example modification:
```python
# In CardReIDModel or LightweightReIDModel class
self.similarity_threshold = 0.65  # Adjusting threshold
```

### 4. Performance Parameters

#### CPU Version (in `live_cpu.py`)

- **frame_skip**: Process every Nth frame (default: 3)
  - Higher values: Faster processing, less CPU usage, lower detection rate
  - Lower values: Better detection rate, but higher CPU usage

- **scale**: Resize factor for input images (default: 480 / max dimension)
  - Higher values: Better detection accuracy but slower processing
  - Lower values: Faster processing but reduced accuracy

Example modification:
```python
# In live_cpu.py
frame_skip = 5  # Process every 6th frame instead of every 4th
scale = 360 / max(height, width)  # Smaller size for faster processing
```

## Evaluating Changes

When making parameter changes, you can evaluate their effect by:

1. Looking at the processed frames saved in `detections_1/` (GPU) or `detections_cpu/` directories
2. Checking the console output for performance metrics
3. Observing the tracking quality in the video display

## Troubleshooting

### Common Issues

- **Memory errors**: Reduce batch size or image resolution
- **Slow processing**: Increase frame_skip, reduce resolution, or disable certain features
- **Tracking errors**: Adjust tracking parameters like max_age, n_init
- **Detection misses**: Reduce conf threshold, adjust iou threshold

## License

This project is licensed under the MIT License - see the LICENSE file for details.# card-detection
