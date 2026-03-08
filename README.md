# Tracking-system
real time object tracker that can follow a specific object in a live webcam feed.
## idea

1. User selects object in first frame
2. Extract CNN feature vector from selection.
3. For each frame: Detect objects with YOLOv8.
4. Extract vectors for detections from YOLOV8.
5. Find best cosine similarity match (>0.5 threshold).
6. Draw box if matched; else show "Object Disappeared".

## Installation
```bash
pip install -r requirements.txt
