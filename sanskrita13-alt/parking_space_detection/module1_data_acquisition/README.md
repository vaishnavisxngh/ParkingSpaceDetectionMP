# Module 1 — Data Acquisition
**Assigned to: Member 1 (Data Engineer)**

## Responsibility
This module is the entry point of the entire pipeline. It handles all data sourcing — whether from a live camera feed or from a pre-recorded video/image dataset. It extracts frames at a configurable rate and feeds them into the preprocessing pipeline.

## What this module does
- Accepts live webcam/IP camera feeds OR pre-recorded video files
- Supports image datasets in JPEG/PNG format (e.g., PKLot dataset)
- Extracts frames from video at a configurable FPS
- Outputs raw frames to a queue/directory for the next module

## Files
| File | Description |
|------|-------------|
| `data_loader.py` | Main data acquisition class |
| `camera_stream.py` | Handles live camera/webcam input |
| `video_reader.py` | Handles pre-recorded video file input |
| `dataset_loader.py` | Loads image datasets from a folder |
| `config.yaml` | Configuration for source type, FPS, paths |

## How to run
```bash
python data_loader.py --source video --path data/sample_lot.mp4 --fps 15
python data_loader.py --source camera --device 0
python data_loader.py --source dataset --path data/PKLot/
```

## Output
Raw frames (NumPy arrays) passed via shared queue to Module 2 (Preprocessing).
