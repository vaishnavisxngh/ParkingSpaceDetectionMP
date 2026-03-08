# 🅿️ Parking Space Detection System
**Mini Project | Computer Vision + Deep Learning | PyTorch**

---

## Overview
An end-to-end parking occupancy detection system using YOLOv5 + OpenCV. The system processes video feeds or image datasets, detects which parking slots are occupied or vacant in real-time, and displays color-coded overlays with occupancy logs.

---

## Project Structure & Team

```
parking_space_detection/
│
├── module1_data_acquisition/       ← Member 1 (Data Engineer)
│   └── data_loader.py              Camera, video, dataset ingestion
│
├── module2_preprocessing/          ← Member 2 (Preprocessing Specialist)
│   └── preprocessor.py             Resize, normalize, enhance frames
│
├── module3_slot_mapping/           ← Member 3 (Mapping & Config)
│   ├── slot_mapper.py              Interactive ROI definition tool
│   └── slots_config.json           Saved parking slot coordinates
│
├── module4_deep_learning/          ← Member 5 (ML Engineer)
│   └── inference_engine.py         YOLOv5 / ResNet18 inference
│
├── module5_classification/         ← Member 6 (Classification Logic)
│   └── classifier.py               Threshold + temporal smoothing
│
├── module6_visualization_logging/  ← Member 3 + Member 6 (Shared)
│   ├── visualizer.py               Color-coded overlay drawing
│   └── logger.py                   CSV/JSON occupancy logger
│
├── main.py                         ← Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Define parking slots (one-time setup)
```bash
python main.py --define_slots --ref_image data/reference_frame.jpg
# Click and drag to draw rectangles over each parking slot
# Press 's' to save a slot, 'q' when done
```

### 3. Run on a video file
```bash
python main.py --source video --path data/parking_lot.mp4
```

### 4. Run on live camera
```bash
python main.py --source camera --device 0
```

### 5. Run on image dataset (PKLot, etc.)
```bash
python main.py --source dataset --path data/PKLot/
```

---

## Controls (while running)
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current annotated frame |

---

## Pipeline Flow

```
[Camera/Video/Dataset]
        ↓
[Module 1: Data Acquisition]     ← Extracts frames at target FPS
        ↓
[Module 2: Preprocessing]        ← Resize, normalize, enhance
        ↓
[Module 3: Slot Mapping]         ← Crop each parking slot ROI
        ↓
[Module 4: Deep Learning]        ← YOLOv5 vehicle detection
        ↓
[Module 5: Classification]       ← Threshold + temporal smoothing
        ↓
[Module 6: Visualization]        ← Green/Red overlays + HUD
        ↓
[Module 6: Logging]              ← CSV/JSON occupancy logs
```

---

## Model Options
| Model | Recommended For | Notes |
|-------|-----------------|-------|
| **YOLOv5s** ✅ | Real-time video | Auto-downloaded via torch.hub, no training needed |
| ResNet18 | Accurate classification | Needs fine-tuning on PKLot dataset for best results |

Default is **YOLOv5s** — best for a mini project demo.

---

## Output
- **Live window** with green (vacant) / red (occupied) slot overlays
- **logs/occupancy.csv** — timestamped occupancy log
- **logs/occupancy.json** — structured JSON log

---

## Dataset
Recommended: [PKLot Dataset](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
- 12,000+ images of parking lots under varying conditions
- Organized by weather: sunny, cloudy, rainy

---

## Requirements
- Python 3.9+
- PyTorch 2.0+ (CPU or CUDA)
- OpenCV
- Internet connection on first run (for YOLOv5 download)
