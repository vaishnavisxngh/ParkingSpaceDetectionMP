# Module 4 — Deep Learning Inference
**Assigned to: Member 5 (ML Engineer)**

## Responsibility
This is the core AI engine. It loads a pre-trained deep learning model (YOLOv5 or ResNet18 via PyTorch) and runs inference on each parking slot's ROI to determine whether a vehicle is present.

## What this module does
- Loads YOLOv5s (via `torch.hub`) or ResNet18 from PyTorch
- Runs inference on individual slot crops or full frames
- Returns confidence scores and vehicle detection results per slot
- Optimized for speed (runs at 15+ FPS on GPU, acceptable on CPU)

## Files
| File | Description |
|------|-------------|
| `inference_engine.py` | Main inference class — loads model and runs predictions |
| `yolo_detector.py` | YOLOv5-specific detection wrapper |
| `resnet_classifier.py` | ResNet18 binary classifier wrapper |
| `model_config.yaml` | Model selection and hyperparameter config |

## Model Options
| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLOv5s | Fast ✅ | High ✅ | **Recommended** — full-frame vehicle detection |
| ResNet18 | Moderate | High | Per-slot binary classification (occupied/vacant) |
| Custom CNN | Fastest | Moderate | Lightweight alternative |

## Setup
```bash
pip install torch torchvision
# YOLOv5 downloads automatically on first run via torch.hub
```

## How to run
```bash
python inference_engine.py --model yolo --image data/sample_frame.jpg
```

## Integration
```python
from module4_deep_learning.inference_engine import InferenceEngine
engine = InferenceEngine(model_type="yolo")
results = engine.infer_slots(frame, slot_crops)
# returns: [{"slot_id": 0, "confidence": 0.92, "has_vehicle": True}, ...]
```
