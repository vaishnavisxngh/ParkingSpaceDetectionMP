# Module 2 — Image Preprocessing
**Assigned to: Member 2 (Preprocessing Specialist)**

## Responsibility
Receives raw frames from Module 1 and prepares them for the deep learning model. This includes resizing, normalizing pixel values, and optionally applying image enhancement to handle poor lighting or noise.

## What this module does
- Resizes frames to a standard input dimension (e.g., 640×640 for YOLO or 224×224 for ResNet)
- Normalizes pixel values to [0, 1] or model-specific range
- Optional: contrast adjustment (CLAHE), noise reduction (Gaussian blur)
- Outputs processed tensors ready for deep learning inference

## Files
| File | Description |
|------|-------------|
| `preprocessor.py` | Main preprocessing pipeline class |
| `enhancer.py` | Optional image enhancement utilities |
| `transforms.py` | PyTorch-compatible transform definitions |

## How to run (standalone test)
```bash
python preprocessor.py --input data/sample_frame.jpg --model_type yolo
```

## Integration
Called by the main pipeline after Module 1 yields a frame:
```python
from module2_preprocessing.preprocessor import Preprocessor
prep = Preprocessor(model_type="yolo")
tensor = prep.process(raw_frame)
```
