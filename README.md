# Parking Space Detection System
### Mini Project — School of Computer Engineering, KIIT University

A real-time parking occupancy detection system using computer vision and deep learning (YOLOv5 + PyTorch).

---

## Team Members
| Roll No | Name |
|---------|------|
| 2305052 | Harshit Deka |
| 2305643 | Priyanshu Kabi |
| 2305667 | Vaishnavi Singh |
| 2305807 | Sanskrita Baishya |
| 2305808 | Shahan Ali Anwer |
| 23052302 | Aniket Sengupta |

---

## Project Structure
```
parking_space_detection/
├── module1_data_acquisition/     
├── module2_preprocessing/     
├── module3_slot_mapping/       
├── module4_deep_learning/      
├── module5_classification/      
├── module6_visualization_logging/ 
├── main.py               
└── requirements.txt
```

---

## How to Run

### Local
```bash
pip install -r requirements.txt

python main.py --define_slots --ref_image data/reference_frame.jpg

python main.py --source video --path data/parking_lot.mp4

python main.py --source dataset --path data/PKLot/test/images --no_display
```

### Google Colab (recommended — free T4 GPU)
1. Upload `parking_space_detection.zip` to Google Drive
2. Mount Drive and unzip into `/content/`
3. Install dependencies: `pip install torch torchvision opencv-python-headless PyYAML`
4. Download PKLot dataset from Kaggle
5. Define slot coordinates using the slot mapper or JSON config
6. Run:
```bash
python main.py --source dataset --path /content/PKLot/test/images \
               --model yolo --conf 0.35 --no_display \
               --save_video /content/output.mp4 --log_dir /content/logs
```

---

## Dataset
- **PKLot** — 12,000+ parking lot images under sunny, cloudy, and rainy conditions
- Source: [Kaggle PKLot Dataset](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset)
- No training required — system uses pretrained YOLOv5s COCO weights

---

## Pipeline
```
Camera / Video / Dataset
        ↓
Module 1 — Data Acquisition       (configurable FPS, multi-source)
        ↓
Module 2 — Preprocessing          (resize, normalize, CLAHE)
        ↓
Module 3 — Slot Mapping           (ROI JSON config)
        ↓
Module 4 — YOLOv5s Inference      (PyTorch, COCO pretrained)
        ↓
Module 5 — Classification         (threshold + 3-frame smoothing)
        ↓
Module 6 — Visualization + Log    (green/red overlay, CSV/JSON)
```

---

## Results
- Classification accuracy: **≥ 85%** on PKLot benchmark
- Processing speed: **≥ 15 FPS** on T4 GPU
- Tested on: Google Colab with Tesla T4

---

## Tech Stack
- Python 3.9+
- PyTorch 2.0+
- YOLOv5s (Ultralytics)
- OpenCV
- PKLot Dataset
