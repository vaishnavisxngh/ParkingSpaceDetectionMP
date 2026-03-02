# Module 6 — Visualization & Logging
**Assigned to: Member 3 + Member 6 (Shared Responsibility)**

## Responsibility
The final output stage. Takes classified slot statuses and renders color-coded overlays (green = vacant, red = occupied) on the live frame. Also maintains occupancy logs in CSV format and optionally saves annotated output video.

## What this module does
- Draws colored bounding boxes over each parking slot (green/red)
- Shows slot label and occupancy count overlay on the frame
- Displays real-time counts: Total / Occupied / Vacant
- Logs occupancy data with timestamps to CSV
- Optional: saves annotated output video
- Optional: Streamlit/Flask dashboard integration hook

## Files
| File | Description |
|------|-------------|
| `visualizer.py` | OpenCV overlay drawing functions |
| `logger.py` | CSV and JSON occupancy logger |
| `dashboard.py` | Optional Streamlit live dashboard |

## Color Scheme
| Status | Color |
|--------|-------|
| Vacant | 🟢 Green `(0, 255, 0)` |
| Occupied | 🔴 Red `(0, 0, 255)` |

## How to run dashboard (optional)
```bash
pip install streamlit
streamlit run dashboard.py
```

## Log format (CSV)
```
timestamp,slot_id,slot_label,status,confidence
2026-02-24 14:30:01,0,A1,Occupied,0.87
2026-02-24 14:30:01,1,A2,Vacant,0.12
```

## Integration
```python
from module6_visualization_logging.visualizer import Visualizer
from module6_visualization_logging.logger import OccupancyLogger

viz = Visualizer()
logger = OccupancyLogger("logs/occupancy.csv")

annotated = viz.draw(frame, slots, statuses, summary)
logger.log(statuses)
cv2.imshow("Parking Monitor", annotated)
```
