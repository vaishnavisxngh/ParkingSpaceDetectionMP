# Module 3 — Parking Slot Mapping
**Assigned to: Member 3 (Mapping & Config) + Member 4 (Integration Engineer)**

## Responsibility
This module lets users define the parking spaces (ROIs) in the camera view either interactively (by clicking on a frame) or by loading saved coordinates from a JSON/YAML config file. It also provides utilities to extract each slot's crop from any frame, which is fed directly into the deep learning module.

## What this module does
- Interactive ROI definition tool: click corners of each parking slot on a reference frame
- Saves/loads slot coordinates from `slots_config.json`
- Extracts cropped ROI images per slot from any frame
- Supports multiple parking slots per camera view
- Integrates seamlessly with Module 2 output and Module 4 input

## Files
| File | Description |
|------|-------------|
| `slot_mapper.py` | Interactive ROI definition + save/load |
| `roi_extractor.py` | Extracts slot crops from frames |
| `slots_config.json` | Saved slot coordinate definitions |
| `slot_visualizer.py` | Draws slot boundaries on frames |

## How to define slots (interactive)
```bash
# Open a reference frame and click to define slot corners
python slot_mapper.py --image data/reference_frame.jpg --output slots_config.json
```
**Controls in the drawing window:**
- Left-click drag → draw a rectangle for one slot
- Press `s` → save current slot
- Press `c` → clear last slot
- Press `q` → save all and quit

## How to load slots
```python
from module3_slot_mapping.slot_mapper import SlotMapper
mapper = SlotMapper()
slots = mapper.load("slots_config.json")  # list of {id, x, y, w, h, label}
```
