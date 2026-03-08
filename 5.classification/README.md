# Module 5 — Slot Classification Logic
**Assigned to: Member 6 (Classification Logic)**

## Responsibility
Takes the raw confidence scores from Module 4 and applies threshold-based decision logic to finalize each slot's status as **Occupied** or **Vacant**. Also tracks slot state changes over time to avoid flickering from single noisy frames (temporal smoothing).

## What this module does
- Applies configurable confidence thresholds to classify each slot
- Temporal smoothing: a slot must be consistently detected N frames before its status changes (debouncing)
- Tracks slot history and detects status change events
- Produces the final `SlotStatus` list consumed by Module 6 (Visualization)

## Files
| File | Description |
|------|-------------|
| `classifier.py` | Core threshold + smoothing classification logic |
| `slot_state.py` | Slot state tracker with history buffer |

## Configuration
Key thresholds (set in `config.yaml` or passed to constructor):
- `occupied_threshold`: Confidence above this → OCCUPIED (default 0.5)
- `smooth_frames`: Number of frames to smooth over (default 3)

## Integration
```python
from module5_classification.classifier import SlotClassifier
clf = SlotClassifier(occupied_threshold=0.5, smooth_frames=3)

# Call on every frame's inference results
statuses = clf.classify(inference_results)
# returns: [SlotStatus(id=0, label="A1", status="Occupied", confidence=0.87), ...]
```
