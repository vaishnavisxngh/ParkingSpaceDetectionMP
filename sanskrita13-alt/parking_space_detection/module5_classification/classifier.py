"""
Module 5 — Slot Classification Logic
Member 6: Classification Logic

Applies threshold-based classification with temporal smoothing to
convert raw confidence scores into stable Occupied/Vacant decisions.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SlotStatus:
    """Final per-slot classification result for one frame."""
    slot_id: int
    slot_label: str
    status: str          # "Occupied" | "Vacant"
    confidence: float    # 0.0 – 1.0
    changed: bool = False   # True if status changed this frame


class SlotState:
    """
    Tracks occupancy history for one slot over time.
    Applies majority-vote smoothing across a rolling window of frames.
    """

    def __init__(self, slot_id: int, slot_label: str,
                 smooth_frames: int = 3, occupied_threshold: float = 0.5):
        self.slot_id = slot_id
        self.slot_label = slot_label
        self.smooth_frames = smooth_frames
        self.occupied_threshold = occupied_threshold

        # Rolling buffers
        self._conf_history: deque = deque(maxlen=smooth_frames)
        self._status_history: deque = deque(maxlen=smooth_frames)

        # Last committed status
        self.current_status: str = "Vacant"
        self.current_confidence: float = 0.0

    def update(self, confidence: float) -> SlotStatus:
        """
        Feed a new confidence score. Returns smoothed SlotStatus.

        Args:
            confidence: Raw confidence score from inference (0.0–1.0)
        """
        raw_status = "Occupied" if confidence >= self.occupied_threshold else "Vacant"
        self._conf_history.append(confidence)
        self._status_history.append(raw_status)

        # Smoothed status: majority vote over history window
        occupied_votes = self._status_history.count("Occupied")
        smoothed_status = ("Occupied" if occupied_votes > self.smooth_frames / 2
                           else "Vacant")
        smoothed_conf = sum(self._conf_history) / len(self._conf_history)

        changed = smoothed_status != self.current_status
        self.current_status = smoothed_status
        self.current_confidence = smoothed_conf

        return SlotStatus(
            slot_id=self.slot_id,
            slot_label=self.slot_label,
            status=smoothed_status,
            confidence=smoothed_conf,
            changed=changed
        )


class SlotClassifier:
    """
    Manages SlotState objects for all parking slots.
    Called once per frame with the inference results from Module 4.

    Usage:
        clf = SlotClassifier(occupied_threshold=0.5, smooth_frames=3)
        statuses = clf.classify(inference_results)
    """

    def __init__(self, occupied_threshold: float = 0.5,
                 smooth_frames: int = 3):
        """
        Args:
            occupied_threshold : Confidence >= this → Occupied
            smooth_frames      : Rolling window size for temporal smoothing
        """
        self.occupied_threshold = occupied_threshold
        self.smooth_frames = smooth_frames
        self._states: Dict[int, SlotState] = {}

    def classify(self, inference_results: List[Dict]) -> List[SlotStatus]:
        """
        Process inference results for all slots in one frame.

        Args:
            inference_results: Output from InferenceEngine.infer_slots()
                Each dict must have: slot_id, slot_label, confidence, has_vehicle

        Returns:
            List of SlotStatus objects — one per slot
        """
        statuses = []
        for result in inference_results:
            slot_id = result["slot_id"]
            slot_label = result.get("slot_label", str(slot_id))
            confidence = result.get("confidence", 0.0)

            # Initialize state tracker on first encounter
            if slot_id not in self._states:
                self._states[slot_id] = SlotState(
                    slot_id=slot_id,
                    slot_label=slot_label,
                    smooth_frames=self.smooth_frames,
                    occupied_threshold=self.occupied_threshold
                )

            status = self._states[slot_id].update(confidence)
            statuses.append(status)

        return statuses

    def summary(self, statuses: List[SlotStatus]) -> Dict:
        """
        Compute aggregate counts from current statuses.

        Returns:
            {"total": N, "occupied": N, "vacant": N, "occupancy_rate": 0.0–1.0}
        """
        total = len(statuses)
        occupied = sum(1 for s in statuses if s.status == "Occupied")
        vacant = total - occupied
        return {
            "total": total,
            "occupied": occupied,
            "vacant": vacant,
            "occupancy_rate": occupied / total if total > 0 else 0.0
        }

    def reset(self):
        """Clear all slot states (e.g., when loading a new video/camera)."""
        self._states.clear()


# ------------------------------------------------------------------
# Quick standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate inference results for 3 slots over several frames
    mock_inference = [
        {"slot_id": 0, "slot_label": "A1", "confidence": 0.85, "has_vehicle": True},
        {"slot_id": 1, "slot_label": "A2", "confidence": 0.10, "has_vehicle": False},
        {"slot_id": 2, "slot_label": "A3", "confidence": 0.60, "has_vehicle": True},
    ]

    clf = SlotClassifier(occupied_threshold=0.5, smooth_frames=3)

    print("Simulating 5 frames:")
    for frame_n in range(5):
        # Introduce slight confidence variation per frame
        varied = [{**r, "confidence": min(1.0, r["confidence"] + (frame_n * 0.02 - 0.04))}
                  for r in mock_inference]
        statuses = clf.classify(varied)
        summary = clf.summary(statuses)
        print(f"\nFrame {frame_n + 1}: Occupied={summary['occupied']} Vacant={summary['vacant']}")
        for s in statuses:
            change_marker = " ← CHANGED" if s.changed else ""
            print(f"  {s.slot_label}: {s.status} (conf={s.confidence:.2f}){change_marker}")
