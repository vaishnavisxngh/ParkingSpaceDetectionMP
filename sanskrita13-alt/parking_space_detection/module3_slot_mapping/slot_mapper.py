"""
Module 3 — Parking Slot Mapping
Member 3: Mapping & Config | Member 4: Integration Engineer

Interactive tool to define parking slot ROIs by clicking on a reference frame.
Saves/loads slot coordinates in JSON format.
"""

import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class ParkingSlot:
    """Represents a single parking slot region of interest."""
    id: int
    x: int       # top-left x
    y: int       # top-left y
    w: int       # width
    h: int       # height
    label: str = ""   # optional name e.g. "A1"

    @property
    def bbox(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def corners(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class SlotMapper:
    """
    Manages parking slot definitions:
    - Interactive drawing via mouse
    - Save/load JSON config
    - Extract slot ROIs from frames
    """

    def __init__(self):
        self.slots: List[ParkingSlot] = []
        self._drawing = False
        self._start_pt = None
        self._current_rect = None
        self._ref_frame = None

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save all defined slots to JSON."""
        data = {
            "slot_count": len(self.slots),
            "slots": [asdict(s) for s in self.slots]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SlotMapper] Saved {len(self.slots)} slots → {path}")

    def load(self, path: str) -> List[ParkingSlot]:
        """Load slot definitions from JSON. Returns list of ParkingSlot."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Slot config not found: {path}")
        with open(path) as f:
            data = json.load(f)
        self.slots = [ParkingSlot(**s) for s in data["slots"]]
        print(f"[SlotMapper] Loaded {len(self.slots)} slots from {path}")
        return self.slots

    # ------------------------------------------------------------------
    # Interactive ROI drawing
    # ------------------------------------------------------------------

    def define_slots_interactive(self, reference_image: np.ndarray,
                                  output_path: str = "slots_config.json") -> List[ParkingSlot]:
        """
        Open an OpenCV window and let the user draw parking slot rectangles.

        Controls:
            Left-click drag  → draw a rectangle
            's'              → save drawn rectangle as a slot
            'c'              → clear last added slot
            'r'              → reset all slots
            'q'              → quit and save JSON
        """
        self._ref_frame = reference_image.copy()
        display = self._ref_frame.copy()
        slot_id = 0

        cv2.namedWindow("Define Parking Slots", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Define Parking Slots", self._mouse_callback)

        print("[SlotMapper] Draw rectangles over each parking slot.")
        print("  LEFT-CLICK DRAG → draw | 's' → save slot | 'c' → clear last | 'q' → done")

        while True:
            frame_show = self._draw_existing_slots(display.copy())

            # Show the rectangle being drawn
            if self._current_rect:
                x1, y1, x2, y2 = self._current_rect
                cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)

            cv2.imshow("Define Parking Slots", frame_show)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s') and self._current_rect:
                x1, y1, x2, y2 = self._current_rect
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                if w > 5 and h > 5:
                    label = input(f"Label for slot {slot_id} (press Enter to skip): ").strip()
                    self.slots.append(ParkingSlot(id=slot_id, x=x, y=y, w=w, h=h,
                                                   label=label or f"S{slot_id}"))
                    print(f"[SlotMapper] Slot {slot_id} saved: x={x}, y={y}, w={w}, h={h}")
                    slot_id += 1
                    self._current_rect = None

            elif key == ord('c') and self.slots:
                removed = self.slots.pop()
                print(f"[SlotMapper] Removed last slot: {removed}")

            elif key == ord('r'):
                self.slots.clear()
                slot_id = 0
                print("[SlotMapper] All slots cleared.")

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        self.save(output_path)
        return self.slots

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start_pt = (x, y)
            self._current_rect = (x, y, x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            x1, y1 = self._start_pt
            self._current_rect = (x1, y1, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            x1, y1 = self._start_pt
            self._current_rect = (x1, y1, x, y)

    def _draw_existing_slots(self, frame: np.ndarray) -> np.ndarray:
        for slot in self.slots:
            x1, y1, x2, y2 = slot.corners
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, slot.label, (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Slots: {len(self.slots)} | s=save c=clear r=reset q=done",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    # ------------------------------------------------------------------
    # ROI Extraction
    # ------------------------------------------------------------------

    def extract_rois(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Crop each slot region from a frame.

        Returns:
            List of BGR crops, one per slot (same order as self.slots)
        """
        crops = []
        h, w = frame_bgr.shape[:2]
        for slot in self.slots:
            x = max(0, slot.x)
            y = max(0, slot.y)
            x2 = min(w, slot.x + slot.w)
            y2 = min(h, slot.y + slot.h)
            crop = frame_bgr[y:y2, x:x2]
            crops.append(crop)
        return crops

    def get_slots(self) -> List[ParkingSlot]:
        return self.slots

    def slot_count(self) -> int:
        return len(self.slots)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Module 3: Parking Slot Mapper")
    parser.add_argument("--image", required=True,
                        help="Reference frame image to draw slots on")
    parser.add_argument("--output", default="slots_config.json",
                        help="Output JSON path for slot coordinates")
    parser.add_argument("--load", default=None,
                        help="Load existing config to preview")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Cannot read image {args.image}")
        exit(1)

    mapper = SlotMapper()

    if args.load:
        mapper.load(args.load)
        # Preview loaded slots
        preview = frame.copy()
        for slot in mapper.slots:
            x1, y1, x2, y2 = slot.corners
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, slot.label, (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Loaded Slots Preview", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        mapper.define_slots_interactive(frame, output_path=args.output)
