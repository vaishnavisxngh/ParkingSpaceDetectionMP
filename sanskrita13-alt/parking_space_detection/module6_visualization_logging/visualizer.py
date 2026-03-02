"""
Module 6 — Visualization
Member 3 + Member 6 (Shared Responsibility)

Renders color-coded overlays on frames showing parking slot occupancy.
Green = Vacant, Red = Occupied.
"""

import cv2
import numpy as np
from typing import List, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from module3_slot_mapping.slot_mapper import ParkingSlot
from module5_classification.classifier import SlotStatus


# Color constants (BGR)
COLOR_VACANT   = (0, 200, 0)     # Green
COLOR_OCCUPIED = (0, 0, 220)     # Red
COLOR_UNKNOWN  = (180, 180, 0)   # Yellow
COLOR_TEXT     = (255, 255, 255)
COLOR_BG       = (30, 30, 30)
ALPHA          = 0.35            # Overlay transparency


class Visualizer:
    """
    Draws parking occupancy overlays on video frames.
    """

    def __init__(self, font_scale: float = 0.55, thickness: int = 2,
                 show_confidence: bool = True, alpha: float = ALPHA):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.show_confidence = show_confidence
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Primary drawing method
    # ------------------------------------------------------------------

    def draw(self, frame_bgr: np.ndarray,
             slots: List[ParkingSlot],
             statuses: List[SlotStatus],
             summary: Dict) -> np.ndarray:
        """
        Annotate a frame with slot overlays and summary HUD.

        Args:
            frame_bgr : Raw BGR frame from OpenCV
            slots     : List of ParkingSlot (for coordinates)
            statuses  : List of SlotStatus (for classification results)
            summary   : Dict from SlotClassifier.summary()

        Returns:
            Annotated BGR frame
        """
        output = frame_bgr.copy()
        status_map = {s.slot_id: s for s in statuses}

        # Draw each slot rectangle with filled transparent overlay
        overlay = output.copy()
        for slot in slots:
            status = status_map.get(slot.id)
            color = self._status_color(status)
            x1, y1, x2, y2 = slot.corners
            # Filled rectangle on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Blend overlay with original
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)

        # Draw borders and labels on top
        for slot in slots:
            status = status_map.get(slot.id)
            color = self._status_color(status)
            x1, y1, x2, y2 = slot.corners

            # Border
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)

            # Slot label
            label_text = slot.label
            if status and self.show_confidence:
                label_text += f" {status.confidence:.0%}"

            # Background pill for text
            (tw, th), _ = cv2.getTextSize(label_text, self.font,
                                           self.font_scale, self.thickness)
            cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(output, label_text, (x1 + 3, y1 - 4),
                        self.font, self.font_scale, COLOR_TEXT, self.thickness)

            # Status text inside box
            if status:
                status_text = "OCC" if status.status == "Occupied" else "FREE"
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                (stw, sth), _ = cv2.getTextSize(status_text, self.font, 0.65, 2)
                cv2.putText(output, status_text, (cx - stw // 2, cy + sth // 2),
                            self.font, 0.65, COLOR_TEXT, 2)

        # Draw HUD panel
        output = self._draw_hud(output, summary)
        return output

    # ------------------------------------------------------------------
    # HUD (heads-up display)
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, summary: Dict) -> np.ndarray:
        """Draw top-left summary panel with slot counts."""
        total    = summary.get("total", 0)
        occupied = summary.get("occupied", 0)
        vacant   = summary.get("vacant", 0)
        rate     = summary.get("occupancy_rate", 0.0)

        lines = [
            f"PARKING MONITOR",
            f"Total   : {total}",
            f"Occupied: {occupied}",
            f"Vacant  : {vacant}",
            f"Usage   : {rate:.0%}"
        ]

        x, y, pad = 12, 12, 8
        line_h = 24
        box_w = 200
        box_h = pad * 2 + line_h * len(lines)

        # Background
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), COLOR_BG, -1)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (80, 80, 80), 1)

        for i, line in enumerate(lines):
            color = COLOR_TEXT
            if "Occupied" in line:
                color = COLOR_OCCUPIED
            elif "Vacant" in line:
                color = COLOR_VACANT
            elif "MONITOR" in line:
                color = (200, 200, 255)
            ty = y + pad + (i + 1) * line_h - 4
            cv2.putText(frame, line, (x + pad, ty),
                        self.font, 0.52, color, 1)

        return frame

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _status_color(self, status: SlotStatus) -> tuple:
        if status is None:
            return COLOR_UNKNOWN
        return COLOR_OCCUPIED if status.status == "Occupied" else COLOR_VACANT

    def save_frame(self, frame: np.ndarray, path: str):
        """Save a single annotated frame as image."""
        cv2.imwrite(path, frame)

    def create_video_writer(self, output_path: str, fps: int,
                             frame_size: tuple) -> cv2.VideoWriter:
        """Create a VideoWriter for saving annotated output video."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, fps, frame_size)
