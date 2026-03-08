"""
Module 6 — Occupancy Logger
Member 3 + Member 6 (Shared Responsibility)

Logs parking occupancy data with timestamps to CSV and JSON.
"""

import csv
import json
import os
from datetime import datetime
from typing import List
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from module5_classification.classifier import SlotStatus


class OccupancyLogger:
    """
    Logs slot occupancy to CSV and optionally JSON.

    CSV columns: timestamp, slot_id, slot_label, status, confidence
    """

    def __init__(self, csv_path: str = "logs/occupancy.csv",
                 json_path: str = None,
                 log_interval: int = 1):
        """
        Args:
            csv_path     : Output CSV file path
            json_path    : Optional JSON log path
            log_interval : Log every N frames (default=1 = every frame)
        """
        self.csv_path = csv_path
        self.json_path = json_path
        self.log_interval = log_interval
        self._frame_count = 0
        self._records = []

        # Ensure output directory exists
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

        # Write CSV header if new file
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "slot_id", "slot_label",
                                  "status", "confidence"])

        print(f"[Logger] Logging to: {csv_path}")

    def log(self, statuses: List[SlotStatus], summary: dict = None):
        """
        Log current frame's slot statuses.

        Args:
            statuses : List of SlotStatus objects
            summary  : Optional summary dict (for JSON enrichment)
        """
        self._frame_count += 1
        if self._frame_count % self.log_interval != 0:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # CSV logging
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for s in statuses:
                writer.writerow([
                    timestamp,
                    s.slot_id,
                    s.slot_label,
                    s.status,
                    f"{s.confidence:.4f}"
                ])

        # In-memory JSON records
        if self.json_path:
            entry = {
                "timestamp": timestamp,
                "slots": [
                    {
                        "id": s.slot_id,
                        "label": s.slot_label,
                        "status": s.status,
                        "confidence": round(s.confidence, 4)
                    }
                    for s in statuses
                ]
            }
            if summary:
                entry["summary"] = summary
            self._records.append(entry)

    def flush_json(self):
        """Write all in-memory records to JSON file."""
        if not self.json_path:
            return
        os.makedirs(os.path.dirname(self.json_path) if os.path.dirname(self.json_path) else ".",
                    exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(self._records, f, indent=2)
        print(f"[Logger] JSON log written: {self.json_path} ({len(self._records)} entries)")

    def get_stats(self) -> dict:
        """Return basic stats about what's been logged."""
        return {
            "frames_logged": self._frame_count,
            "csv_path": self.csv_path,
            "json_path": self.json_path,
            "total_records": len(self._records)
        }
