"""
Module 1 — Data Acquisition
Member 1: Data Engineer

Handles all data input sources: live camera, video files, and image datasets.
Frames are yielded one at a time for downstream processing.
"""

import cv2
import os
import time
import argparse
from pathlib import Path


class DataLoader:
    """
    Unified data loader that supports:
    - Live camera/webcam feeds
    - Pre-recorded video files
    - Image dataset folders (PKLot, etc.)
    """

    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, source: str, path: str = None, device: int = 0,
                 fps: int = 15, max_frames: int = None):
        """
        Args:
            source  : 'camera' | 'video' | 'dataset'
            path    : Path to video file or image folder (not needed for camera)
            device  : Camera device index (default 0)
            fps     : Target frames per second for extraction
            max_frames: Limit total frames (None = unlimited)
        """
        self.source = source
        self.path = path
        self.device = device
        self.fps = fps
        self.max_frames = max_frames
        self._frame_count = 0
        self._cap = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def frames(self):
        """Generator — yields (frame_index, frame_bgr) tuples."""
        if self.source == "camera":
            yield from self._from_camera()
        elif self.source == "video":
            yield from self._from_video()
        elif self.source == "dataset":
            yield from self._from_dataset()
        else:
            raise ValueError(f"Unknown source '{self.source}'. Use camera/video/dataset.")

    def release(self):
        """Release any open capture handle."""
        if self._cap and self._cap.isOpened():
            self._cap.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _from_camera(self):
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device}")

        interval = 1.0 / self.fps
        print(f"[DataLoader] Camera opened (device={self.device}, target fps={self.fps})")

        while True:
            start = time.time()
            ret, frame = self._cap.read()
            if not ret:
                print("[DataLoader] Camera read failed — stopping.")
                break

            yield self._frame_count, frame
            self._frame_count += 1

            if self.max_frames and self._frame_count >= self.max_frames:
                break

            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.release()

    def _from_video(self):
        if not self.path or not Path(self.path).exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")

        video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        skip = max(1, round(video_fps / self.fps))  # extract every N-th frame
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[DataLoader] Video: {self.path} | source_fps={video_fps:.1f} | "
              f"extracting every {skip} frames | total={total_frames}")

        raw_index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if raw_index % skip == 0:
                yield self._frame_count, frame
                self._frame_count += 1

                if self.max_frames and self._frame_count >= self.max_frames:
                    break

            raw_index += 1

        self.release()

    def _from_dataset(self):
        if not self.path or not Path(self.path).exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.path}")

        image_paths = sorted([
            p for p in Path(self.path).rglob("*")
            if p.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ])

        if not image_paths:
            raise RuntimeError(f"No images found in {self.path}")

        print(f"[DataLoader] Dataset: {len(image_paths)} images found in {self.path}")

        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[DataLoader] Warning: Could not read {img_path}, skipping.")
                continue

            yield self._frame_count, frame
            self._frame_count += 1

            if self.max_frames and self._frame_count >= self.max_frames:
                break


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Module 1: Data Acquisition")
    parser.add_argument("--source", choices=["camera", "video", "dataset"],
                        required=True, help="Input source type")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to video file or dataset folder")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--fps", type=int, default=15,
                        help="Target frames per second")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max frames to extract (default=unlimited)")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview window")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loader = DataLoader(
        source=args.source,
        path=args.path,
        device=args.device,
        fps=args.fps,
        max_frames=args.max_frames
    )

    for idx, frame in loader.frames():
        print(f"[Frame {idx}] shape={frame.shape}")
        if args.preview:
            cv2.imshow("Data Loader Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.preview:
        cv2.destroyAllWindows()
    print(f"[DataLoader] Done. Total frames extracted: {loader._frame_count}")
