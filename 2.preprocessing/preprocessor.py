"""
Module 2 — Image Preprocessing
Member 2: Preprocessing Specialist

Prepares raw frames for deep learning inference:
- Resizing to model input dimensions
- Pixel normalization
- Optional image enhancement (CLAHE, blur)
- Returns both processed NumPy arrays and PyTorch tensors
"""

import cv2
import numpy as np
import torch
import argparse
from pathlib import Path


# Model-specific input sizes
MODEL_INPUT_SIZES = {
    "yolo":   (640, 640),
    "resnet": (224, 224),
    "cnn":    (64, 64),      # lightweight custom CNN
}

# Normalization stats (ImageNet mean/std for ResNet/YOLO)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Preprocessor:
    """
    Transforms raw BGR frames (from OpenCV) into model-ready tensors.

    Usage:
        prep = Preprocessor(model_type="yolo", enhance=True)
        tensor = prep.process(frame)        # -> torch.Tensor [1, 3, H, W]
        roi_tensor = prep.process_roi(frame, (x, y, w, h))
    """

    def __init__(self, model_type: str = "yolo", enhance: bool = False,
                 normalize: str = "imagenet"):
        """
        Args:
            model_type : 'yolo' | 'resnet' | 'cnn'
            enhance    : Apply CLAHE contrast + Gaussian denoising
            normalize  : 'imagenet' | 'zero_one' | 'none'
        """
        if model_type not in MODEL_INPUT_SIZES:
            raise ValueError(f"Unknown model_type '{model_type}'. "
                             f"Choose from {list(MODEL_INPUT_SIZES)}")
        self.model_type = model_type
        self.target_size = MODEL_INPUT_SIZES[model_type]
        self.enhance = enhance
        self.normalize = normalize

        # CLAHE for contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Full preprocessing pipeline for a complete frame.

        Returns:
            torch.Tensor of shape [1, 3, H, W] — batch of 1, float32
        """
        img = self._to_rgb(frame_bgr)
        if self.enhance:
            img = self._enhance(img)
        img = self._resize(img)
        img = self._normalize(img)
        return self._to_tensor(img)

    def process_roi(self, frame_bgr: np.ndarray,
                    roi: tuple) -> torch.Tensor:
        """
        Crop a region of interest and preprocess it independently.

        Args:
            frame_bgr : Full frame (H, W, 3) BGR
            roi       : (x, y, w, h) bounding box

        Returns:
            torch.Tensor [1, 3, H, W]
        """
        x, y, w, h = roi
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            raise ValueError(f"Empty ROI crop for bbox {roi}")
        return self.process(crop)

    def batch_process_rois(self, frame_bgr: np.ndarray,
                           rois: list) -> torch.Tensor:
        """
        Process multiple ROIs at once — returns a batched tensor.

        Returns:
            torch.Tensor [N, 3, H, W]
        """
        tensors = [self.process_roi(frame_bgr, roi) for roi in rois]
        return torch.cat(tensors, dim=0)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _to_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR (OpenCV default) to RGB."""
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _enhance(self, img_rgb: np.ndarray) -> np.ndarray:
        """Apply CLAHE on luminance channel + Gaussian denoising."""
        # Work in LAB color space for better luminance control
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        # Mild Gaussian blur to reduce sensor noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=0.5)
        return enhanced

    def _resize(self, img_rgb: np.ndarray) -> np.ndarray:
        """Resize to model input size with letterboxing to preserve aspect ratio."""
        h, w = img_rgb.shape[:2]
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_top  = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        return canvas

    def _normalize(self, img_rgb: np.ndarray) -> np.ndarray:
        """Normalize pixel values."""
        img = img_rgb.astype(np.float32) / 255.0
        if self.normalize == "imagenet":
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        elif self.normalize == "zero_one":
            pass  # already in [0, 1]
        # 'none' → leave as-is
        return img

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """HWC numpy → BCHW torch tensor."""
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor  # [1, C, H, W]


# ------------------------------------------------------------------
# CLI entry point (for standalone testing)
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Module 2: Image Preprocessing")
    parser.add_argument("--input", required=True, help="Path to image file")
    parser.add_argument("--model_type", choices=["yolo", "resnet", "cnn"],
                        default="yolo")
    parser.add_argument("--enhance", action="store_true",
                        help="Apply CLAHE + denoising")
    parser.add_argument("--normalize", choices=["imagenet", "zero_one", "none"],
                        default="imagenet")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"Error: Could not read {args.input}")
        exit(1)

    prep = Preprocessor(model_type=args.model_type,
                        enhance=args.enhance,
                        normalize=args.normalize)
    tensor = prep.process(frame)
    print(f"[Preprocessor] Input shape : {frame.shape}")
    print(f"[Preprocessor] Output tensor: {tensor.shape} | dtype={tensor.dtype}")
    print(f"[Preprocessor] Value range  : [{tensor.min():.3f}, {tensor.max():.3f}]")
