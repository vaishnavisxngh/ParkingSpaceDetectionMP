"""
Module 4 — Deep Learning Inference
Member 5: ML Engineer

Runs vehicle detection/classification on parking slot ROIs using PyTorch.
Supports YOLOv5s (recommended) and ResNet18.
"""

import cv2
import torch
import numpy as np
import argparse
from typing import List, Dict, Any
from pathlib import Path


# -----------------------------------------------------------------------
# YOLO vehicle class IDs (COCO dataset)
# -----------------------------------------------------------------------
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YOLODetector:
    """
    YOLOv5s-based vehicle detector using torch.hub.
    Detects vehicles in full frames; maps detections to slot ROIs.
    """

    def __init__(self, conf_threshold: float = 0.4, device: str = "auto"):
        self.conf_threshold = conf_threshold
        self.device = self._resolve_device(device)
        print(f"[YOLODetector] Loading YOLOv5s on {self.device}...")
        # Downloads YOLOv5s pretrained weights automatically
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s",
                                     pretrained=True, verbose=False)
        self.model.to(self.device)
        self.model.eval()
        self.model.conf = conf_threshold
        print("[YOLODetector] Model ready.")

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Run YOLO on a full frame. Returns list of vehicle detections.

        Returns:
            List of dicts: {label, confidence, bbox: (x1,y1,x2,y2)}
        """
        # YOLO expects RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)
        detections = []
        for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
            cls_id = int(cls_id)
            if cls_id in VEHICLE_CLASSES and float(conf) >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "label": VEHICLE_CLASSES[cls_id],
                    "confidence": float(conf),
                    "bbox": (x1, y1, x2, y2)
                })
        return detections

    def check_slots(self, frame_bgr: np.ndarray, slots) -> List[Dict]:
        """
        For each slot ROI, check if any vehicle detection overlaps with it.

        Args:
            frame_bgr : Full frame
            slots     : List of ParkingSlot objects

        Returns:
            List of slot results: {slot_id, label, confidence, has_vehicle}
        """
        detections = self.detect(frame_bgr)
        results = []
        for slot in slots:
            sx1, sy1, sx2, sy2 = slot.corners
            best_conf = 0.0
            best_label = "none"
            for det in detections:
                dx1, dy1, dx2, dy2 = det["bbox"]
                # Compute intersection over slot area (IoS)
                ix1 = max(sx1, dx1)
                iy1 = max(sy1, dy1)
                ix2 = min(sx2, dx2)
                iy2 = min(sy2, dy2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                slot_area = (sx2 - sx1) * (sy2 - sy1)
                ios = inter_area / slot_area if slot_area > 0 else 0.0
                if ios > 0.2 and det["confidence"] > best_conf:
                    best_conf = det["confidence"]
                    best_label = det["label"]
            results.append({
                "slot_id": slot.id,
                "slot_label": slot.label,
                "confidence": best_conf,
                "has_vehicle": best_conf > 0,
                "vehicle_type": best_label
            })
        return results

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device


class ResNetClassifier:
    """
    ResNet18-based binary classifier for per-slot occupied/vacant detection.
    Uses a fine-tuned or feature-extraction approach on slot crops.
    """

    def __init__(self, weights_path: str = None, device: str = "auto",
                 conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"

        from torchvision import models, transforms
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])

        self.model = models.resnet18(pretrained=(weights_path is None))
        # Replace final layer for binary classification
        import torch.nn as nn
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        if weights_path and Path(weights_path).exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"[ResNetClassifier] Loaded weights from {weights_path}")
        else:
            print("[ResNetClassifier] Using ImageNet pretrained features (no fine-tuned weights).")

        self.model.to(self.device)
        self.model.eval()

    def classify_crop(self, crop_bgr: np.ndarray) -> Dict:
        """Classify a single slot crop as occupied or vacant."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transforms(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        occupied_prob = probs[1].item()
        return {
            "confidence": occupied_prob,
            "has_vehicle": occupied_prob >= self.conf_threshold
        }


class InferenceEngine:
    """
    Unified inference engine — wraps YOLO or ResNet.
    This is the primary interface used by the pipeline.
    """

    def __init__(self, model_type: str = "yolo", conf_threshold: float = 0.4,
                 device: str = "auto", weights_path: str = None):
        """
        Args:
            model_type      : 'yolo' | 'resnet'
            conf_threshold  : Minimum confidence to mark slot as occupied
            device          : 'cpu' | 'cuda' | 'auto'
            weights_path    : Path to custom ResNet weights (optional)
        """
        self.model_type = model_type
        if model_type == "yolo":
            self.detector = YOLODetector(conf_threshold=conf_threshold, device=device)
        elif model_type == "resnet":
            self.classifier = ResNetClassifier(weights_path=weights_path,
                                                device=device,
                                                conf_threshold=conf_threshold)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Use 'yolo' or 'resnet'.")

    def infer_slots(self, frame_bgr: np.ndarray, slots,
                    slot_crops: List[np.ndarray] = None) -> List[Dict]:
        """
        Run inference on all slots.

        Args:
            frame_bgr  : Full frame (used by YOLO)
            slots      : List of ParkingSlot objects
            slot_crops : List of pre-cropped BGR images (used by ResNet)

        Returns:
            List of dicts per slot: {slot_id, slot_label, confidence, has_vehicle, ...}
        """
        if self.model_type == "yolo":
            return self.detector.check_slots(frame_bgr, slots)

        elif self.model_type == "resnet":
            if slot_crops is None:
                raise ValueError("ResNet requires slot_crops to be provided.")
            results = []
            for slot, crop in zip(slots, slot_crops):
                if crop is None or crop.size == 0:
                    result = {"slot_id": slot.id, "slot_label": slot.label,
                              "confidence": 0.0, "has_vehicle": False}
                else:
                    r = self.classifier.classify_crop(crop)
                    result = {"slot_id": slot.id, "slot_label": slot.label, **r}
                results.append(result)
            return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Module 4: Deep Learning Inference")
    parser.add_argument("--model", choices=["yolo", "resnet"], default="yolo")
    parser.add_argument("--image", required=True, help="Test image path")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Cannot read {args.image}")
        exit(1)

    engine = InferenceEngine(model_type=args.model, conf_threshold=args.conf,
                              device=args.device)

    # Quick test: load slot config if it exists
    config_path = "../module3_slot_mapping/slots_config.json"
    if Path(config_path).exists():
        import sys
        sys.path.append("..")
        from module3_slot_mapping.slot_mapper import SlotMapper
        mapper = SlotMapper()
        slots = mapper.load(config_path)
        crops = mapper.extract_rois(frame)
        results = engine.infer_slots(frame, slots, slot_crops=crops)
        for r in results:
            status = "OCCUPIED" if r["has_vehicle"] else "VACANT"
            print(f"  Slot {r['slot_label']}: {status} (conf={r['confidence']:.2f})")
    else:
        print("No slots_config.json found. Run module3 first.")
