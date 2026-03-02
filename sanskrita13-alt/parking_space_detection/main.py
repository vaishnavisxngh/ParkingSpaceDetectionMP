"""
Parking Space Detection System — Main Pipeline
=============================================
Ties all 6 modules together into an end-to-end pipeline.

Usage:
    python main.py --source video --path data/parking_lot.mp4
    python main.py --source camera --device 0
    python main.py --source dataset --path data/PKLot/

    # Define slots first (one-time setup):
    python main.py --define_slots --image data/reference_frame.jpg
"""

import cv2
import sys
import argparse
import time
from pathlib import Path

# Module imports
from module1_data_acquisition.data_loader   import DataLoader
from module2_preprocessing.preprocessor     import Preprocessor
from module3_slot_mapping.slot_mapper       import SlotMapper
from module4_deep_learning.inference_engine import InferenceEngine
from module5_classification.classifier      import SlotClassifier
from module6_visualization_logging.visualizer import Visualizer
from module6_visualization_logging.logger     import OccupancyLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parking Space Detection System (PyTorch)")
    parser.add_argument("--source", choices=["camera", "video", "dataset"],
                        default="video", help="Input source type")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to video file or image folder")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default=0)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Target frame rate")
    parser.add_argument("--model", choices=["yolo", "resnet"], default="yolo",
                        help="Deep learning model (default=yolo)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold")
    parser.add_argument("--slots_config", type=str,
                        default="module3_slot_mapping/slots_config.json",
                        help="Path to parking slot config JSON")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for occupancy logs")
    parser.add_argument("--save_video", type=str, default=None,
                        help="Save annotated output video to this path")
    parser.add_argument("--define_slots", action="store_true",
                        help="Launch interactive slot definition tool")
    parser.add_argument("--ref_image", type=str, default=None,
                        help="Reference image for slot definition")
    parser.add_argument("--no_display", action="store_true",
                        help="Run headlessly (no OpenCV window)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit total frames processed")
    parser.add_argument("--gpu", type=str, default="auto",
                        help="Device: auto | cpu | cuda")
    return parser.parse_args()


def define_slots_mode(ref_image_path: str, output_path: str):
    """Interactive slot definition — saves JSON and exits."""
    frame = cv2.imread(ref_image_path)
    if frame is None:
        print(f"ERROR: Cannot open reference image: {ref_image_path}")
        sys.exit(1)
    mapper = SlotMapper()
    mapper.define_slots_interactive(frame, output_path=output_path)
    print(f"Slot config saved to: {output_path}")
    sys.exit(0)


def run_pipeline(args):
    """Main inference loop."""
    print("=" * 60)
    print("  Parking Space Detection System  ")
    print("=" * 60)

    # ── Module 1: Data Acquisition ──────────────────────────────────
    loader = DataLoader(source=args.source, path=args.path,
                        device=args.device, fps=args.fps,
                        max_frames=args.max_frames)

    # ── Module 2: Preprocessing ─────────────────────────────────────
    preprocessor = Preprocessor(model_type=args.model, enhance=False)

    # ── Module 3: Slot Mapping ──────────────────────────────────────
    mapper = SlotMapper()
    if not Path(args.slots_config).exists():
        print(f"WARNING: Slot config not found at '{args.slots_config}'")
        print("Run with --define_slots --ref_image <frame.jpg> first!")
        print("Continuing without slot overlay...")
        slots = []
    else:
        slots = mapper.load(args.slots_config)
        print(f"Loaded {len(slots)} parking slots.")

    # ── Module 4: Deep Learning Inference ───────────────────────────
    print(f"Loading model: {args.model.upper()} (conf threshold={args.conf})")
    engine = InferenceEngine(model_type=args.model,
                              conf_threshold=args.conf,
                              device=args.gpu)

    # ── Module 5: Classification ─────────────────────────────────────
    classifier = SlotClassifier(occupied_threshold=args.conf, smooth_frames=3)

    # ── Module 6: Visualization & Logging ───────────────────────────
    visualizer = Visualizer(show_confidence=True)
    csv_log = f"{args.log_dir}/occupancy.csv"
    json_log = f"{args.log_dir}/occupancy.json"
    logger = OccupancyLogger(csv_path=csv_log, json_path=json_log, log_interval=5)

    video_writer = None
    frame_times = []

    print("\nStarting pipeline... Press 'q' to quit.\n")

    try:
        for frame_idx, frame in loader.frames():
            t0 = time.time()

            # ── Preprocess (used internally by engine if needed) ────
            # Module 2 output feeds module 4 via slot crops
            slot_crops = mapper.extract_rois(frame) if slots else []

            # ── Inference ───────────────────────────────────────────
            if slots:
                inference_results = engine.infer_slots(
                    frame, slots, slot_crops=slot_crops)
            else:
                inference_results = []

            # ── Classify ────────────────────────────────────────────
            statuses = classifier.classify(inference_results) if inference_results else []
            summary  = classifier.summary(statuses)

            # ── Visualize ───────────────────────────────────────────
            annotated = visualizer.draw(frame, slots, statuses, summary)

            # FPS overlay
            elapsed = time.time() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (12, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # ── Log ─────────────────────────────────────────────────
            logger.log(statuses, summary)

            # ── Save video ──────────────────────────────────────────
            if args.save_video:
                if video_writer is None:
                    h, w = annotated.shape[:2]
                    video_writer = visualizer.create_video_writer(
                        args.save_video, args.fps, (w, h))
                video_writer.write(annotated)

            # ── Display ─────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow("Parking Space Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nUser quit.")
                    break
                elif key == ord("s"):
                    out_path = f"logs/frame_{frame_idx:06d}.jpg"
                    visualizer.save_frame(annotated, out_path)
                    print(f"Saved frame → {out_path}")

            # Console status every 30 frames
            if frame_idx % 30 == 0 and statuses:
                print(f"Frame {frame_idx:05d} | "
                      f"Occupied: {summary['occupied']}/{summary['total']} | "
                      f"FPS: {avg_fps:.1f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        loader.release()
        if video_writer:
            video_writer.release()
        logger.flush_json()
        if not args.no_display:
            cv2.destroyAllWindows()

        stats = logger.get_stats()
        print("\n" + "=" * 60)
        print("Pipeline complete.")
        print(f"  Frames processed : {stats['frames_logged']}")
        print(f"  CSV log          : {stats['csv_path']}")
        print(f"  JSON log         : {stats['json_path']}")
        print("=" * 60)


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Slot definition mode
    if args.define_slots:
        ref = args.ref_image or args.path
        if not ref:
            print("ERROR: Provide --ref_image or --path for slot definition.")
            sys.exit(1)
        define_slots_mode(ref, args.slots_config)

    # Normal pipeline
    run_pipeline(args)
