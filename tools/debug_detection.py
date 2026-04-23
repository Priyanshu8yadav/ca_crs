"""
Save one annotated detection frame for detector tuning.

Example:
    python tools/debug_detection.py --input scen_b.mp4 --frame 300 \
        --model yolov8m.pt --tile_size 320 --imgsz 640
"""

import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.density_module import DensityEstimator


def read_frame(path, frame_index):
    src = int(path) if str(path).isdigit() else path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
    return frame


def draw_boxes(frame, boxes):
    out = frame.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    return out


def main(args):
    frame = read_frame(args.input, args.frame)
    estimator = DensityEstimator(
        conf=args.density_conf,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        overlap=args.tile_overlap,
        imgsz=args.imgsz,
        model_name=args.model,
        iou_thresh=args.iou_thresh,
        count_correction=args.count_correction,
        max_det_per_tile=args.max_det_per_tile,
        tile_size=args.tile_size,
        full_frame_pass=not args.no_full_frame_pass,
        augment=args.augment,
    )

    raw_count, boxes = estimator.detect(frame, corrected=False)
    corrected_count = estimator.correct_count(raw_count)
    out = draw_boxes(frame, boxes)

    label = (
        f"raw={raw_count} corrected={corrected_count} "
        f"model={getattr(estimator, 'loaded_model_name', args.model)} "
        f"tile={args.tile_size} imgsz={args.imgsz} conf={args.density_conf}"
    )
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 1500), 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f"[debug_detection] raw_count={raw_count}")
    print(f"[debug_detection] corrected_count={corrected_count}")
    print(f"[debug_detection] saved={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scen_b.mp4")
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--output", default="results/debug_detection.jpg")
    parser.add_argument("--density_conf", type=float, default=0.04)
    parser.add_argument("--tile_rows", type=int, default=5)
    parser.add_argument("--tile_cols", type=int, default=4)
    parser.add_argument("--tile_overlap", type=float, default=0.30)
    parser.add_argument("--tile_size", type=int, default=320)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--iou_thresh", type=float, default=0.45)
    parser.add_argument("--max_det_per_tile", type=int, default=3000)
    parser.set_defaults(no_full_frame_pass=True)
    parser.add_argument("--full_frame_pass", dest="no_full_frame_pass",
                        action="store_false")
    parser.add_argument("--no_full_frame_pass", dest="no_full_frame_pass",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--count_correction", type=float, default=1.8)
    main(parser.parse_args())
