"""
main.py
───────
CA-CRS+ single-camera, multi-zone pipeline.

Usage:
    python main.py --input crowd_video.mp4
    python main.py --input crowd_video.mp4 --save_output
    python main.py --input 0                    # webcam
    python main.py --input crowd.mp4 --marshals 15 --medics 3
"""

import cv2
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.zone_manager     import ZoneManager
from modules.density_module   import DensityEstimator
from modules.motion_module    import MotionAnalyzer
from modules.risk_scoring     import CACRSScorer
from modules.gate_recommender import GateRecommender
from modules.resource_manager import ResourceManager
from modules.simulation       import InterventionSimulator
from modules.visualizer       import Visualizer
from modules.logger           import ResultLogger


def determine_frame_stride(args, source_fps: float) -> int:
    if args.frame_stride and args.frame_stride > 1:
        return int(args.frame_stride)
    if args.target_fps > 0 and source_fps > 0:
        return max(1, int(round(source_fps / args.target_fps)))
    return 1


def save_run_metadata(output_dir, args, input_path, source_fps,
                      input_total_frames, effective_stride,
                      processed_frames, elapsed_seconds, density_est):
    path = os.path.join(output_dir, "run_metadata.txt")
    sampled_fps = source_fps / effective_stride if source_fps > 0 else 0.0
    loaded_model = getattr(density_est, "loaded_model_name", args.model)
    with open(path, "w") as handle:
        handle.write("=== CA-CRS+ Run Metadata ===\n\n")
        handle.write(f"Input              : {input_path}\n")
        handle.write(f"Model              : {loaded_model}\n")
        handle.write(f"Source FPS         : {source_fps:.3f}\n")
        handle.write(f"Source frames      : {input_total_frames}\n")
        handle.write(f"Frame stride       : {effective_stride}\n")
        handle.write(f"Sampled FPS        : {sampled_fps:.3f}\n")
        handle.write(f"Processed frames   : {processed_frames}\n")
        handle.write(f"Elapsed seconds    : {elapsed_seconds:.2f}\n")
        handle.write(f"YOLO conf          : {args.density_conf}\n")
        handle.write(f"Tile size          : {args.tile_size}\n")
        handle.write(f"Tile overlap       : {args.tile_overlap}\n")
        handle.write(f"Image size         : {args.imgsz}\n")
        handle.write(f"Count correction   : {args.count_correction}\n")
        handle.write(f"Full-frame pass    : {not args.no_full_frame_pass}\n")
        handle.write(f"Augment            : {args.augment}\n")


def main(args):
    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.input}")
        return

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    input_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ret, frame0 = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        return

    effective_stride = determine_frame_stride(args, source_fps)
    sampled_fps = source_fps / effective_stride if source_fps > 0 else 0.0

    # ── Init modules ──────────────────────────────────────────────────────
    zone_mgr    = ZoneManager(frame_shape=frame0.shape)
    density_est = DensityEstimator(
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
                    augment=args.augment)
    motion_an   = MotionAnalyzer()
    scorer      = CACRSScorer()
    gate_rec    = GateRecommender()
    resources   = ResourceManager(
                    marshals=args.marshals,
                    medics=args.medics,
                    ambulances=args.ambulances)
    simulator   = InterventionSimulator()
    visualizer  = Visualizer()
    logger      = ResultLogger(args.output_dir)

    # ── Video writer ──────────────────────────────────────────────────────
    writer = None
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 295  # +sidebar
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_fps = sampled_fps if sampled_fps > 0 else 20.0
        writer = cv2.VideoWriter(
            os.path.join(args.output_dir, "output.mp4"),
            fourcc, writer_fps, (w, h)
        )

    prev_gray   = None
    source_frame_idx = 1
    processed_frames = 0
    frame = frame0
    start_time  = time.time()
    zone_count_ema = {}

    print("[INFO] CA-CRS+ running. Press Q to quit, S to save screenshot.")
    print(
        f"[INFO] Model={getattr(density_est, 'loaded_model_name', args.model)} "
        f"| stride={effective_stride} | sampled_fps={sampled_fps:.2f}"
    )

    while True:
        if args.max_processed_frames > 0 and processed_frames >= args.max_processed_frames:
            break

        if effective_stride > 1 and ((source_frame_idx - 1) % effective_stride != 0):
            ret, frame = cap.read()
            if not ret:
                break
            source_frame_idx += 1
            continue

        processed_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detection ─────────────────────────────────────────────────────
        _count_total, boxes = density_est.detect(frame, corrected=True)

        # ── Assign persons to zones ───────────────────────────────────────
        zone_boxes_raw = zone_mgr.assign_boxes_to_zones(boxes)
        zone_counts = {
            zone["name"]: density_est.correct_count(
                len(zone_boxes_raw.get(zone["name"], []))
            )
            for zone in zone_mgr.zones
        }
        if args.count_smoothing > 0:
            alpha = min(max(args.count_smoothing, 0.0), 1.0)
            for zname, zcount in list(zone_counts.items()):
                prev = zone_count_ema.get(zname, float(zcount))
                smooth = alpha * float(zcount) + (1.0 - alpha) * prev
                zone_count_ema[zname] = smooth
                zone_counts[zname] = int(round(smooth))

        # ── Motion (global) ───────────────────────────────────────────────
        if prev_gray is not None:
            s_norm, c_norm, _ = motion_an.analyze(prev_gray, gray)
        else:
            s_norm, c_norm = 0.0, 0.0

        # ── Per-zone CA-CRS+ ──────────────────────────────────────────────
        zone_results     = {}
        zone_risk_levels = {}

        # Compute CRS for all zones first (needed for ripple check)
        zone_crs_map = {}
        for zone in zone_mgr.zones:
            zname   = zone["name"]
            zid     = zone["id"]
            zcount  = zone_counts.get(zname, 0)
            zarea   = zone["area_m2"]
            d_norm  = density_est.dnorm(zcount, zarea)
            crs, comps, _ = scorer.score(d_norm, s_norm, c_norm)
            zone_crs_map[zid] = crs

        # Now compute full results with ripple check
        for zone in zone_mgr.zones:
            zname   = zone["name"]
            zid     = zone["id"]
            zcount  = zone_counts.get(zname, 0)
            zarea   = zone["area_m2"]
            d_norm  = density_est.dnorm(zcount, zarea)
            density = density_est.density(zcount, zarea)

            crs, comps, phi = scorer.score(d_norm, s_norm, c_norm)
            risk             = scorer.classify(crs)
            factor           = scorer.dominant_factor(comps)

            # Adjacent zone CRS for ripple check
            adj_ids    = zone_mgr.get_adjacent_zone_ids(zid)
            adj_scores = {aid: zone_crs_map.get(aid, 0.0)
                          for aid in adj_ids}

            # Gate recommendation with ripple check
            zg  = zone.get("gates", [])
            gas = gate_rec.recommend_gates(
                zname, zid, factor, risk, zg, adj_scores
            )
            gate_summary = gate_rec.format_summary(gas)

            # Simulation validation
            action     = gas[0]["action"] if gas else "HOLD"
            sim_result = simulator.validate_action(
                d_norm, s_norm, c_norm, action, crs
            )
            proj_crs   = sim_result["proj_crs"]
            crr        = sim_result["crr"]

            # Collect boxes for this zone (approximate by zone index)
            zone_results[zname] = {
                "count":    zcount,
                "density":  density,
                "d_norm":   d_norm,
                "s_norm":   s_norm,
                "c_norm":   c_norm,
                "crs":      crs,
                "risk":     risk,
                "factor":   factor,
                "gates":    gas,
                "proj_crs": proj_crs,
                "crr":      crr,
                "boxes":    zone_boxes_raw.get(zname, []),
            }
            zone_risk_levels[zname] = risk

            # Log
            logger.log(
                source_frame_idx, zname, zid,
                zcount, density, d_norm, s_norm, c_norm,
                crs, risk, factor, gate_summary,
                proj_crs, crr, ""
            )

        # ── Resource check ────────────────────────────────────────────────
        resource_status = resources.check(zone_risk_levels, zone_counts)

        # Update log resource status
        for rec in logger.records[-len(zone_mgr.zones):]:
            rec["resource_status"] = resource_status["status"]

        # ── GRS ───────────────────────────────────────────────────────────
        crs_values = [zone_results[z]["crs"] for z in zone_results]
        grs_val    = sum(crs_values) / len(crs_values) if crs_values else 0.0
        grs_data   = {"grs": round(grs_val, 4)}

        # ── Visualize ─────────────────────────────────────────────────────
        frame = zone_mgr.draw_zones(frame, zone_risk_levels)
        out   = visualizer.draw_single_zone(
            frame, boxes, zone_results,
            resource_status, grs_data
        )
        out = visualizer.draw_architecture_text(out)

        # FPS overlay
        elapsed  = time.time() - start_time
        fps_val  = processed_frames / elapsed if elapsed > 0 else 0
        cv2.putText(out, f"FPS:{fps_val:.1f}  Src:{source_frame_idx}  Proc:{processed_frames}",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 180, 180), 1, cv2.LINE_AA)

        if writer:
            writer.write(out)

        if not args.no_display:
            cv2.imshow("CA-CRS+ | Multi-Zone", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ss_path = os.path.join(
                    args.output_dir,
                    f"screenshot_frame{source_frame_idx:05d}.png"
                )
                cv2.imwrite(ss_path, out)
                print(f"[INFO] Screenshot saved: {ss_path}")
        elif processed_frames % 25 == 0:
            print(
                f"[INFO] Processed {processed_frames} sampled frames "
                f"(source frame {source_frame_idx})"
            )

        prev_gray = gray
        ret, frame = cap.read()
        if not ret:
            break
        source_frame_idx += 1

    # ── Save results ──────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logger.save_csv()
    logger.plot_crs_timeline()
    logger.plot_components()
    logger.save_summary()
    save_run_metadata(
        args.output_dir,
        args,
        args.input,
        source_fps,
        input_total_frames,
        effective_stride,
        processed_frames,
        time.time() - start_time,
        density_est,
    )

    print(f"[INFO] All results saved to: {args.output_dir}/")
    print(f"[INFO] Sampled frames processed: {processed_frames}, "
          f"Avg FPS: {processed_frames/max(time.time()-start_time, 1e-6):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CA-CRS+ Crowd Safety System"
    )
    parser.add_argument("--input",      default="0",
                        help="Video path or 0 for webcam")
    parser.add_argument("--marshals",   type=int, default=20)
    parser.add_argument("--medics",     type=int, default=5)
    parser.add_argument("--ambulances", type=int, default=3)
    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--no_display", action="store_true",
                        help="Run batch mode without OpenCV display window")
    parser.add_argument("--frame_stride", type=int, default=0,
                        help="Process every Nth frame; 0 uses target_fps")
    parser.add_argument("--target_fps", type=float, default=8.0,
                        help="Adaptive sampling target FPS when frame_stride=0")
    parser.add_argument("--max_processed_frames", type=int, default=0,
                        help="Stop after this many sampled frames; 0 disables")
    parser.add_argument("--density_conf", type=float, default=0.04,
                        help="YOLO confidence threshold for person detection")
    parser.add_argument("--tile_rows", type=int, default=5,
                        help="Fallback number of vertical tiles")
    parser.add_argument("--tile_cols", type=int, default=4,
                        help="Fallback number of horizontal tiles")
    parser.add_argument("--tile_overlap", type=float, default=0.30,
                        help="Fractional overlap between adjacent tiles")
    parser.add_argument("--tile_size", type=int, default=320,
                        help=("Fixed tile size in original pixels; set <=0 "
                              "to use tile_rows/tile_cols"))
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size per tile")
    parser.add_argument("--model", default="yolov8m.pt",
                        help="YOLO weights path/name, or auto")
    parser.add_argument("--iou_thresh", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--max_det_per_tile", type=int, default=3000,
                        help="Maximum detections returned per tile")
    parser.set_defaults(no_full_frame_pass=True)
    parser.add_argument("--full_frame_pass", dest="no_full_frame_pass",
                        action="store_false",
                        help="Enable extra full-frame YOLO pass (slower)")
    parser.add_argument("--no_full_frame_pass", dest="no_full_frame_pass",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--augment", action="store_true",
                        help="Enable slower YOLO test-time augmentation")
    parser.add_argument("--count_smoothing", type=float, default=0.0,
                        help=("EMA smoothing alpha for corrected zone counts; "
                              "0 disables smoothing"))
    parser.add_argument("--count_correction", type=float, default=1.8,
                        help=("Multiplier from raw YOLO detections to "
                              "estimated crowd count"))
    args = parser.parse_args()
    main(args)
