"""
main_multicamera.py
────────────────────
CA-CRS+ multi-camera pipeline.

Each camera feed is processed as an independent zone.
A central orchestrator aggregates zone scores into
a Global Risk Score (GRS) and performs triage.

Usage:
    python main_multicamera.py --cameras cam1.mp4 cam2.mp4 cam3.mp4
    python main_multicamera.py --cameras 0 1 2       # webcams
"""

import cv2
import threading
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.density_module   import DensityEstimator
from modules.motion_module    import MotionAnalyzer
from modules.risk_scoring     import CACRSScorer
from modules.gate_recommender import GateRecommender
from modules.resource_manager import ResourceManager
from modules.simulation       import InterventionSimulator
from modules.orchestrator     import VenueOrchestrator
from modules.logger           import ResultLogger


# ── Camera zone config for multi-camera mode ──────────────────────────────
CAMERA_ZONE_CONFIG = [
    {
        "id": 0, "name": "Entry Corridor",
        "area_m2": 35.0, "capacity": 245,
        "gates": ["Gate-Main-Entry", "Gate-Side-A"],
        "adjacent_zones": [1],
    },
    {
        "id": 1, "name": "Main Hall",
        "area_m2": 80.0, "capacity": 560,
        "gates": ["Gate-Side-A", "Gate-Center-B"],
        "adjacent_zones": [0, 2],
    },
    {
        "id": 2, "name": "Exit Plaza",
        "area_m2": 45.0, "capacity": 315,
        "gates": ["Gate-Center-B", "Gate-Exit-Main"],
        "adjacent_zones": [1],
    },
]


def determine_frame_stride(frame_stride: int,
                           target_fps: float,
                           source_fps: float) -> int:
    if frame_stride and frame_stride > 1:
        return int(frame_stride)
    if target_fps > 0 and source_fps > 0:
        return max(1, int(round(source_fps / target_fps)))
    return 1


def process_camera(cam_idx: int,
                   source,
                   zone_cfg: dict,
                   orchestrator: VenueOrchestrator,
                   gate_rec: GateRecommender,
                   simulator: InterventionSimulator,
                   logger: ResultLogger,
                   stop_event: threading.Event,
                   frame_store: dict,
                   lock: threading.Lock,
                   density_kwargs: dict = None,
                   run_kwargs: dict = None):
    """
    Thread function: process one camera feed independently.
    Transmits only risk scalars to orchestrator (not raw video).
    """
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Camera {cam_idx}] Cannot open source: {source}")
        return

    run_kwargs = run_kwargs or {}
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    effective_stride = determine_frame_stride(
        run_kwargs.get("frame_stride", 0),
        run_kwargs.get("target_fps", 0.0),
        source_fps,
    )
    max_processed_frames = int(run_kwargs.get("max_processed_frames", 0))

    density_est = DensityEstimator(**(density_kwargs or {}))
    motion_an   = MotionAnalyzer()
    scorer      = CACRSScorer()

    zname    = zone_cfg["name"]
    zid      = zone_cfg["id"]
    zarea    = zone_cfg["area_m2"]
    zgates   = zone_cfg["gates"]
    zadj     = zone_cfg["adjacent_zones"]

    prev_gray  = None
    source_frame_idx = 1
    processed_frames = 0

    ret, frame = cap.read()
    if not ret:
        cap.release()
        print(f"[Camera {cam_idx}] Cannot read first frame from: {source}")
        return

    print(
        f"[Camera {cam_idx}] Processing zone: {zname} "
        f"| model={getattr(density_est, 'loaded_model_name', density_kwargs.get('model_name', '?'))} "
        f"| stride={effective_stride}"
    )

    while not stop_event.is_set():
        if max_processed_frames > 0 and processed_frames >= max_processed_frames:
            break

        if effective_stride > 1 and ((source_frame_idx - 1) % effective_stride != 0):
            ret, frame = cap.read()
            if not ret:
                break
            source_frame_idx += 1
            continue

        processed_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection and density
        count, boxes   = density_est.detect(frame)
        d_norm = density_est.dnorm(count, zarea)
        density = density_est.density(count, zarea)

        # Motion
        if prev_gray is not None:
            s_norm, c_norm, _ = motion_an.analyze(prev_gray, gray)
        else:
            s_norm, c_norm = 0.0, 0.0

        # Scoring
        crs, comps, phi = scorer.score(d_norm, s_norm, c_norm)
        risk            = scorer.classify(crs)
        factor          = scorer.dominant_factor(comps)

        # Get adjacent zone scores from orchestrator (ripple check)
        adj_scores = orchestrator.get_adjacent_crs(zid, zadj)

        # Gate recommendation with ripple check
        gas  = gate_rec.recommend_gates(
            zname, zid, factor, risk, zgates, adj_scores
        )
        gate_summary = gate_rec.format_summary(gas)

        # Simulation
        action     = gas[0]["action"] if gas else "HOLD"
        sim_result = simulator.validate_action(
            d_norm, s_norm, c_norm, action, crs
        )

        # Report to orchestrator (lightweight scalars only)
        orchestrator.update_zone(zid, {
            "crs":          crs,
            "risk_level":   risk,
            "factor":       factor,
            "gate_actions": gas,
            "count":        count,
            "timestamp":    time.time(),
        })

        # Store frame for display
        with lock:
            frame_store[cam_idx] = {
                "frame":    frame,
                "crs":      crs,
                "risk":     risk,
                "factor":   factor,
                "gates":    gate_summary,
                "proj":     sim_result["proj_crs"],
                "crr":      sim_result["crr"],
                "zone":     zname,
            }

        logger.log(
            source_frame_idx, zname, zid,
            count, density, d_norm, s_norm, c_norm,
            crs, risk, factor, gate_summary,
            sim_result["proj_crs"], sim_result["crr"], ""
        )

        prev_gray = gray
        ret, frame = cap.read()
        if not ret:
            break
        source_frame_idx += 1

    cap.release()
    print(
        f"[Camera {cam_idx}] Done. "
        f"Processed {processed_frames} sampled frames."
    )


def main(args):
    sources = args.cameras
    n_cams  = len(sources)

    if n_cams > len(CAMERA_ZONE_CONFIG):
        print(f"[WARN] More cameras than zone configs. "
              f"Using first {len(CAMERA_ZONE_CONFIG)} cameras.")
        sources = sources[:len(CAMERA_ZONE_CONFIG)]
        n_cams  = len(sources)

    zone_ids    = [CAMERA_ZONE_CONFIG[i]["id"] for i in range(n_cams)]
    orchestrator = VenueOrchestrator(zone_ids)
    gate_rec     = GateRecommender()
    simulator    = InterventionSimulator()
    resources    = ResourceManager(
                     marshals=args.marshals,
                     medics=args.medics,
                     ambulances=args.ambulances)
    logger       = ResultLogger(args.output_dir)
    density_kwargs = {
        "conf": args.density_conf,
        "tile_rows": args.tile_rows,
        "tile_cols": args.tile_cols,
        "overlap": args.tile_overlap,
        "imgsz": args.imgsz,
        "model_name": args.model,
        "iou_thresh": args.iou_thresh,
        "count_correction": args.count_correction,
        "max_det_per_tile": args.max_det_per_tile,
        "tile_size": args.tile_size,
        "full_frame_pass": not args.no_full_frame_pass,
        "augment": args.augment,
    }
    run_kwargs = {
        "frame_stride": args.frame_stride,
        "target_fps": args.target_fps,
        "max_processed_frames": args.max_processed_frames,
    }

    stop_event  = threading.Event()
    frame_store = {}
    lock        = threading.Lock()

    # Start one thread per camera
    threads = []
    for i, src in enumerate(sources):
        t = threading.Thread(
            target=process_camera,
            args=(i, src, CAMERA_ZONE_CONFIG[i],
                  orchestrator, gate_rec, simulator,
                  logger, stop_event, frame_store, lock,
                  density_kwargs, run_kwargs),
            daemon=True
        )
        t.start()
        threads.append(t)

    print(f"[INFO] {n_cams} camera threads started. Press Q to quit.")

    # ── Display loop ──────────────────────────────────────────────────────
    last_status_time = 0.0
    while True:
        with lock:
            frames_data = dict(frame_store)
        alive = any(t.is_alive() for t in threads)

        if args.no_display:
            if frames_data and (time.time() - last_status_time) > 5.0:
                grs_data = orchestrator.compute_grs()
                print(
                    f"[INFO] Multi-camera status | GRS:{grs_data.get('grs', 0.0):.3f} "
                    f"| Worst zone: {grs_data.get('worst_zone', '?')} "
                    f"| Triage: {grs_data.get('triage_order', [])}"
                )
                last_status_time = time.time()
            if not alive:
                break
            time.sleep(0.1)
            continue

        if frames_data:
            grs_data = orchestrator.compute_grs()

            # Build composite display
            panels = []
            for i in range(n_cams):
                fd = frames_data.get(i)
                if fd is None:
                    continue
                f = fd["frame"].copy()
                h, w = f.shape[:2]
                # Risk color banner
                from modules.risk_scoring import SAFE, WARNING, DANGER
                RISK_COLORS = {SAFE: (0,200,0),
                               WARNING: (0,165,255),
                               DANGER: (0,0,255)}
                color = RISK_COLORS.get(fd["risk"], (200,200,200))
                cv2.rectangle(f, (0, 0), (w, 30), color, -1)
                cv2.putText(f, f"Cam{i}: {fd['zone']}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(f, f"CRS:{fd['crs']:.3f} [{fd['risk']}]",
                            (5, h-35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)
                cv2.putText(f, f"{fd['gates']}",
                            (5, h-18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (200,200,100), 1, cv2.LINE_AA)
                panels.append(f)

            if panels:
                # Resize to same height, then hstack
                target_h = min(p.shape[0] for p in panels)
                resized  = [cv2.resize(p, (int(p.shape[1]*target_h/p.shape[0]),
                                          target_h))
                            for p in panels]
                composite = cv2.hconcat(resized)

                # GRS banner at bottom
                banner_h = 35
                banner   = composite.copy()
                banner_strip = composite[-banner_h:].copy()
                grs_val  = grs_data.get("grs", 0.0)
                grs_color = (0,0,255) if grs_val >= 0.70 else \
                            (0,165,255) if grs_val >= 0.35 else \
                            (0,200,0)
                triage   = grs_data.get("triage_order", [])
                grs_text = (f"GRS:{grs_val:.3f}  |  "
                            f"Worst zone: {grs_data.get('worst_zone','?')}  |  "
                            f"Triage: {triage}")
                cv2.putText(composite, grs_text,
                            (8, composite.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, grs_color, 1, cv2.LINE_AA)

                cv2.imshow("CA-CRS+ | Multi-Camera Venue", composite)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ss_path = os.path.join(args.output_dir,
                                   f"mc_screenshot_{int(time.time())}.png")
            if frames_data:
                cv2.imwrite(ss_path, composite)
                print(f"[INFO] Screenshot: {ss_path}")
        if not alive:
            break

    stop_event.set()
    for t in threads:
        t.join(timeout=3)
    cv2.destroyAllWindows()

    logger.save_csv()
    if not args.skip_builtin_figures:
        logger.plot_crs_timeline("Multi-Camera Scenarios")
        logger.plot_components()
    logger.save_summary()
    print(f"[INFO] Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras",    nargs="+",
                        default=["0"],
                        help="Camera sources (paths or indices)")
    parser.add_argument("--marshals",   type=int, default=20)
    parser.add_argument("--medics",     type=int, default=5)
    parser.add_argument("--ambulances", type=int, default=3)
    parser.add_argument("--output_dir", default="results/multicam")
    parser.add_argument("--no_display", action="store_true",
                        help="Run batch mode without OpenCV display window")
    parser.add_argument("--skip_builtin_figures", action="store_true",
                        help="Skip top-level logger figure exports")
    parser.add_argument("--frame_stride", type=int, default=0,
                        help="Process every Nth frame per camera; 0 uses target_fps")
    parser.add_argument("--target_fps", type=float, default=2.0,
                        help="Adaptive sampling target FPS per camera")
    parser.add_argument("--max_processed_frames", type=int, default=0,
                        help="Stop after this many sampled frames per camera; 0 disables")
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
    parser.add_argument("--imgsz", type=int, default=512,
                        help="YOLO inference size per tile")
    parser.add_argument("--model", default="yolov8s.pt",
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
    parser.add_argument("--count_correction", type=float, default=1.8,
                        help=("Multiplier from raw YOLO detections to "
                              "estimated crowd count"))
    args = parser.parse_args()
    main(args)
