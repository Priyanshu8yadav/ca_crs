"""
Run the CA-CRS+ pipeline over multiple videos with Mac-friendly defaults,
then generate per-video paper figures plus an aggregate comparison summary.

Example:
    python tools/run_paper_batch.py
    python tools/run_paper_batch.py --count_correction 4.0 --target_fps 8
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SCENARIO_LABELS = {
    "scen_a": "Entry Corridor",
    "scen_b": "Main Hall",
    "scen_c": "Exit Plaza",
}


def existing_default_inputs():
    candidates = [
        ROOT / "scen_a.mp4",
        ROOT / "scen_b.mp4",
        ROOT / "scen_c.mp4",
    ]
    return [str(path) for path in candidates if path.exists()]


def floatf(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def run_checked(cmd):
    print("[run_paper_batch] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def load_records(csv_path):
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))


def frame_max_series(records, field):
    groups = defaultdict(list)
    for record in records:
        groups[int(record["frame"])].append(floatf(record[field]))
    frames = sorted(groups)
    values = [max(groups[frame]) for frame in frames]
    return frames, values


def summarize_csv(video_name, csv_path):
    records = load_records(csv_path)
    unique_frames = sorted({int(record["frame"]) for record in records})
    danger_records = [record for record in records if record["risk"] == "DANGER"]
    danger_frames = sorted({int(record["frame"]) for record in danger_records})
    _, frame_max_crs = frame_max_series(records, "crs")
    _, frame_max_proj = frame_max_series(records, "proj_crs")
    crr_values = [floatf(record["crr"]) for record in records]
    positive_crr = [value for value in crr_values if value > 0]
    label = SCENARIO_LABELS.get(video_name, video_name)

    return {
        "video": label,
        "records": len(records),
        "frames": len(unique_frames),
        "danger_records": len(danger_records),
        "danger_record_pct": 100.0 * len(danger_records) / max(len(records), 1),
        "danger_frames": len(danger_frames),
        "danger_frame_pct": 100.0 * len(danger_frames) / max(len(unique_frames), 1),
        "mean_crs": float(np.mean([floatf(record["crs"]) for record in records])) if records else 0.0,
        "mean_proj_crs": float(np.mean([floatf(record["proj_crs"]) for record in records])) if records else 0.0,
        "mean_frame_max_crs": float(np.mean(frame_max_crs)) if frame_max_crs else 0.0,
        "mean_frame_max_proj_crs": float(np.mean(frame_max_proj)) if frame_max_proj else 0.0,
        "mean_positive_crr": float(np.mean(positive_crr)) if positive_crr else 0.0,
    }


def save_aggregate_summary(rows, outdir):
    csv_path = outdir / "aggregate_summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_paper_batch] Saved {csv_path}")


def save_aggregate_plot(rows, outdir):
    labels = [row["video"] for row in rows]
    danger_pct = [row["danger_frame_pct"] for row in rows]
    mean_crs = [row["mean_frame_max_crs"] for row in rows]
    mean_proj = [row["mean_frame_max_proj_crs"] for row in rows]

    x = np.arange(len(labels))
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1.bar(x, danger_pct, color="#C0392B", width=0.6)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Danger Frames (%)")
    ax1.set_title("Danger Frame Share by Video")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x - width / 2, mean_crs, width=width, color="#C0392B", label="CA-CRS+")
    ax2.bar(x + width / 2, mean_proj, width=width, color="#27AE60", label="Projected")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Mean Frame-Max CRS")
    ax2.set_title("Before vs. Projected Risk")
    ax2.legend(framealpha=0.85)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / "aggregate_overview.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[run_paper_batch] Saved {path}")


def main(args):
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for video in args.inputs:
        video_path = Path(video).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Missing input video: {video_path}")

        stem = video_path.stem
        run_dir = outdir / stem
        csv_path = run_dir / "results.csv"

        if args.skip_existing and csv_path.exists():
            print(f"[run_paper_batch] Skipping existing run for {stem}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(ROOT / "main.py"),
                "--input", str(video_path),
                "--output_dir", str(run_dir),
                "--no_display",
                "--skip_builtin_figures",
                "--density_conf", str(args.density_conf),
                "--tile_overlap", str(args.tile_overlap),
                "--tile_size", str(args.tile_size),
                "--imgsz", str(args.imgsz),
                "--model", args.model,
                "--count_correction", str(args.count_correction),
                "--target_fps", str(args.target_fps),
                "--marshals", str(args.marshals),
                "--medics", str(args.medics),
                "--ambulances", str(args.ambulances),
                "--zone_layout", args.zone_layout,
                "--zone_area_m2", str(args.zone_area_m2),
                "--zone_capacity", str(args.zone_capacity),
            ]
            if args.frame_stride > 0:
                cmd.extend(["--frame_stride", str(args.frame_stride)])
            if args.max_processed_frames > 0:
                cmd.extend(["--max_processed_frames", str(args.max_processed_frames)])
            if args.full_frame_pass:
                cmd.append("--full_frame_pass")
            if args.augment:
                cmd.append("--augment")
            if args.save_output_video:
                cmd.append("--save_output")
            run_checked(cmd)

        fig_dir = run_dir / "figures"
        scenario_label = SCENARIO_LABELS.get(
            stem, stem.replace("_", " ").title()
        )

        fig_cmd = [
            sys.executable,
            str(ROOT / "generate_paper_figures.py"),
            "--csv", str(csv_path),
            "--outdir", str(fig_dir),
            "--scenario", scenario_label,
            "--marshals", str(args.marshals),
        ]
        run_checked(fig_cmd)

        summaries.append(summarize_csv(stem, csv_path))

    if summaries:
        save_aggregate_summary(summaries, outdir)
        save_aggregate_plot(summaries, outdir)

        report_path = outdir / "aggregate_summary.txt"
        with open(report_path, "w") as handle:
            handle.write("=== CA-CRS+ Batch Summary ===\n\n")
            for row in summaries:
                handle.write(f"{row['video']}\n")
                handle.write(f"  Sampled frames    : {row['frames']}\n")
                handle.write(
                    f"  Danger frames     : {row['danger_frames']} "
                    f"({row['danger_frame_pct']:.1f}%)\n"
                )
                handle.write(
                    f"  Mean frame-max CRS: {row['mean_frame_max_crs']:.4f}\n"
                )
                handle.write(
                    f"  Mean proj. CRS    : {row['mean_frame_max_proj_crs']:.4f}\n"
                )
                handle.write(
                    f"  Mean positive CRR : {row['mean_positive_crr']:.1f}%\n\n"
                )
        print(f"[run_paper_batch] Saved {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CA-CRS+ on multiple videos and build paper outputs",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=existing_default_inputs(),
        help="Video files to process",
    )
    parser.add_argument("--outdir", default="paper_runs")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--density_conf", type=float, default=0.04)
    parser.add_argument("--tile_overlap", type=float, default=0.30)
    parser.add_argument("--tile_size", type=int, default=320)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--count_correction", type=float, default=1.8)
    parser.add_argument("--frame_stride", type=int, default=0)
    parser.add_argument("--target_fps", type=float, default=8.0)
    parser.add_argument("--max_processed_frames", type=int, default=0)
    parser.add_argument("--marshals", type=int, default=20)
    parser.add_argument("--medics", type=int, default=5)
    parser.add_argument("--ambulances", type=int, default=3)
    parser.add_argument("--zone_layout", choices=["full_frame", "three_strip"],
                        default="full_frame")
    parser.add_argument("--zone_area_m2", type=float, default=140.0)
    parser.add_argument("--zone_capacity", type=int, default=980)
    parser.add_argument("--save_output_video", action="store_true")
    parser.add_argument("--full_frame_pass", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    main(parser.parse_args())
