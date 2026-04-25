#!/usr/bin/env python3
"""
Delete and regenerate the five paper-facing CA-CRS+ scenario outputs.

This script rebuilds:
  - paper_runs/scen_a
  - paper_runs/scen_b
  - paper_runs/scen_c
  - results/multicam
  - results/scen_d_probe3

It uses the same settings documented in the paper-facing README.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_checked(cmd: list[str]) -> None:
    print("[rerun_paper_scenarios] " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def remove_path(path: Path) -> None:
    if path.is_dir():
        print(f"[rerun_paper_scenarios] Removing directory: {path}")
        shutil.rmtree(path)
    elif path.exists():
        print(f"[rerun_paper_scenarios] Removing file: {path}")
        path.unlink()


def regenerate_single_zone(
    name: str,
    video: str,
    zone_name: str,
    zone_gates: tuple[str, str],
) -> None:
    outdir = ROOT / "paper_runs" / name
    run_checked([
        PYTHON,
        str(ROOT / "main.py"),
        "--input", str(ROOT / video),
        "--output_dir", str(outdir),
        "--no_display",
        "--skip_builtin_figures",
        "--model", "yolov8m.pt",
        "--imgsz", "640",
        "--tile_size", "320",
        "--target_fps", "2",
        "--count_correction", "1.8",
        "--zone_layout", "full_frame",
        "--zone_area_m2", "140.0",
        "--zone_capacity", "980",
        "--zone_name", zone_name,
        "--zone_gates", zone_gates[0], zone_gates[1],
    ])
    run_checked([
        PYTHON,
        str(ROOT / "generate_paper_figures.py"),
        "--csv", str(outdir / "results.csv"),
        "--outdir", str(outdir / "figures"),
        "--scenario", zone_name,
    ])


def regenerate_multicam() -> None:
    outdir = ROOT / "results" / "multicam"
    run_checked([
        PYTHON,
        str(ROOT / "main_multicamera.py"),
        "--cameras",
        str(ROOT / "scen_a.mp4"),
        str(ROOT / "scen_b.mp4"),
        str(ROOT / "scen_c.mp4"),
        "--output_dir", str(outdir),
        "--no_display",
        "--skip_builtin_figures",
        "--model", "yolov8s.pt",
        "--imgsz", "512",
        "--tile_size", "320",
        "--target_fps", "1",
        "--count_correction", "1.8",
    ])
    run_checked([
        PYTHON,
        str(ROOT / "generate_paper_figures.py"),
        "--csv", str(outdir / "results.csv"),
        "--multizone",
        "--resources",
        "--outdir", str(outdir / "figures"),
        "--scenario", "Scenario D (Multi-Cam)",
    ])
    remove_path(outdir / "figures" / "fig1_crs_timeline.png")
    remove_path(outdir / "figures" / "fig_components.png")


def regenerate_speed_validation() -> None:
    outdir = ROOT / "results" / "scen_d_probe3"
    run_checked([
        PYTHON,
        str(ROOT / "main.py"),
        "--input", str(ROOT / "scen_d.mp4"),
        "--output_dir", str(outdir),
        "--no_display",
        "--skip_builtin_figures",
        "--model", "yolov8s.pt",
        "--imgsz", "512",
        "--tile_size", "320",
        "--target_fps", "4",
        "--count_correction", "1.0",
        "--zone_layout", "full_frame",
        "--zone_area_m2", "140.0",
        "--zone_capacity", "980",
        "--zone_name", "Panic Flow",
        "--zone_gates", "Gate-Entry-Main", "Gate-Entry-Secondary",
    ])
    run_checked([
        PYTHON,
        str(ROOT / "generate_paper_figures.py"),
        "--csv", str(outdir / "results.csv"),
        "--panic_zone", "Panic Flow",
        "--outdir", str(outdir / "figures"),
        "--scenario", "Scenario D (Panic Response)",
    ])
    remove_path(outdir / "figures" / "fig1_crs_timeline.png")
    remove_path(outdir / "figures" / "fig_components.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete and regenerate the paper-facing CA-CRS+ scenario outputs.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["scen_a", "scen_b", "scen_c", "multicam", "speed"],
        default=["scen_a", "scen_b", "scen_c", "multicam", "speed"],
        help="Subset of paper-facing scenario outputs to regenerate",
    )
    args = parser.parse_args()
    selected = set(args.scenarios)

    cleanup_targets = [
        *([
            ROOT / "paper_runs" / "scen_a",
        ] if "scen_a" in selected else []),
        *([
            ROOT / "paper_runs" / "scen_b",
        ] if "scen_b" in selected else []),
        *([
            ROOT / "paper_runs" / "scen_c",
        ] if "scen_c" in selected else []),
        *([
            ROOT / "paper_runs" / "aggregate_overview.png",
            ROOT / "paper_runs" / "aggregate_summary.csv",
            ROOT / "paper_runs" / "aggregate_summary.txt",
        ] if any(name in selected for name in ("scen_a", "scen_b", "scen_c")) else []),
        *([
            ROOT / "results" / "multicam",
        ] if "multicam" in selected else []),
        *([
            ROOT / "results" / "scen_d_probe3",
        ] if "speed" in selected else []),
    ]
    for target in cleanup_targets:
        remove_path(target)

    if "scen_a" in selected:
        regenerate_single_zone(
            "scen_a",
            "scen_a.mp4",
            "Entry Corridor",
            ("Gate-Exit-Main", "Gate-Exit-Secondary"),
        )
    if "scen_b" in selected:
        regenerate_single_zone(
            "scen_b",
            "scen_b.mp4",
            "Main Hall",
            ("Gate-Main-A", "Gate-Main-B"),
        )
    if "scen_c" in selected:
        regenerate_single_zone(
            "scen_c",
            "scen_c.mp4",
            "Exit Plaza",
            ("Gate-Plaza-A", "Gate-Plaza-B"),
        )

    if any(name in selected for name in ("scen_a", "scen_b", "scen_c")):
        inputs = []
        if "scen_a" in selected:
            inputs.append(str(ROOT / "scen_a.mp4"))
        if "scen_b" in selected:
            inputs.append(str(ROOT / "scen_b.mp4"))
        if "scen_c" in selected:
            inputs.append(str(ROOT / "scen_c.mp4"))
        run_checked([
            PYTHON,
            str(ROOT / "tools" / "run_paper_batch.py"),
            "--inputs",
            *inputs,
            "--outdir", str(ROOT / "paper_runs"),
            "--model", "yolov8m.pt",
            "--imgsz", "640",
            "--tile_size", "320",
            "--target_fps", "2",
            "--count_correction", "1.8",
            "--zone_layout", "full_frame",
            "--zone_area_m2", "140.0",
            "--zone_capacity", "980",
            "--skip_existing",
        ])

    if "multicam" in selected:
        regenerate_multicam()
    if "speed" in selected:
        regenerate_speed_validation()


if __name__ == "__main__":
    main()
