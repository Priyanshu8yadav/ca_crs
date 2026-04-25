# CA-CRS+

CA-CRS+ is a crowd-risk prototype for zone-wise and multi-camera crowd monitoring using tiled person detection, motion cues, causal risk scoring, gate recommendation, resource checks, and projected post-intervention risk.

This repository is also the artifact repository linked from the conference paper. The paper-facing outputs are already generated and stored here so a reviewer can inspect the exact numbers used in the manuscript without rerunning the full pipeline.

## Start Here

If you are reading this repo from the paper, start with:

- [`paper_artifacts/README.md`](paper_artifacts/README.md)
- [`paper_artifacts/final_metrics.csv`](paper_artifacts/final_metrics.csv)

Those two files map each reported result to the exact output folder and figure file used for the manuscript.

You can also verify the manifest and the paper's secondary claims with:

```bash
python3 tools/check_paper_artifacts.py
```

## Paper-Facing Artifact Map

- Scenario A (single-zone density case):
  - [`paper_runs/scen_a/summary_stats.txt`](paper_runs/scen_a/summary_stats.txt)
  - [`paper_runs/scen_a/results.csv`](paper_runs/scen_a/results.csv)
  - [`paper_runs/scen_a/figures/fig1_crs_timeline.png`](paper_runs/scen_a/figures/fig1_crs_timeline.png)
  - [`paper_runs/scen_a/figures/fig_components.png`](paper_runs/scen_a/figures/fig_components.png)
- Scenario B (single-zone lower-risk case):
  - [`paper_runs/scen_b/summary_stats.txt`](paper_runs/scen_b/summary_stats.txt)
  - [`paper_runs/scen_b/results.csv`](paper_runs/scen_b/results.csv)
- Scenario C (single-zone mixed-flow case):
  - [`paper_runs/scen_c/summary_stats.txt`](paper_runs/scen_c/summary_stats.txt)
  - [`paper_runs/scen_c/results.csv`](paper_runs/scen_c/results.csv)
- Scenario D (multi-camera / multi-zone case used in the paper):
  - [`results/multicam/summary_stats.txt`](results/multicam/summary_stats.txt)
  - [`results/multicam/results.csv`](results/multicam/results.csv)
  - [`results/multicam/figures/fig_multizone.png`](results/multicam/figures/fig_multizone.png)
  - [`results/multicam/figures/fig_resources.png`](results/multicam/figures/fig_resources.png)
- Supplementary speed-validation sequence (used to validate the speed pathway, not part of the main 4-row scenario table):
  - [`results/scen_d_probe3/summary_stats.txt`](results/scen_d_probe3/summary_stats.txt)
  - [`results/scen_d_probe3/results.csv`](results/scen_d_probe3/results.csv)
  - [`results/scen_d_probe3/figures/fig_speed_gate.png`](results/scen_d_probe3/figures/fig_speed_gate.png)

## Reproducing the Main Runs

The repository does not version raw input videos or YOLO weights. To rerun the
paper-facing experiments locally, place the scenario videos and model weights
in the repo root. The single-video scenarios use a full-frame single-zone
layout; only the multi-camera experiment uses one zone per camera. Commands:

```bash
python3 main.py --input scen_a.mp4 --output_dir paper_runs/scen_a --no_display --skip_builtin_figures --model yolov8m.pt --imgsz 640 --tile_size 320 --target_fps 2 --count_correction 1.8 --zone_layout full_frame --zone_area_m2 140.0 --zone_capacity 980 --zone_name "Entry Corridor" --zone_gates Gate-Exit-Main Gate-Exit-Secondary
python3 main.py --input scen_b.mp4 --output_dir paper_runs/scen_b --no_display --skip_builtin_figures --model yolov8m.pt --imgsz 640 --tile_size 320 --target_fps 2 --count_correction 1.8 --zone_layout full_frame --zone_area_m2 140.0 --zone_capacity 980 --zone_name "Main Hall" --zone_gates Gate-Main-A Gate-Main-B
python3 main.py --input scen_c.mp4 --output_dir paper_runs/scen_c --no_display --skip_builtin_figures --model yolov8m.pt --imgsz 640 --tile_size 320 --target_fps 2 --count_correction 1.8 --zone_layout full_frame --zone_area_m2 140.0 --zone_capacity 980 --zone_name "Exit Plaza" --zone_gates Gate-Plaza-A Gate-Plaza-B
python3 main_multicamera.py --cameras scen_a.mp4 scen_b.mp4 scen_c.mp4 --output_dir results/multicam --model yolov8s.pt --imgsz 512 --tile_size 320 --target_fps 1 --count_correction 1.8
```

For the supplementary panic-speed validation:

```bash
python3 main.py --input scen_d.mp4 --output_dir results/scen_d_probe3 --no_display --skip_builtin_figures --model yolov8s.pt --imgsz 512 --tile_size 320 --target_fps 4 --count_correction 1.0 --zone_layout full_frame --zone_area_m2 140.0 --zone_capacity 980 --zone_name "Panic Flow" --zone_gates Gate-Entry-Main Gate-Entry-Secondary
python3 generate_paper_figures.py --csv results/scen_d_probe3/results.csv --panic_zone "Panic Flow" --outdir results/scen_d_probe3/figures --scenario "Scenario D (Panic Response)"
```

## Notes on Scope

- The conference paper is page-limited, so not every generated figure is included in the manuscript.
- The main paper table reports Scenarios A, B, C, and the multi-camera Scenario D.
- The speed-validation sequence is kept as a supplementary artifact to show the speed-driven pathway and `CLOSE ENTRY` behavior.
- Smoke runs, debug exports, and exploratory outputs have been removed from the tracked reviewer-facing tree. The files listed in `paper_artifacts/` are the curated sources of truth for the manuscript.

## Audit Notes

- The verified paper-facing notes live in [`paper_artifacts/claim_audit.md`](paper_artifacts/claim_audit.md).
- For Scenario D, the M6 weighted GRS derived from `results/multicam/results.csv` is `0.5438`.
- The corresponding unweighted mean of the per-frame zone averages is `0.5407`, which is distinct from the danger-weighted GRS defined by the orchestrator equation.
- For the supplementary speed-validation run, see [`paper_artifacts/claim_audit.md`](paper_artifacts/claim_audit.md) for the current verified sampled-frame and record counts.
