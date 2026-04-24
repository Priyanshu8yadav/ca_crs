# Paper Artifact Guide

This folder maps the conference-paper claims to the exact generated files in this repository.

## Main Scenario Table

The manuscript reports four primary rows:

| Paper scenario | Output folder | Summary file | Reported values |
| --- | --- | --- | --- |
| A (Entry Corridor) | [`../paper_runs/scen_a`](../paper_runs/scen_a) | [`../paper_runs/scen_a/summary_stats.txt`](../paper_runs/scen_a/summary_stats.txt) | 44 frames, 81.8% DANGER, mean CRS 0.666, mean CRR 29.0% |
| B (Main Hall) | [`../paper_runs/scen_b`](../paper_runs/scen_b) | [`../paper_runs/scen_b/summary_stats.txt`](../paper_runs/scen_b/summary_stats.txt) | 64 frames, 1.6% DANGER, mean CRS 0.492, mean CRR 21.6% |
| C (Exit Plaza) | [`../paper_runs/scen_c`](../paper_runs/scen_c) | [`../paper_runs/scen_c/summary_stats.txt`](../paper_runs/scen_c/summary_stats.txt) | 54 frames, 9.3% DANGER, mean CRS 0.498, mean CRR 22.5% |
| D (Multi-Cam) | [`../results/multicam`](../results/multicam) | [`../results/multicam/summary_stats.txt`](../results/multicam/summary_stats.txt) | 34 frames, 14.7% DANGER, mean CRS 0.588, mean CRR 23.8% |

The same values are listed in [`final_metrics.csv`](final_metrics.csv).

To re-check the manifest against the stored summary files:

```bash
python3 tools/check_paper_artifacts.py
```

## Figures Used in the Paper

- Fig. 1 timeline:
  - [`../paper_runs/scen_a/figures/fig1_crs_timeline.png`](../paper_runs/scen_a/figures/fig1_crs_timeline.png)
- Component decomposition:
  - [`../paper_runs/scen_a/figures/fig_components.png`](../paper_runs/scen_a/figures/fig_components.png)
- Multi-zone risk differentiation:
  - [`../results/multicam/figures/fig_multizone.png`](../results/multicam/figures/fig_multizone.png)
- Resource demand vs capacity:
  - [`../results/multicam/figures/fig_resources.png`](../results/multicam/figures/fig_resources.png)

## Supplementary Speed Validation

The page-limited manuscript does not place this sequence in the main four-row table, but it is the supporting evidence for the speed-driven pathway:

- Output folder:
  - [`../results/scen_d_probe3`](../results/scen_d_probe3)
- Summary:
  - [`../results/scen_d_probe3/summary_stats.txt`](../results/scen_d_probe3/summary_stats.txt)
- Key figure:
  - [`../results/scen_d_probe3/figures/fig_speed_gate.png`](../results/scen_d_probe3/figures/fig_speed_gate.png)

Key verified supplementary facts:

- `SPEED` is dominant in 98.4% of records.
- `S_norm` reaches 1.0 at peak frames.
- The system issues repeated `CLOSE` recommendations in the Entry Zone.
- This sequence contains WARNING-level speed spikes rather than DANGER-level density collapse.

## Important Interpretation Note

In this repository:

- `scen_d.mp4` is the supplementary panic-speed validation clip.
- The manuscript's `Scenario D (Multi-Cam)` refers to the three-stream run in [`../results/multicam`](../results/multicam), not to `scen_d.mp4`.

That distinction is intentional and matches the corrected manuscript text.
