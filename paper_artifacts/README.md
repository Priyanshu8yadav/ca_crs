# Paper Artifact Guide

This folder maps the conference-paper claims to the exact generated files in this repository.

## Main Scenario Table

The manuscript reports four primary rows:

| Paper scenario | Output folder | Summary file | Notes |
| --- | --- | --- | --- |
| A (Entry Corridor) | [`../paper_runs/scen_a`](../paper_runs/scen_a) | [`../paper_runs/scen_a/summary_stats.txt`](../paper_runs/scen_a/summary_stats.txt) | density-focused full-frame single-zone case |
| B (Main Hall) | [`../paper_runs/scen_b`](../paper_runs/scen_b) | [`../paper_runs/scen_b/summary_stats.txt`](../paper_runs/scen_b/summary_stats.txt) | lower-danger full-frame single-zone case |
| C (Exit Plaza) | [`../paper_runs/scen_c`](../paper_runs/scen_c) | [`../paper_runs/scen_c/summary_stats.txt`](../paper_runs/scen_c/summary_stats.txt) | mixed-flow full-frame single-zone case |
| D (Multi-Cam) | [`../results/multicam`](../results/multicam) | [`../results/multicam/summary_stats.txt`](../results/multicam/summary_stats.txt) | three-stream multi-zone orchestration case |

The exact reported values are maintained in [`final_metrics.csv`](final_metrics.csv).

To re-check the manifest and the derived paper claims against the stored files:

```bash
python3 tools/check_paper_artifacts.py
```

The human-readable audit summary is in [`claim_audit.md`](claim_audit.md).

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

Interpretation notes:

- The paper-facing speed validation uses a full-frame single-zone layout named `Panic Flow`.
- Exact verified counts, dominant-factor percentages, and gate-action totals are recorded in [`claim_audit.md`](claim_audit.md).
- This sequence is kept outside the main four-row scenario table and used to validate the speed-driven pathway.

## Important Interpretation Note

In this repository:

- `scen_d.mp4` is the supplementary panic-speed validation clip.
- The manuscript's `Scenario D (Multi-Cam)` refers to the three-stream run in [`../results/multicam`](../results/multicam), not to `scen_d.mp4`.

That distinction is intentional and matches the corrected manuscript text.

## GRS Interpretation Note

- The M6 orchestrator equation yields a weighted mean `GRS = 0.5438` on Scenario D.
- The value `0.5407` is the unweighted mean of the per-frame zone averages.
- If the manuscript labels the quantity as `GRS`, the weighted `0.5438` value is the consistent one to cite.
