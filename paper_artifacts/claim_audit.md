# Claim Audit

This note records the repo-side verification of the current paper draft against
the stored artifacts.

## Verified Main-Table Values

The four main-table rows in `final_metrics.csv` match the generated
`summary_stats.txt` files exactly within rounding:

- Scenario A: 44 frames, 75.0% DANGER, mean CRS 0.793, mean CRR 40.4%
- Scenario B: 64 frames, 1.6% DANGER, mean CRS 0.504, mean CRR 18.9%
- Scenario C: 54 frames, 9.3% DANGER, mean CRS 0.529, mean CRR 21.2%
- Scenario D: 59 frames, 10.2% DANGER, mean CRS 0.556, mean CRR 22.7%

## Verified Secondary Claims

### Scenario D (Multi-Cam)

Derived from `results/multicam/results.csv`:

- Weighted mean GRS from the M6 orchestrator equation: `0.5438`
- Unweighted mean of the per-frame zone averages: `0.5407`
- GRS range: `0.4262` to `1.0000`
- Highest-risk zone counts by frame:
  - Exit Plaza: `21 / 59` frames (`35.6%`)
  - Entry Corridor: `19 / 59` frames (`32.2%`)
  - Main Hall: `19 / 59` frames (`32.2%`)
- Resource-demand range: `11` to `93` marshals

Interpretation:

- Entry Corridor still has the highest per-zone mean CRS (`0.667`) and the most
  DANGER records (`5`), but the rerun's framewise "highest-risk zone" split is
  much more balanced than in the previous artifact set.
- The implemented GRS equation in `modules/orchestrator.py` is danger-weighted,
  so `0.5438` is the internally consistent value to cite as mean GRS.
- The older manuscript values no longer describe the current rerun outputs and
  should be replaced.

### Supplementary Speed Validation

Derived from `results/scen_d_probe3/results.csv`:

- Total records: `86`
- Unique sampled frames: `86`
- WARNING records: `25`
- `SPEED` dominant: `98.8%`
- `CLOSE` recommendations issued: `25` records
- Peak `S_norm`: `1.0`

Interpretation:

- The rerun uses a single full-frame zone (`Panic Flow`), so the stored CSV now
  contains one record per sampled frame.
- This artifact provides a cleaner speed-pathway check than the earlier
  three-zone version: the sequence remains below DANGER, but high speed
  repeatedly triggers WARNING-level `CLOSE` actions.

## Figure Check

The paper-facing figure files referenced in the repo are present and visually
consistent with the stored CSV outputs:

- `paper_runs/scen_a/figures/fig1_crs_timeline.png`
- `paper_runs/scen_a/figures/fig_components.png`
- `results/multicam/figures/fig_multizone.png`
- `results/multicam/figures/fig_resources.png`
- `results/scen_d_probe3/figures/fig_speed_gate.png`

## Source-File Note

The draft TeX file in `Downloads` references bare figure filenames. The
paper-facing copies of `fig1_crs_timeline.png`, `fig_components.png`,
`fig_multizone.png`, and `fig_resources.png` have been copied beside the draft
so the Downloads paper folder is now self-contained for LaTeX compilation.
