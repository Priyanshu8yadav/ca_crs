"""
logger.py
─────────
Logs per-frame results to CSV and generates all paper figures.

Paper figures generated:
  fig1_crs_timeline.png    — CA-CRS+ score over time (all scenarios)
  fig_multizone.png        — Per-zone scores, Scenario D
  fig_resources.png        — Marshal demand vs. available capacity
  fig_components.png       — Component contributions over time
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from collections import defaultdict


class ResultLogger:

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records = []

    # ── Logging ───────────────────────────────────────────────────────────

    def log(self, frame_idx, zone_name, zone_id,
            count, density, d_norm, s_norm, c_norm,
            crs, risk, factor, gate_summary,
            proj_crs, crr, resource_status):
        self.records.append({
            "frame":           frame_idx,
            "zone_name":       zone_name,
            "zone_id":         zone_id,
            "count":           count,
            "density":         round(density, 3),
            "d_norm":          round(d_norm, 4),
            "s_norm":          round(s_norm, 4),
            "c_norm":          round(c_norm, 4),
            "crs":             crs,
            "risk":            risk,
            "factor":          factor,
            "gate_summary":    gate_summary,
            "proj_crs":        proj_crs,
            "crr":             crr,
            "resource_status": resource_status,
        })

    def save_csv(self):
        if not self.records:
            print("[Logger] No data to save.")
            return
        path = os.path.join(self.output_dir, "results.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.records[0].keys()
            )
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[Logger] CSV saved → {path}")

    # ── Figure generation ─────────────────────────────────────────────────

    def _records_by_frame(self):
        grouped = defaultdict(list)
        for record in self.records:
            grouped[int(record["frame"])].append(record)
        return grouped

    def plot_crs_timeline(self, scenario_label="Scenarios A-C"):
        """Fig 1: CA-CRS+ score over time."""
        if not self.records:
            return

        grouped = self._records_by_frame()
        frames = sorted(grouped)
        crs = [max(r["crs"] for r in grouped[frame]) for frame in frames]
        proj = [max(r["proj_crs"] for r in grouped[frame]) for frame in frames]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(frames, crs,  color="#E74C3C", lw=1.8, label="CA-CRS+ (before)")
        ax.plot(frames, proj, color="#2ECC71", lw=1.5, ls="--",
                label="Projected CRS (after action)")
        ax.axhline(0.70, color="red",    lw=1.2, ls=":",
                   label="DANGER threshold (0.70)")
        ax.axhline(0.35, color="orange", lw=1.0, ls=":",
                   label="WARNING threshold (0.35)")
        ax.fill_between(frames, 0.70, 1.0,  alpha=0.07, color="red")
        ax.fill_between(frames, 0.35, 0.70, alpha=0.07, color="orange")
        ax.fill_between(frames, 0.0,  0.35, alpha=0.05, color="green")

        ax.set_xlabel("Frame", fontsize=11)
        ax.set_ylabel("CA-CRS+ Score", fontsize=11)
        ax.set_title(f"CA-CRS+ Score Over Time — {scenario_label}")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig1_crs_timeline.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Logger] Saved {path}")

    def plot_multizone(self, zone_records_dict: dict):
        """
        Fig 2: Per-zone CA-CRS+ scores over time.

        Parameters
        ----------
        zone_records_dict : {zone_name: [crs_values]}
        """
        colors = ["#E74C3C", "#3498DB", "#2ECC71",
                  "#9B59B6", "#E67E22"]
        fig, ax = plt.subplots(figsize=(9, 3.5))

        for i, (zname, crs_list) in enumerate(
            zone_records_dict.items()
        ):
            c = colors[i % len(colors)]
            ax.plot(crs_list, label=zname, color=c, lw=1.8)

        ax.axhline(0.70, color="red",    lw=1.2, ls=":",
                   label="DANGER (0.70)")
        ax.axhline(0.35, color="orange", lw=1.0, ls=":",
                   label="WARNING (0.35)")
        ax.fill_between(range(max(len(v) for v in zone_records_dict.values())),
                        0.70, 1.0, alpha=0.06, color="red")
        ax.set_xlabel("Frame", fontsize=11)
        ax.set_ylabel("CA-CRS+ Score", fontsize=11)
        ax.set_title("Multi-Zone CA-CRS+ Score Evolution (Scenario D)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig_multizone.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Logger] Saved {path}")

    def plot_resources(self, demand_list: list,
                       available: int,
                       status_list: list):
        """
        Fig 3: Marshal demand vs. available capacity with status band.

        Parameters
        ----------
        demand_list  : [int] — marshal demand per frame
        available    : int  — available marshal count (constant)
        status_list  : ["ADEQUATE"|"STRAINED"|"CRITICAL"] per frame
        """
        frames = list(range(len(demand_list)))
        status_map = {"ADEQUATE": 0, "STRAINED": 1, "CRITICAL": 2}
        status_vals = [status_map.get(s, 0) for s in status_list]
        res_colors  = ["#2ECC71", "#E67E22", "#E74C3C"]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 4), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )

        ax1.plot(frames, demand_list, color="#E74C3C",
                 lw=1.8, label="Marshal demand")
        ax1.axhline(available, color="#2ECC71",
                    lw=1.8, ls="--",
                    label=f"Available ({available})")
        ax1.set_ylabel("Marshals Required", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(demand_list) * 1.2 + 2)

        for i in range(len(frames) - 1):
            ax2.axvspan(i, i + 1, alpha=0.85,
                        color=res_colors[status_vals[i]])
        ax2.set_yticks([])
        ax2.set_xlabel("Frame", fontsize=10)
        ax2.set_ylabel("Status", fontsize=9)

        patches = [
            mpatches.Patch(color=res_colors[0], label="Adequate"),
            mpatches.Patch(color=res_colors[1], label="Strained"),
            mpatches.Patch(color=res_colors[2], label="Critical"),
        ]
        ax2.legend(handles=patches, fontsize=8,
                   loc="upper right", ncol=3)

        fig.suptitle("Resource Adequacy vs. Multi-Zone Risk Level",
                     fontsize=11)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig_resources.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Logger] Saved {path}")

    def plot_components(self):
        """Fig 4: Weighted component contributions over time."""
        if not self.records:
            return
        grouped = self._records_by_frame()
        frames = sorted(grouped)
        d_vals = [
            np.mean([r["d_norm"] * 0.40 for r in grouped[frame]])
            for frame in frames
        ]
        s_vals = [
            np.mean([r["s_norm"] * 0.30 for r in grouped[frame]])
            for frame in frames
        ]
        c_vals = [
            np.mean([r["c_norm"] * 0.30 for r in grouped[frame]])
            for frame in frames
        ]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.stackplot(frames, d_vals, s_vals, c_vals,
                     labels=["Density (w₁·D)", "Speed (Φ)", "Conflict (w₃·C)"],
                     colors=["#3498DB", "#E67E22", "#9B59B6"],
                     alpha=0.80)
        ax.set_xlabel("Frame", fontsize=11)
        ax.set_ylabel("Weighted Contribution", fontsize=11)
        ax.set_title("CA-CRS+ Component Contributions Over Time")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "fig_components.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Logger] Saved {path}")

    def save_summary(self):
        """Save plain-text summary for paper Table III values."""
        if not self.records:
            return
        path = os.path.join(self.output_dir, "summary_stats.txt")
        unique_frames = sorted({int(r["frame"]) for r in self.records})
        danger_records  = [r for r in self.records if r["risk"] == "DANGER"]
        danger_frames = sorted({int(r["frame"]) for r in danger_records})
        total_records   = len(self.records)
        mean_crs        = np.mean([r["crs"]      for r in self.records])
        mean_proj       = np.mean([r["proj_crs"]  for r in self.records])
        crr_values      = [r["crr"] for r in self.records]
        pos_crr_values  = [v for v in crr_values if v > 0]
        mean_crr        = np.mean(crr_values) if crr_values else 0.0
        mean_pos_crr    = np.mean(pos_crr_values) if pos_crr_values else 0.0
        improved        = [r for r in self.records
                           if r["proj_crs"] < r["crs"]]

        factor_counts = Counter(r["factor"] for r in danger_records)

        with open(path, "w") as f:
            f.write("=== CA-CRS+ Summary Statistics ===\n\n")
            f.write(f"Total records      : {total_records}\n")
            f.write(f"Unique frames      : {len(unique_frames)}\n")
            f.write(f"DANGER records     : {len(danger_records)} "
                    f"({100*len(danger_records)/max(total_records,1):.1f}%)\n")
            f.write(f"DANGER frames      : {len(danger_frames)} "
                    f"({100*len(danger_frames)/max(len(unique_frames),1):.1f}%)\n")
            f.write(f"Mean CA-CRS+       : {mean_crs:.4f}\n")
            f.write(f"Mean Projected CRS : {mean_proj:.4f}\n")
            f.write(f"Mean CRR (%)       : {mean_crr:.1f}%\n")
            f.write(f"Mean positive CRR  : {mean_pos_crr:.1f}%\n")
            f.write(f"Improved projections: {len(improved)} "
                    f"({100*len(improved)/max(total_records,1):.1f}%)\n\n")
            f.write("Dominant factor breakdown (DANGER records):\n")
            for factor, cnt in factor_counts.most_common():
                pct = 100 * cnt / max(len(danger_records), 1)
                f.write(f"  {factor:<12}: {cnt:4d} ({pct:.1f}%)\n")

        print(f"[Logger] Summary saved → {path}")
