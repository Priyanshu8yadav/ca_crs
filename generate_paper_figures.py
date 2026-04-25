"""
generate_paper_figures.py
──────────────────────────
Run AFTER main.py or main_multicamera.py to generate
all paper-quality figures from real experimental data.

Usage:
    python generate_paper_figures.py --csv results/results.csv
    python generate_paper_figures.py --csv results/results.csv --multizone
"""

import argparse
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter

plt.rcParams.update({
    "font.family":    "DejaVu Serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#FCFCFC",
    "figure.facecolor": "white",
})

# ── Load CSV ──────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def floatf(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def frame_groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[int(record["frame"])].append(record)
    return groups


def style_axes(ax):
    ax.grid(True, alpha=0.26, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved] {path}")


# ── Figure 1: CA-CRS+ timeline ────────────────────────────────────────────

def fig1_crs_timeline(records, scenario_name="Real Crowd Video", outdir="results/paper_figures"):
    groups = frame_groups(records)
    frames = sorted(groups)
    crs = [max(floatf(r["crs"]) for r in groups[frame]) for frame in frames]
    proj = [max(floatf(r["proj_crs"]) for r in groups[frame]) for frame in frames]
    peak_idx = int(np.argmax(crs)) if crs else 0

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(frames, crs,  color="#C0392B", lw=1.8, label="CA-CRS+ score")
    ax.plot(frames, proj, color="#27AE60", lw=1.5, ls="--",
            label="Projected post-intervention score")
    ax.axhline(0.70, color="#C0392B", lw=1.1, ls=":",
               alpha=0.85, label="DANGER threshold (0.70)")
    ax.axhline(0.35, color="#E67E22", lw=1.0, ls=":",
               alpha=0.85, label="WARNING threshold (0.35)")
    ax.fill_between(frames, 0.70, 1.0,  alpha=0.06, color="#E74C3C")
    ax.fill_between(frames, 0.35, 0.70, alpha=0.06, color="#E67E22")
    ax.fill_between(frames, 0.0,  0.35, alpha=0.06, color="#27AE60")
    ax.set_xlabel("Frame")
    ax.set_ylabel("CA-CRS+ Score")
    ax.set_title(f"CA-CRS+ Score Over Time — {scenario_name}")
    ax.set_ylim(0, 1.05)
    if crs:
        ax.scatter([frames[peak_idx]], [crs[peak_idx]], color="#922B21", s=28, zorder=5)
        ax.annotate(
            f"peak={crs[peak_idx]:.3f}",
            xy=(frames[peak_idx], crs[peak_idx]),
            xytext=(10, -16),
            textcoords="offset points",
            fontsize=8,
            color="#922B21",
        )
    ax.legend(loc="upper left", framealpha=0.85)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, f"{outdir}/fig1_crs_timeline.png")


# ── Figure 2: Multi-zone per-zone CRS ────────────────────────────────────

def fig2_multizone(records, outdir="results/paper_figures"):
    zone_data = defaultdict(lambda: {"frames": [], "crs": []})
    for r in records:
        z = r["zone_name"]
        zone_data[z]["frames"].append(int(r["frame"]))
        zone_data[z]["crs"].append(floatf(r["crs"]))

    colors  = ["#C0392B", "#2980B9", "#27AE60",
               "#8E44AD", "#D35400"]
    fig, ax = plt.subplots(figsize=(9, 3.6))

    preferred_order = ["Entry Corridor", "Main Hall", "Exit Plaza"]
    ordered_names = [name for name in preferred_order if name in zone_data]
    ordered_names.extend(
        sorted(name for name in zone_data if name not in preferred_order)
    )

    for i, zname in enumerate(ordered_names):
        zd = zone_data[zname]
        c = colors[i % len(colors)]
        ax.plot(zd["frames"], zd["crs"],
                label=zname, color=c, lw=1.8)

    max_frame = max(
        max(v["frames"]) for v in zone_data.values()
    )
    ax.axhline(0.70, color="#C0392B", lw=1.1, ls=":", alpha=0.85,
               label="DANGER (0.70)")
    ax.axhline(0.35, color="#E67E22", lw=1.0, ls=":", alpha=0.85,
               label="WARNING (0.35)")
    ax.fill_between(range(max_frame + 1), 0.70, 1.0,
                    alpha=0.05, color="#E74C3C")
    ax.set_xlabel("Frame")
    ax.set_ylabel("CA-CRS+ Score")
    ax.set_title("Per-Zone CA-CRS+ Scores — Multi-Zone Scenario D")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.85)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, f"{outdir}/fig_multizone.png")


# ── Figure 3: Resource adequacy ───────────────────────────────────────────

def fig3_resources(records, available_marshals=20, outdir="results/paper_figures"):
    """
    Estimate marshal demand from risk levels per frame.
    Groups all zones per frame, sums their demand.
    """
    frame_groups = defaultdict(list)
    for r in records:
        frame_groups[int(r["frame"])].append(r)

    frames   = sorted(frame_groups.keys())
    ALPHA    = {"SAFE": 0, "WARNING": 4, "DANGER": 10}
    demands  = []
    statuses = []

    for f in frames:
        total_d = 0
        for r in frame_groups[f]:
            level = r["risk"]
            count = max(int(floatf(r.get("count", 50))), 1)
            log_f = math.log10(1 + count)
            total_d += math.ceil(ALPHA.get(level, 0) * log_f)
        demands.append(total_d)
        shortfall = total_d - available_marshals
        if shortfall <= 0:
            statuses.append("ADEQUATE")
        elif shortfall <= 3:
            statuses.append("STRAINED")
        else:
            statuses.append("CRITICAL")

    res_map    = {"ADEQUATE": 0, "STRAINED": 1, "CRITICAL": 2}
    res_colors = ["#27AE60", "#E67E22", "#C0392B"]
    sv         = [res_map[s] for s in statuses]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(frames, demands, color="#C0392B",
             lw=1.8, label="Marshal demand")
    ax1.axhline(available_marshals, color="#27AE60",
                lw=1.8, ls="--",
                label=f"Available ({available_marshals})")
    ax1.set_ylabel("Marshals Required")
    ax1.legend(framealpha=0.85)
    style_axes(ax1)
    ymax = max(max(demands), available_marshals) * 1.2 + 1
    ax1.set_ylim(0, ymax)

    for i in range(len(frames) - 1):
        ax2.axvspan(frames[i], frames[i + 1],
                    alpha=0.85, color=res_colors[sv[i]])
    ax2.set_yticks([])
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Status")

    patches = [
        mpatches.Patch(color=res_colors[0], label="Adequate"),
        mpatches.Patch(color=res_colors[1], label="Strained"),
        mpatches.Patch(color=res_colors[2], label="Critical"),
    ]
    ax2.legend(handles=patches, ncol=3, framealpha=0.85,
               loc="upper right")

    fig.suptitle("Resource Adequacy vs. Multi-Zone Risk Level",
                 fontsize=11)
    fig.tight_layout()
    save_figure(fig, f"{outdir}/fig_resources.png")


# ── Figure 4: Component contributions ────────────────────────────────────

def fig4_components(records, zone_filter=None, outdir="results/paper_figures"):
    if zone_filter:
        records = [r for r in records
                   if r["zone_name"] == zone_filter]
        frames = [int(r["frame"]) for r in records]
        d_c = [floatf(r["d_norm"]) * 0.40 for r in records]
        s_c = [floatf(r["s_norm"]) * 0.30 for r in records]
        c_c = [floatf(r["c_norm"]) * 0.30 for r in records]
    else:
        groups = frame_groups(records)
        frames = sorted(groups)
        d_c = [np.mean([floatf(r["d_norm"]) * 0.40 for r in groups[frame]]) for frame in frames]
        s_c = [np.mean([floatf(r["s_norm"]) * 0.30 for r in groups[frame]]) for frame in frames]
        c_c = [np.mean([floatf(r["c_norm"]) * 0.30 for r in groups[frame]]) for frame in frames]

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.stackplot(
        frames, d_c, s_c, c_c,
        labels=["Density $(w_1 D_{norm})$",
                "Speed / $\\Phi$ component",
                "Conflict $(w_3 C_{norm})$"],
        colors=["#2980B9", "#E67E22", "#8E44AD"],
        alpha=0.82
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Weighted Contribution")
    zone_label = f" — {zone_filter}" if zone_filter else ""
    ax.set_title(f"CA-CRS+ Component Contributions{zone_label}")
    ax.legend(loc="upper right", framealpha=0.85)
    style_axes(ax)
    ax.set_ylim(0)
    fig.tight_layout()
    save_figure(fig, f"{outdir}/fig_components.png")


# ── Figure 5: Panic speed spike + gate action ────────────────────────────

def fig5_speed_gate(records, zone_filter="Panic Flow", outdir="results/paper_figures"):
    zone_records = [r for r in records if r["zone_name"] == zone_filter]
    if not zone_records:
        print(f"[Fig5] Skipped — no records for zone: {zone_filter}")
        return

    frames = [int(r["frame"]) for r in zone_records]
    speed = [floatf(r["s_norm"]) for r in zone_records]
    crs = [floatf(r["crs"]) for r in zone_records]
    close_frames = [
        int(r["frame"]) for r in zone_records
        if "CLOSE" in r.get("gate_summary", "")
    ]

    fig, ax1 = plt.subplots(figsize=(9, 3.6))
    ax1.plot(frames, speed, color="#E67E22", lw=1.9, label="$S_{norm}$")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Normalized Speed")
    ax1.set_ylim(0, 1.05)
    style_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(frames, crs, color="#C0392B", lw=1.7, label="CA-CRS+")
    ax2.axhline(0.35, color="#E67E22", lw=1.0, ls=":", alpha=0.85)
    ax2.axhline(0.70, color="#C0392B", lw=1.0, ls=":", alpha=0.85)
    ax2.set_ylabel("CA-CRS+ Score")
    ax2.set_ylim(0, 1.05)

    for frame in sorted(set(close_frames)):
        ax1.axvline(frame, color="#2C3E50", lw=0.9, alpha=0.20)

    if close_frames:
        ax1.scatter(
            close_frames,
            [1.02] * len(close_frames),
            marker="v",
            s=28,
            color="#2C3E50",
            label="CLOSE command",
            clip_on=False,
            zorder=5,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", framealpha=0.85)
    ax1.set_title(f"Panic Response: Speed Spike and Gate Action — {zone_filter}")
    fig.tight_layout()
    save_figure(fig, f"{outdir}/fig_speed_gate.png")


# ── Table values helper ───────────────────────────────────────────────────

def print_table_values(records):
    """
    Prints values needed to fill in paper tables.
    Copy these numbers into your .tex file.
    """
    print("\n" + "="*60)
    print("TABLE VALUES FOR YOUR PAPER")
    print("="*60)

    danger = [r for r in records if r["risk"] == "DANGER"]
    total_records = len(records)
    unique_frames = sorted({int(r["frame"]) for r in records})
    danger_frames = sorted({int(r["frame"]) for r in danger})

    print(f"\nTotal records    : {total_records}")
    print(f"Unique frames    : {len(unique_frames)}")
    print(f"DANGER records   : {len(danger)}"
          f" ({100*len(danger)/max(total_records,1):.1f}%)")
    print(f"DANGER frames    : {len(danger_frames)}"
          f" ({100*len(danger_frames)/max(len(unique_frames),1):.1f}%)")
    print(f"Mean CA-CRS+     : "
          f"{np.mean([floatf(r['crs']) for r in records]):.3f}")
    print(f"Mean Proj. CRS   : "
          f"{np.mean([floatf(r['proj_crs']) for r in records]):.3f}")
    print(f"Mean CRR (%)     : "
          f"{np.mean([floatf(r['crr']) for r in records]):.1f}%")
    print(f"Mean positive CRR: "
          f"{np.mean([floatf(r['crr']) for r in records if floatf(r['crr']) > 0]):.1f}%")

    # Per-zone breakdown
    print("\nPer-zone DANGER count:")
    zone_groups = defaultdict(list)
    for r in records:
        zone_groups[r["zone_name"]].append(r)

    for zname, zrecs in sorted(zone_groups.items()):
        zdanger = [r for r in zrecs if r["risk"] == "DANGER"]
        zmean   = np.mean([floatf(r["crs"]) for r in zrecs])
        print(f"  {zname:<20}: {len(zdanger):4d} DANGER frames, "
              f"mean CRS={zmean:.3f}")

    # Gate action breakdown
    print("\nGate action distribution (DANGER frames):")
    actions = [r["gate_summary"].split(":")[1].split("[")[0].strip()
               for r in danger if r.get("gate_summary")]
    for act, cnt in Counter(actions).most_common():
        print(f"  {act:<12}: {cnt}")

    print("\n" + "="*60)
    print("Copy numbers above into Table III of your paper.")
    print("="*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paper figures from CA-CRS+ results"
    )
    parser.add_argument("--csv",        required=True,
                        help="Path to results CSV")
    parser.add_argument("--marshals",   type=int, default=20)
    parser.add_argument("--multizone",  action="store_true",
                        help="Generate multi-zone figure")
    parser.add_argument("--zone",       default=None,
                        help="Filter zone for component figure")
    parser.add_argument("--outdir",     default="results/paper_figures",
                        help="Directory for saved figures")
    parser.add_argument("--scenario",   default="Real Crowd Video",
                        help="Scenario label for Figure 1 title")
    parser.add_argument("--panic_zone", default=None,
                        help="Generate speed/gate figure for the selected zone")
    parser.add_argument("--resources",  action="store_true",
                        help="Generate marshal demand / resource adequacy figure")
    args = parser.parse_args()

    records = load_csv(args.csv)
    print(f"[INFO] Loaded {len(records)} records from {args.csv}")
    os.makedirs(args.outdir, exist_ok=True)

    fig1_crs_timeline(records, args.scenario, args.outdir)
    fig4_components(records, args.zone, args.outdir)
    if args.resources:
        fig3_resources(records, args.marshals, args.outdir)

    if args.multizone:
        fig2_multizone(records, args.outdir)
    if args.panic_zone:
        fig5_speed_gate(records, args.panic_zone, args.outdir)

    print_table_values(records)
    print(f"\n[DONE] All figures saved to: {args.outdir}/")
