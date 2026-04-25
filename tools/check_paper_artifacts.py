#!/usr/bin/env python3
"""
Validate that the paper-facing metrics manifest matches the generated summary
files stored in this repository.

Usage:
    python3 tools/check_paper_artifacts.py
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper_artifacts" / "final_metrics.csv"

LINE_PATTERNS = {
    "unique_frames": re.compile(r"^Unique frames\s*:\s*(\d+)"),
    "danger_frames_pct": re.compile(r"^DANGER frames\s*:\s*\d+\s+\(([\d.]+)%\)"),
    "mean_ca_crs": re.compile(r"^Mean CA-CRS\+\s*:\s*([\d.]+)"),
    "mean_projected_crs": re.compile(r"^Mean Projected CRS\s*:\s*([\d.]+)"),
    "mean_crr_pct": re.compile(r"^Mean CRR \(%\)\s*:\s*([\d.]+)%"),
}

FLOAT_FIELDS = {
    "danger_frames_pct",
    "mean_ca_crs",
    "mean_projected_crs",
    "mean_crr_pct",
}

MULTICAM_EXPECTED = {
    "weighted_mean_grs": 0.5438,
    "simple_mean_zone_score": 0.5407,
    "min_grs": 0.4262,
    "max_grs": 1.0,
    "entry_corridor_worst_pct": 32.2,
    "entry_corridor_worst_frames": 19,
    "main_hall_worst_frames": 19,
    "exit_plaza_worst_frames": 21,
    "resource_demand_min": 11,
    "resource_demand_max": 93,
}

SPEED_EXPECTED = {
    "total_records": 86,
    "unique_frames": 86,
    "warning_records": 25,
    "speed_dominant_pct": 98.8,
    "close_records": 25,
    "max_s_norm": 1.0,
}


def parse_summary(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        for key, pattern in LINE_PATTERNS.items():
            match = pattern.match(line)
            if not match:
                continue
            if key in FLOAT_FIELDS:
                values[key] = float(match.group(1))
            else:
                values[key] = int(match.group(1))
    missing = [key for key in LINE_PATTERNS if key not in values]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    return values


def almost_equal(expected: float, actual: float, tolerance: float = 0.11) -> bool:
    return abs(expected - actual) <= tolerance


def parse_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def derive_multicam_metrics(path: Path) -> dict[str, float]:
    rows = parse_csv(path)
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame"])].append(row)

    weighted_grs = []
    simple_means = []
    worst_zone_counts: Counter[str] = Counter()
    marshal_demands = []
    alpha = {"SAFE": 0, "WARNING": 4, "DANGER": 10}

    for frame in sorted(grouped):
        frame_rows = grouped[frame]
        weighted_sum = 0.0
        total_weight = 0.0
        simple_sum = 0.0
        worst_zone = None
        worst_crs = -1.0
        demand = 0

        for row in frame_rows:
            crs = float(row["crs"])
            risk = row["risk"]
            weight = 2.0 if risk == "DANGER" else 1.0
            weighted_sum += crs * weight
            total_weight += weight
            simple_sum += crs
            if crs > worst_crs:
                worst_crs = crs
                worst_zone = row["zone_name"]
            demand += math.ceil(alpha[risk] * math.log10(1 + max(int(float(row["count"])), 0)))

        weighted_grs.append(weighted_sum / total_weight if total_weight else 0.0)
        simple_means.append(simple_sum / len(frame_rows))
        if worst_zone is not None:
            worst_zone_counts[worst_zone] += 1
        marshal_demands.append(demand)

    total_frames = len(grouped)
    return {
        "weighted_mean_grs": round(sum(weighted_grs) / total_frames, 4),
        "simple_mean_zone_score": round(sum(simple_means) / total_frames, 4),
        "min_grs": round(min(weighted_grs), 4),
        "max_grs": round(max(weighted_grs), 4),
        "entry_corridor_worst_pct": round(100.0 * worst_zone_counts["Entry Corridor"] / total_frames, 1),
        "entry_corridor_worst_frames": worst_zone_counts["Entry Corridor"],
        "main_hall_worst_frames": worst_zone_counts["Main Hall"],
        "exit_plaza_worst_frames": worst_zone_counts["Exit Plaza"],
        "resource_demand_min": min(marshal_demands),
        "resource_demand_max": max(marshal_demands),
    }


def derive_speed_metrics(path: Path) -> dict[str, float]:
    rows = parse_csv(path)
    total = len(rows)
    unique_frames = len({row["frame"] for row in rows})
    warning_records = sum(row["risk"] == "WARNING" for row in rows)
    speed_dominant = sum(row["factor"] == "SPEED" for row in rows)
    close_records = sum("CLOSE" in row["gate_summary"] for row in rows)
    max_s_norm = max(float(row["s_norm"]) for row in rows)
    return {
        "total_records": total,
        "unique_frames": unique_frames,
        "warning_records": warning_records,
        "speed_dominant_pct": round(100.0 * speed_dominant / total, 1),
        "close_records": close_records,
        "max_s_norm": round(max_s_norm, 4),
    }


def check_expected_block(name: str, actual: dict[str, float], expected: dict[str, float]) -> int:
    failures = 0
    for key, exp_value in expected.items():
        act_value = actual[key]
        is_float = isinstance(exp_value, float)
        ok = almost_equal(exp_value, act_value, tolerance=0.11 if is_float else 0.0) if is_float else exp_value == act_value
        if ok:
            print(f"[OK]   {name}.{key} = {act_value}")
        else:
            print(f"[FAIL] {name}.{key}: expected {exp_value}, got {act_value}")
            failures += 1
    return failures


def main() -> int:
    if not MANIFEST.exists():
        print(f"[FAIL] Missing manifest: {MANIFEST}")
        return 1

    failures = 0
    with MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary_path = ROOT / row["summary_file"]
            if not summary_path.exists():
                print(f"[FAIL] {row['scenario']}: missing summary file {summary_path}")
                failures += 1
                continue

            actual = parse_summary(summary_path)
            mismatches = []
            for field in LINE_PATTERNS:
                expected = float(row[field]) if field in FLOAT_FIELDS else int(row[field])
                actual_value = actual[field]
                if field in FLOAT_FIELDS:
                    if not almost_equal(expected, actual_value):
                        mismatches.append(f"{field}: expected {expected}, got {actual_value}")
                elif expected != actual_value:
                    mismatches.append(f"{field}: expected {expected}, got {actual_value}")

            if mismatches:
                print(f"[FAIL] {row['scenario']}")
                for mismatch in mismatches:
                    print(f"  - {mismatch}")
                failures += 1
            else:
                print(f"[OK]   {row['scenario']} -> {summary_path.relative_to(ROOT)}")

    print()
    multicam_metrics = derive_multicam_metrics(ROOT / "results" / "multicam" / "results.csv")
    failures += check_expected_block("multicam", multicam_metrics, MULTICAM_EXPECTED)

    print()
    speed_metrics = derive_speed_metrics(ROOT / "results" / "scen_d_probe3" / "results.csv")
    failures += check_expected_block("speed_validation", speed_metrics, SPEED_EXPECTED)

    if failures:
        print(f"\n{failures} manifest check(s) failed.")
        return 1

    print("\nAll paper-facing metrics and derived claims match the stored artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
