#!/usr/bin/env python3
"""
Validate that the paper-facing metrics manifest matches the generated summary
files stored in this repository.

Usage:
    python3 tools/check_paper_artifacts.py
"""

from __future__ import annotations

import csv
import re
import sys
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

    if failures:
        print(f"\n{failures} manifest check(s) failed.")
        return 1

    print("\nAll paper-facing metrics match the generated summary files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
