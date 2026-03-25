#!/usr/bin/env python3
"""
Convert simple antenna layout CSV to the format expected by Array64Config.from_csv().

Input format columns:
    antenna, x, y, z, snap, adc, packet_idx, functional, row, col

Output format columns:
    pos_id, plank_id, ew_slot, ew_label, x_east_m, y_north_m, z_up_m,
    installed, pos_type, snap_A, adc_A, enabled_A, quality_A,
    snap_B, adc_B, enabled_B, quality_B,
    ant64, include_in_ant64, include_in_beamforming, is_reference_candidate, comment

Usage:
    python convert_antenna_layout.py antenna_layout_current.csv -o casm_antenna_layout_current.csv
    python convert_antenna_layout.py antenna_layout_pre_feb16.csv antenna_layout_current.csv
"""

import argparse
import csv
import sys
from pathlib import Path


OUTPUT_COLUMNS = [
    "pos_id", "plank_id", "ew_slot", "ew_label",
    "x_east_m", "y_north_m", "z_up_m",
    "installed", "pos_type",
    "snap_A", "adc_A", "enabled_A", "quality_A",
    "snap_B", "adc_B", "enabled_B", "quality_B",
    "ant64", "include_in_ant64", "include_in_beamforming",
    "is_reference_candidate", "comment",
]


def convert_layout(input_path, output_path):
    """Convert simple layout CSV to Array64Config-compatible format."""
    rows_out = []

    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            antenna_idx = int(row["antenna"])
            x = row["x"].strip()
            y = row["y"].strip()
            z = row["z"].strip()
            snap = row["snap"].strip()
            adc = row["adc"].strip()
            functional = row["functional"].strip() == "1"
            plank_id = row["row"].strip()
            ew_label = row["col"].strip()

            # Build pos_id from row and col (e.g., "N21_E01")
            ew_slot = ew_label.replace("E", "").replace("O", "")
            pos_id = f"{plank_id}_{ew_label}"

            # ant64 slot is just the 0-based antenna index
            ant64 = antenna_idx - 1

            has_snap = snap != "" and adc != ""

            out_row = {
                "pos_id": pos_id,
                "plank_id": plank_id,
                "ew_slot": ew_slot if ew_slot.isdigit() else "",
                "ew_label": ew_label,
                "x_east_m": x,
                "y_north_m": y,
                "z_up_m": z,
                "installed": functional,
                "pos_type": "antenna",
                "snap_A": f"{snap}.0" if has_snap else "",
                "adc_A": f"{adc}.0" if has_snap else "",
                "enabled_A": functional and has_snap,
                "quality_A": "unknown",
                "snap_B": "",
                "adc_B": "",
                "enabled_B": False,
                "quality_B": "unknown",
                "ant64": f"{ant64}.0",
                "include_in_ant64": True,
                "include_in_beamforming": functional and has_snap,
                "is_reference_candidate": False,
                "comment": "",
            }
            rows_out.append(out_row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Converted {len(rows_out)} antennas: {input_path} -> {output_path}")


def default_output_name(input_path):
    """Generate output filename: antenna_layout_X.csv -> casm_antenna_layout_X.csv"""
    stem = Path(input_path).stem
    return f"casm_{stem}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Convert simple antenna layout CSV to Array64Config format."
    )
    parser.add_argument(
        "input_files", nargs="+",
        help="Input CSV file(s) in simple format",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV path (only valid with a single input file)",
    )
    args = parser.parse_args()

    if args.output and len(args.input_files) > 1:
        print("Error: -o/--output can only be used with a single input file.", file=sys.stderr)
        sys.exit(1)

    for input_path in args.input_files:
        output_path = args.output or default_output_name(input_path)
        convert_layout(input_path, output_path)


if __name__ == "__main__":
    main()
