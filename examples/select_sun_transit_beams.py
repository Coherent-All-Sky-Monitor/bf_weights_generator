#!/usr/bin/env python3
"""
Select 8 coherent beams for sun transit testing on 2026-03-12.

Reads the weights HDF5 file, verifies beam indices match expected alt/az
coordinates, and writes a CSV with the selected beams plus an incoherent
beam row.

Usage:
    python examples/select_sun_transit_beams.py [--weights PATH] [--output PATH]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Add parent to path so we can import bf_weights_generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bf_weights_generator.io import load_int8_weights_hdf5

DEFAULT_WEIGHTS = (
    "/home/casm/software/vishnu/beamforming_weights/20261003/"
    "int8_weights_512beam_mar10_svd.h5"
)

# Selected beams: (beam_index, expected_alt, expected_az, group, justification)
SELECTED_BEAMS = [
    (229, 48.70, 180.00, "Sun transit",
     "Primary test beam on south meridian at sun transit altitude (~49.7 deg). "
     "Must show peak sun signal at ~12:00 PST."),
    (81, 63.65, 180.00, "South meridian (above)",
     "~15 deg above sun on south meridian. Tests N-S beam steering accuracy."),
    (321, 41.22, 180.00, "South meridian (below)",
     "~7.5 deg below sun. Confirms beam peak at correct altitude."),
    (483, 30.00, 180.00, "South meridian (low)",
     "Lowest beam on south meridian (~20 deg from sun). Should be quiet; "
     "lights up if altitude steering is broken."),
    (251, 44.96, 0.00, "North (sign flip)",
     "Critical sign diagnostic. Similar altitude to sun but on NORTH meridian. "
     "If this beam sees the sun, the N-S sign is flipped."),
    (0, 89.83, 0.00, "Zenith",
     "Near-zenith quiet sky reference. ~40 deg from sun. "
     "Potential Cyg-A (alt~86 deg) night detection."),
    (231, 48.70, 196.36, "Off-meridian sun",
     "Sun closest approach at ~12:45 PST. Tests E-W beam steering; "
     "sun hits this beam as it moves westward after transit."),
    (53, 67.39, 180.00, "Night calibrator",
     "South meridian at alt~67 deg. Vir-A transits at alt~65 deg tonight. "
     "Extends south meridian column for daytime sign diagnosis."),
]

ALT_TOL = 0.1  # degrees tolerance for coordinate verification
AZ_TOL = 0.1


def main():
    parser = argparse.ArgumentParser(
        description="Select beams for sun transit testing"
    )
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS,
        help="Path to int8 weights HDF5 file",
    )
    parser.add_argument(
        "--output", default="sun_transit_beams_20260312.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Load weights file
    print(f"Loading weights: {args.weights}")
    w = load_int8_weights_hdf5(args.weights)
    print(f"  {w.n_beams} beams, {w.n_channels} channels")

    # Verify each selected beam's coordinates
    print("\nVerifying beam coordinates:")
    all_ok = True
    for beam_idx, exp_alt, exp_az, group, _ in SELECTED_BEAMS:
        if beam_idx >= w.n_beams:
            print(f"  FAIL: Beam {beam_idx} out of range (max {w.n_beams - 1})")
            all_ok = False
            continue

        p = w.pointings[beam_idx]
        alt_ok = abs(p.alt_deg - exp_alt) < ALT_TOL
        az_ok = abs(p.az_deg - exp_az) < AZ_TOL
        status = "OK" if (alt_ok and az_ok) else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False

        print(
            f"  Beam {beam_idx:3d}: alt={p.alt_deg:6.2f} (exp {exp_alt:6.2f}) "
            f"az={p.az_deg:6.2f} (exp {exp_az:6.2f})  [{status}]  {group}"
        )

    if not all_ok:
        print("\nWARNING: Some beams did not match expected coordinates!")
    else:
        print("\nAll beam coordinates verified.")

    # Write CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "beam_index", "alt_deg", "az_deg", "group", "justification",
        ])
        # Incoherent beam row
        writer.writerow([
            "", "", "", "Incoherent",
            "Weight-independent power sum (|V|^2). Baseline reference with "
            "wide primary beam pattern.",
        ])
        # Coherent beams (use actual coordinates from the file)
        for beam_idx, _, _, group, justification in SELECTED_BEAMS:
            if beam_idx < w.n_beams:
                p = w.pointings[beam_idx]
                writer.writerow([
                    beam_idx, f"{p.alt_deg:.2f}", f"{p.az_deg:.2f}",
                    group, justification,
                ])

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
