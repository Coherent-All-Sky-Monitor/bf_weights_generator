#!/usr/bin/env python
"""
Generate combined geometric + delay-calibration beamformer weights.

Computes complex64 weights that combine geometric steering phases with
SVD-derived delay calibration corrections. Saves to HDF5 with full metadata.

Usage:
    python examples/generate_combined_weights.py \
        --compute-layout casm_antenna_layout_pre_feb16.csv \
        --output-layout  casm_antenna_layout_current.csv \
        --cal-weights    delay_cal_weights/svd_weights_sun_2026-02-14_phase-only_thr2.0_norfi.npz \
        --n-beams 8 \
        --output combined_weights_8beam.h5
"""

import argparse
import numpy as np
from pathlib import Path

from bf_weights_generator import (
    Array64Config,
    FrequencyConfig,
    load_calibration_weights,
    generate_beam_grid,
    generate_combined_weights,
    save_combined_weights_hdf5,
)
from bf_weights_generator.snap_weights import parse_beams_arg


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined geometric + cal beamformer weights (complex64)."
    )
    parser.add_argument(
        "--compute-layout", required=True,
        help="CSV layout for computing geometric weights (antenna positions).",
    )
    parser.add_argument(
        "--output-layout", default=None,
        help="CSV layout for SNAP output ordering. Defaults to compute-layout.",
    )
    parser.add_argument(
        "--cal-weights", default=None,
        help="Path to SVD calibration weights (.npz).",
    )
    parser.add_argument(
        "--n-beams", type=int, default=8,
        help="Number of beams (default: 8).",
    )
    parser.add_argument(
        "--beams", default=None,
        help="Beam specification: 'transit', 'zenith', or 'alt:az,alt:az,...'.",
    )
    parser.add_argument(
        "--alt-min", type=float, default=30.0,
        help="Minimum altitude for beam grid (default: 30 deg).",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--freq-order", default="descending", choices=["ascending", "descending"],
        help="Frequency axis ordering (default: descending, 469->375 MHz).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output file.",
    )
    args = parser.parse_args()

    # Load layouts
    compute_layout = Array64Config.from_csv(args.compute_layout)
    output_layout = (
        Array64Config.from_csv(args.output_layout)
        if args.output_layout
        else compute_layout
    )
    print(f"Compute layout: {compute_layout}")
    print(f"Output layout:  {output_layout}")

    # Load cal weights
    cal = None
    if args.cal_weights:
        cal = load_calibration_weights(args.cal_weights)
        print(
            f"Cal weights: {cal.weights.shape}, {np.sum(cal.flags)} good channels, "
            f"source={cal.source}, ref_ant={cal.ref_ant_id}"
        )

    # Generate beam pointings
    if args.beams:
        pointings = parse_beams_arg(args.beams)
    else:
        pointings = generate_beam_grid(
            n_beams=args.n_beams,
            array_config=compute_layout,
            alt_min_deg=args.alt_min,
        )
    print(f"Beams: {len(pointings)}")
    for i, p in enumerate(pointings):
        print(f"  {i}: Alt={p.alt_deg:6.1f} Az={p.az_deg:6.1f}  {p.name}")

    # Compute combined weights
    result = generate_combined_weights(
        pointing=pointings,
        array_config=compute_layout,
        cal_weights=cal,
        output_array_config=output_layout,
        freq_order=args.freq_order,
    )
    print(f"\nWeights shape: {result.weights.shape}, dtype: {result.weights.dtype}")
    print(f"Frequencies: {result.frequencies_hz[0]/1e6:.3f} - "
          f"{result.frequencies_hz[-1]/1e6:.3f} MHz ({result.freq_order})")
    print(f"Good channels: {result.n_good_channels} / {result.n_channels}")

    # Check weight properties
    active_snaps = [
        si for si in range(64) if output_layout.snap_to_ant64[si] >= 0
    ]
    mags = np.abs(result.weights[:, active_snaps, :])
    good_mags = mags[:, :, result.flags]
    print(f"Active SNAP inputs: {len(active_snaps)}")
    if good_mags.size > 0:
        print(f"Weight magnitudes (good ch, active ant): "
              f"mean={np.mean(good_mags):.4f}, std={np.std(good_mags):.4f}")

    # Save
    save_combined_weights_hdf5(result, args.output, overwrite=args.overwrite)
    fsize = Path(args.output).stat().st_size / 1e6
    print(f"\nSaved to {args.output} ({fsize:.1f} MB)")


if __name__ == "__main__":
    main()
