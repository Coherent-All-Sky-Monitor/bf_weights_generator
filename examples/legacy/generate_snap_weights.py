#!/usr/bin/env python3
"""
Generate int8-quantized beamformer weights for SNAP hardware.

This script reads antenna layouts from CSV files and generates weights
in the format expected by the SNAP beamformer.

Examples:
    # Generate 8 beams with auto-computed spacing
    python generate_snap_weights.py -l layout.csv -o weights.h5 --n-beams 8

    # Generate beams with 20 degree spacing
    python generate_snap_weights.py -l layout.csv -o weights.h5 --spacing 20

    # Custom altitude range
    python generate_snap_weights.py -l layout.csv -o weights.h5 --n-beams 16 --alt-min 45

    # Specific beam positions (alt:az format)
    python generate_snap_weights.py -l layout.csv -o weights.h5 --beams "90:0,70:0,70:90,70:180"

    # Use preset beam patterns
    python generate_snap_weights.py -l layout.csv -o weights.h5 --beams transit
    python generate_snap_weights.py -l layout.csv -o weights.h5 --beams zenith
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from bf_weights_generator import (
    Array64Config,
    SnapWeightsGenerator,
    FrequencyConfig,
    save_int8_weights_hdf5,
    inspect_int8_weights_file,
    TRANSIT_SURVEY_BEAMS,
    parse_beams_arg,
    generate_beam_grid,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate int8-quantized SNAP beamformer weights from CSV antenna layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--layout", "-l",
        required=True,
        help="Path to antenna layout CSV file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output HDF5 file path",
    )

    # Beam specification - mutually exclusive group
    beam_group = parser.add_argument_group("Beam Configuration (choose one method)")
    beam_exclusive = beam_group.add_mutually_exclusive_group()
    beam_exclusive.add_argument(
        "--n-beams", "-n",
        type=int,
        help="Number of beams to generate (auto-computes spacing)",
    )
    beam_exclusive.add_argument(
        "--beams", "-b",
        help="Beam specification: 'transit', 'zenith', or 'alt:az,alt:az,...'",
    )

    # Grid generation options (used with --n-beams or --spacing)
    grid_group = parser.add_argument_group("Grid Options (for --n-beams or --spacing)")
    grid_group.add_argument(
        "--spacing",
        type=float,
        default=15.0,
        help="Beam spacing in degrees (default: 15.0, ignored if --n-beams set)",
    )
    grid_group.add_argument(
        "--alt-min",
        type=float,
        default=30.0,
        help="Minimum altitude in degrees (default: 30.0)",
    )
    grid_group.add_argument(
        "--alt-max",
        type=float,
        default=90.0,
        help="Maximum altitude in degrees (default: 90.0)",
    )

    # Other options
    parser.add_argument(
        "--n-chan",
        type=int,
        default=3072,
        help="Number of frequency channels (default: 3072)",
    )
    parser.add_argument(
        "--overwrite", "-f",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information",
    )

    args = parser.parse_args()

    # Load array configuration from CSV
    print(f"Loading antenna layout from: {args.layout}")
    array_config = Array64Config.from_csv(args.layout)
    print(f"  Active antennas: {array_config.n_active}/64")
    print(f"  Active indices: {list(array_config.active_indices)}")

    if args.verbose:
        print(f"  SNAP mapping:")
        for i, ant64_idx in enumerate(array_config.snap_to_ant64):
            if ant64_idx >= 0:
                print(f"    SNAP input {i:2d} -> ant64 slot {ant64_idx:2d} ({array_config.pos_ids[ant64_idx]})")

    # Determine beam pointings
    if args.beams:
        # Use preset or custom beams
        try:
            pointings = parse_beams_arg(args.beams)
            beam_source = f"preset '{args.beams}'" if args.beams in ['transit', 'zenith'] else "custom"
        except ValueError as e:
            print(f"Error parsing beams: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.n_beams:
        # Generate grid with specified number of beams
        pointings = generate_beam_grid(
            n_beams=args.n_beams,
            spacing_deg=args.spacing,
            alt_min_deg=args.alt_min,
            alt_max_deg=args.alt_max,
        )
        beam_source = f"grid (target={args.n_beams}, alt=[{args.alt_min:.0f},{args.alt_max:.0f}])"
    else:
        # Generate grid with specified spacing
        pointings = generate_beam_grid(
            spacing_deg=args.spacing,
            alt_min_deg=args.alt_min,
            alt_max_deg=args.alt_max,
        )
        beam_source = f"grid (spacing={args.spacing}deg, alt=[{args.alt_min:.0f},{args.alt_max:.0f}])"

    print(f"\nBeam configuration: {len(pointings)} beams ({beam_source})")
    for i, p in enumerate(pointings):
        print(f"  Beam {i:2d}: alt={p.alt_deg:5.1f} deg, az={p.az_deg:5.1f} deg  ({p.name})")

    # Create frequency configuration
    freq_config = FrequencyConfig(n_chan=args.n_chan)
    print(f"\nFrequency configuration: {freq_config}")

    # Generate weights
    print("\nComputing weights...")
    generator = SnapWeightsGenerator(array_config, freq_config)
    weights = generator.compute_int8_weights(pointings)
    print(f"  Output shape: {weights.shape}")
    print(f"  Memory size: {weights.weights_int8.nbytes / 1e6:.2f} MB")

    # Save to HDF5
    output_path = Path(args.output)
    print(f"\nSaving to: {output_path}")
    save_int8_weights_hdf5(weights, output_path, overwrite=args.overwrite)
    print(f"  File size: {output_path.stat().st_size / 1e6:.2f} MB")

    # Print summary
    if args.verbose:
        print("\nFile inspection:")
        info = inspect_int8_weights_file(output_path)
        for key, value in info.items():
            print(f"  {key}: {value}")

    print("\nDone!")


if __name__ == "__main__":
    main()
