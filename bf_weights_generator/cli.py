"""
Command-line interface for beamformer weight generation.
"""

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .weights import GeometricBeamformer, PhaseCenter, BeamMode
from .config import ArrayConfig, FrequencyConfig
from .io import save_weights_hdf5, save_weights_npz, inspect_weights_file, HDF5_AVAILABLE


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="bf-weights",
        description="Generate geometric beamformer weights for CASM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate coherent weights for Cygnus A
  bf-weights --ra 299.868 --dec 40.734 --name CygA \\
             --start-time 1700000000 --duration 3600 --cadence 10 \\
             -o cyga_weights.h5

  # Generate 8 incoherent beams
  bf-weights --mode incoherent --n-beams 8 \\
             --start-time 1700000000 --duration 3600 --cadence 1 \\
             -o incoherent_weights.h5

  # Generate weights for multiple sources (from JSON file)
  bf-weights --sources sources.json \\
             --start-time 1700000000 --duration 3600 \\
             -o multi_beam_weights.h5

  # Inspect an existing weights file
  bf-weights --inspect weights.h5
        """
    )

    # Source specification
    source_group = parser.add_argument_group("Source specification")
    source_group.add_argument(
        "--ra", type=float,
        help="Right Ascension in degrees (J2000)"
    )
    source_group.add_argument(
        "--dec", type=float,
        help="Declination in degrees (J2000)"
    )
    source_group.add_argument(
        "--name", type=str, default="",
        help="Source name (optional)"
    )
    source_group.add_argument(
        "--sources", type=str,
        help="JSON file with source list [{\"ra\": deg, \"dec\": deg, \"name\": str}, ...]"
    )

    # Beamforming mode
    mode_group = parser.add_argument_group("Beamforming mode")
    mode_group.add_argument(
        "--mode", type=str, choices=["coherent", "incoherent"],
        default="coherent",
        help="Beamforming mode (default: coherent)"
    )
    mode_group.add_argument(
        "--n-beams", type=int, default=1,
        help="Number of beams for incoherent mode (default: 1)"
    )

    # Time specification
    time_group = parser.add_argument_group("Time specification")
    time_group.add_argument(
        "--start-time", type=float,
        help="Start time as Unix timestamp"
    )
    time_group.add_argument(
        "--duration", type=float,
        help="Duration in seconds"
    )
    time_group.add_argument(
        "--cadence", type=float, default=1.0,
        help="Time cadence in seconds (default: 1.0)"
    )
    time_group.add_argument(
        "--times-file", type=str,
        help="File with Unix timestamps (one per line or numpy .npy)"
    )

    # Frequency configuration
    freq_group = parser.add_argument_group("Frequency configuration")
    freq_group.add_argument(
        "--n-chan", type=int, default=3072,
        help="Number of frequency channels (default: 3072)"
    )
    freq_group.add_argument(
        "--total-bw-mhz", type=float, default=125.0,
        help="Total system bandwidth in MHz (default: 125.0)"
    )
    freq_group.add_argument(
        "--total-n-chan", type=int, default=4096,
        help="Total system channels (default: 4096)"
    )
    freq_group.add_argument(
        "--freq-end-mhz", type=float, default=468.75,
        help="Upper frequency edge in MHz (default: 468.75)"
    )

    # Array configuration
    array_group = parser.add_argument_group("Array configuration")
    array_group.add_argument(
        "--flag-antennas", type=str,
        help="Comma-separated list of antenna indices to flag (e.g., '3,5,12')"
    )
    array_group.add_argument(
        "--positions-file", type=str,
        help="CSV file with antenna positions (columns: antenna,x,y,z)"
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", type=str,
        help="Output file path (use .h5 or .npz extension)"
    )
    output_group.add_argument(
        "--format", type=str, choices=["hdf5", "npz", "auto"],
        default="auto",
        help="Output format (default: auto, determined by extension)"
    )
    output_group.add_argument(
        "--compression", type=int, default=4,
        help="Compression level for HDF5 (0-9, default: 4)"
    )
    output_group.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output file"
    )

    # Utility
    util_group = parser.add_argument_group("Utility")
    util_group.add_argument(
        "--inspect", type=str, metavar="FILE",
        help="Inspect an existing weights file and exit"
    )
    util_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    return parser.parse_args(args)


def load_positions_from_csv(filepath: str) -> np.ndarray:
    """Load antenna positions from CSV file."""
    positions = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Skip header
            if 'antenna' in line.lower() or 'x' in line.lower():
                continue
            parts = line.split(',')
            if len(parts) >= 4:
                # Format: antenna,x,y,z
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                positions.append([x, y, z])
            elif len(parts) == 3:
                # Format: x,y,z
                positions.append([float(p) for p in parts])
    return np.array(positions, dtype=np.float64)


def load_times_from_file(filepath: str) -> np.ndarray:
    """Load timestamps from file."""
    path = Path(filepath)
    if path.suffix == '.npy':
        return np.load(filepath)
    else:
        # Text file, one timestamp per line
        times = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    times.append(float(line))
        return np.array(times, dtype=np.float64)


def main(args=None):
    """Main entry point for CLI."""
    args = parse_args(args)

    # Handle inspect mode
    if args.inspect:
        info = inspect_weights_file(args.inspect)
        print(f"File: {args.inspect}")
        print(f"Format: {info['format']}")
        print(f"Mode: {info['mode']}")
        print(f"Shape: {info['weights_shape']}")
        print(f"  n_beams: {info['n_beams']}")
        print(f"  n_times: {info['n_times']}")
        print(f"  n_antennas: {info['n_antennas']}")
        print(f"  n_channels: {info['n_channels']}")
        print(f"Dtype: {info['weights_dtype']}")
        print(f"File size: {info['file_size_mb']:.2f} MB")
        print(f"Created: {info['created_utc']}")
        print(f"Version: {info['version']}")
        return 0

    # Validate required arguments
    if not args.output:
        print("Error: --output is required", file=sys.stderr)
        return 1

    # Determine mode
    mode = BeamMode(args.mode)

    # Build phase centers
    phase_centers = []
    if mode == BeamMode.COHERENT:
        if args.sources:
            # Load from JSON file
            with open(args.sources, 'r') as f:
                sources = json.load(f)
            for src in sources:
                phase_centers.append(PhaseCenter(
                    ra_deg=src['ra'],
                    dec_deg=src['dec'],
                    name=src.get('name', '')
                ))
        elif args.ra is not None and args.dec is not None:
            phase_centers.append(PhaseCenter(
                ra_deg=args.ra,
                dec_deg=args.dec,
                name=args.name
            ))
        else:
            print("Error: For coherent mode, provide --ra/--dec or --sources", file=sys.stderr)
            return 1

    # Build time array
    if args.times_file:
        unix_times = load_times_from_file(args.times_file)
    elif args.start_time is not None and args.duration is not None:
        n_times = int(np.ceil(args.duration / args.cadence))
        unix_times = args.start_time + np.arange(n_times) * args.cadence
    else:
        print("Error: Provide --start-time/--duration or --times-file", file=sys.stderr)
        return 1

    # Build frequency config
    freq_config = FrequencyConfig(
        n_chan=args.n_chan,
        total_bw_mhz=args.total_bw_mhz,
        total_n_chan=args.total_n_chan,
        freq_end_voltage_mhz=args.freq_end_mhz
    )

    # Build array config
    if args.positions_file:
        positions = load_positions_from_csv(args.positions_file)
        array_config = ArrayConfig(positions_enu=positions)
    else:
        array_config = ArrayConfig()

    # Flag antennas if requested
    if args.flag_antennas:
        flag_indices = [int(x.strip()) for x in args.flag_antennas.split(',')]
        array_config.flag_antennas(flag_indices)

    if args.verbose:
        print(f"Array config: {array_config}")
        print(f"Freq config: {freq_config}")
        print(f"Mode: {mode.value}")
        print(f"Time range: {unix_times[0]} - {unix_times[-1]} ({len(unix_times)} samples)")
        if mode == BeamMode.COHERENT:
            for pc in phase_centers:
                print(f"  Phase center: {pc}")
        else:
            print(f"  Number of incoherent beams: {args.n_beams}")

    # Create beamformer and compute weights
    bf = GeometricBeamformer(array_config=array_config, freq_config=freq_config)

    if args.verbose:
        print("Computing weights...")

    if mode == BeamMode.COHERENT:
        weights = bf.compute_weights(
            phase_centers=phase_centers,
            unix_times=unix_times,
            mode=mode
        )
    else:
        weights = bf.compute_weights(
            unix_times=unix_times,
            mode=mode,
            n_beams=args.n_beams
        )

    if args.verbose:
        print(f"Weights shape: {weights.shape}")

    # Determine output format
    output_path = Path(args.output)
    if args.format == "auto":
        if output_path.suffix.lower() in ['.h5', '.hdf5']:
            output_format = "hdf5"
        elif output_path.suffix.lower() == '.npz':
            output_format = "npz"
        else:
            output_format = "hdf5" if HDF5_AVAILABLE else "npz"
    else:
        output_format = args.format

    # Save weights
    if args.verbose:
        print(f"Saving to {args.output} ({output_format} format)...")

    if output_format == "hdf5":
        if not HDF5_AVAILABLE:
            print("Error: h5py not available. Install with: pip install h5py", file=sys.stderr)
            return 1
        save_weights_hdf5(weights, args.output,
                          compression_opts=args.compression,
                          overwrite=args.overwrite)
    else:
        save_weights_npz(weights, args.output)

    if args.verbose:
        file_size = output_path.stat().st_size / 1e6
        print(f"Done. File size: {file_size:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
