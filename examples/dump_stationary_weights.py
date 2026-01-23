#!/usr/bin/env python3
"""
Example: Dump stationary beam weights at a fixed Alt/Az pointing.

This script demonstrates how to:
1. Define antenna positions from SNAP ADC mapping
2. Create a stationary beam pointing (e.g., zenith)
3. Compute geometric weights for the stationary beam
4. Save weights to HDF5 or NPZ format

Usage:
    # Zenith pointing, save as HDF5
    python dump_stationary_weights.py --alt 90 --az 0 --output zenith_weights.h5

    # South at 60° altitude, save as NPZ
    python dump_stationary_weights.py --alt 60 --az 180 --output south60_weights.npz

    # Multiple beams (grid), save as HDF5
    python dump_stationary_weights.py --grid --output beam_grid.h5
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, '/home/casm/software/vishnu/bf_weights_generator')

from bf_weights_generator import (
    GeometricBeamformer,
    StationaryPointing,
    ArrayConfig,
    FrequencyConfig,
    save_weights_hdf5,
    save_weights_npz,
)

# =============================================================================
# SNAP 2 ADC to Antenna Mapping
# =============================================================================
SNAP2_ADC_TO_ANTENNA = {
    1: (10, (1.65, -10.5, 0.0)),
    2: (12, (-4.553, -5.5, -0.279)),
    3: (4, (1.65, 0.0, 0.0)),
    6: (6, (0.0, -10.5, 0.0)),
    8: (8, (0.825, -10.5, 0.0)),
    9: (11, (2.03, -10.5, 0.0)),
    11: (7, (0.38, -10.5, 0.0)),
}

ACTIVE_ADC_CHANNELS = sorted(SNAP2_ADC_TO_ANTENNA.keys())


def get_snap2_antenna_config():
    """Get ArrayConfig for SNAP 2 antennas."""
    positions = []
    for adc_ch in ACTIVE_ADC_CHANNELS:
        _, pos = SNAP2_ADC_TO_ANTENNA[adc_ch]
        positions.append(pos)

    return ArrayConfig(
        positions_enu=np.array(positions),
        antenna_flags=np.ones(len(positions), dtype=bool)
    )


def dump_single_beam(
    alt_deg: float,
    az_deg: float,
    output_path: str,
    name: str = "stationary",
    mode: str = "coherent",
):
    """
    Compute and save stationary beam weights for a single pointing.

    Parameters
    ----------
    alt_deg : float
        Altitude in degrees (0=horizon, 90=zenith)
    az_deg : float
        Azimuth in degrees (0=North, 90=East)
    output_path : str
        Output file path (.h5 or .npz)
    name : str
        Name for the pointing
    mode : str
        'coherent' or 'incoherent'
    """
    print("=" * 60)
    print("Dumping Stationary Beam Weights")
    print("=" * 60)

    # Setup
    array_config = get_snap2_antenna_config()
    freq_config = FrequencyConfig(
        n_chan=3072,
        total_bw_mhz=125.0,
        total_n_chan=4096,
        freq_end_voltage_mhz=468.75
    )

    print(f"\nArray: {array_config.n_antennas} antennas")
    print(f"Frequencies: {freq_config.n_chan} channels, "
          f"{freq_config.get_frequencies_mhz()[0]:.2f} - "
          f"{freq_config.get_frequencies_mhz()[-1]:.2f} MHz")

    # Create beamformer
    bf = GeometricBeamformer(
        array_config=array_config,
        freq_config=freq_config,
    )

    # Create pointing
    pointing = StationaryPointing(alt_deg=alt_deg, az_deg=az_deg, name=name)

    print(f"\nPointing: {name}")
    print(f"  Alt = {alt_deg:.2f}°, Az = {az_deg:.2f}°")
    print(f"  Direction cosines: l={pointing.l:.4f}, m={pointing.m:.4f}, n={pointing.n:.4f}")

    # Compute weights
    print(f"\nComputing {mode} weights...")
    weights = bf.compute_stationary_weights(
        pointings=[pointing],
        mode=mode,
    )

    print(f"  Weights shape: {weights.weights.shape}")
    print(f"  (n_beams={weights.n_beams}, n_ant={weights.n_antennas}, n_chan={weights.n_channels})")

    # Print delays
    delays = bf.compute_stationary_delays(pointing)
    print(f"\n  Geometric delays (ns):")
    for i, adc_ch in enumerate(ACTIVE_ADC_CHANNELS):
        ant_idx = SNAP2_ADC_TO_ANTENNA[adc_ch][0]
        print(f"    ADC {adc_ch:2d} (Ant {ant_idx:2d}): {delays[i]*1e9:8.3f} ns")

    # Save weights
    print(f"\nSaving to: {output_path}")
    if output_path.endswith('.h5') or output_path.endswith('.hdf5'):
        save_weights_hdf5(weights, output_path, overwrite=True)
        print("  Format: HDF5")
    else:
        save_weights_npz(weights, output_path)
        print("  Format: NPZ")

    print("\nDone!")
    return weights


def dump_beam_grid(
    output_path: str,
    alt_min: float = 30.0,
    alt_max: float = 90.0,
    alt_step: float = 15.0,
    az_step: float = 45.0,
    mode: str = "coherent",
):
    """
    Compute and save weights for a grid of stationary beams.

    Parameters
    ----------
    output_path : str
        Output file path (.h5 or .npz)
    alt_min, alt_max : float
        Altitude range in degrees
    alt_step : float
        Altitude step in degrees
    az_step : float
        Azimuth step in degrees
    mode : str
        'coherent' or 'incoherent'
    """
    print("=" * 60)
    print("Dumping Stationary Beam Grid")
    print("=" * 60)

    # Setup
    array_config = get_snap2_antenna_config()
    freq_config = FrequencyConfig(
        n_chan=3072,
        total_bw_mhz=125.0,
        total_n_chan=4096,
        freq_end_voltage_mhz=468.75
    )

    bf = GeometricBeamformer(
        array_config=array_config,
        freq_config=freq_config,
    )

    # Generate beam grid
    pointings = []
    alts = np.arange(alt_min, alt_max + 0.1, alt_step)

    for alt in alts:
        if alt >= 89.0:
            # Near zenith, just one beam
            pointings.append(StationaryPointing(
                alt_deg=90.0, az_deg=0.0, name=f"alt90_az0"
            ))
        else:
            # Full azimuth coverage
            n_az = int(360.0 / az_step)
            for i in range(n_az):
                az = i * az_step
                pointings.append(StationaryPointing(
                    alt_deg=alt, az_deg=az, name=f"alt{int(alt)}_az{int(az)}"
                ))

    print(f"\nGenerated {len(pointings)} beam pointings")
    print(f"  Altitude range: {alt_min}° to {alt_max}° (step {alt_step}°)")
    print(f"  Azimuth step: {az_step}°")

    # Compute weights
    print(f"\nComputing {mode} weights...")
    weights = bf.compute_stationary_weights(
        pointings=pointings,
        mode=mode,
    )

    print(f"  Weights shape: {weights.weights.shape}")
    print(f"  (n_beams={weights.n_beams}, n_ant={weights.n_antennas}, n_chan={weights.n_channels})")

    # Save
    print(f"\nSaving to: {output_path}")
    if output_path.endswith('.h5') or output_path.endswith('.hdf5'):
        save_weights_hdf5(weights, output_path, overwrite=True)
        print("  Format: HDF5")
    else:
        save_weights_npz(weights, output_path)
        print("  Format: NPZ")

    # Print beam summary
    print(f"\nBeam summary:")
    for i, p in enumerate(pointings[:10]):
        print(f"  Beam {i:3d}: Alt={p.alt_deg:5.1f}°, Az={p.az_deg:5.1f}° ({p.name})")
    if len(pointings) > 10:
        print(f"  ... and {len(pointings) - 10} more beams")

    print("\nDone!")
    return weights


def main():
    parser = argparse.ArgumentParser(
        description='Dump stationary beam weights to HDF5 or NPZ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single beam at zenith, save as HDF5
  python dump_stationary_weights.py --alt 90 --az 0 -o zenith.h5

  # Single beam at zenith, save as NPZ
  python dump_stationary_weights.py --alt 90 --az 0 -o zenith.npz

  # Beam pointing South at 60° altitude
  python dump_stationary_weights.py --alt 60 --az 180 --name South60 -o south60.h5

  # Incoherent weights (unity)
  python dump_stationary_weights.py --alt 90 --az 0 --mode incoherent -o incoh.h5

  # Generate a grid of beams
  python dump_stationary_weights.py --grid -o beam_grid.h5

  # Custom grid parameters
  python dump_stationary_weights.py --grid --alt-min 45 --alt-max 90 --alt-step 15 --az-step 30 -o grid.h5
        """
    )

    # Single beam options
    parser.add_argument('--alt', type=float, help='Altitude in degrees (0-90)')
    parser.add_argument('--az', type=float, help='Azimuth in degrees (0-360)')
    parser.add_argument('--name', type=str, default='stationary', help='Beam name')

    # Grid options
    parser.add_argument('--grid', action='store_true', help='Generate a grid of beams')
    parser.add_argument('--alt-min', type=float, default=30.0, help='Min altitude for grid')
    parser.add_argument('--alt-max', type=float, default=90.0, help='Max altitude for grid')
    parser.add_argument('--alt-step', type=float, default=15.0, help='Altitude step for grid')
    parser.add_argument('--az-step', type=float, default=45.0, help='Azimuth step for grid')

    # Common options
    parser.add_argument('--mode', choices=['coherent', 'incoherent'], default='coherent',
                        help='Beamforming mode')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output file (.h5/.hdf5 for HDF5, .npz for NumPy)')

    args = parser.parse_args()

    if args.grid:
        dump_beam_grid(
            output_path=args.output,
            alt_min=args.alt_min,
            alt_max=args.alt_max,
            alt_step=args.alt_step,
            az_step=args.az_step,
            mode=args.mode,
        )
    else:
        if args.alt is None or args.az is None:
            parser.error("Must specify --alt and --az for single beam, or use --grid")

        dump_single_beam(
            alt_deg=args.alt,
            az_deg=args.az,
            output_path=args.output,
            name=args.name,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
