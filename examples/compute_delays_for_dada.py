#!/usr/bin/env python3
"""
Example: Compute geometric delays for CASM voltage data.

This script demonstrates how to:
1. Define antenna positions from ADC channel mapping
2. Compute geometric delays for a given source (tracking mode)
3. Compute beamformer weights (coherent or incoherent)
4. Apply weights to voltage data from a DADA file

Usage:
    python compute_delays_for_dada.py --dada /path/to/file.dada --source casa
    python compute_delays_for_dada.py --dada /path/to/file.dada --ra 350.85 --dec 58.815
    python compute_delays_for_dada.py --dada /path/to/file.dada --mode incoherent
"""

import sys
import argparse
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add paths
sys.path.insert(0, '/home/casm/software/vishnu/OVRO_DATA_EXPERIMENTS_DEC_2025/VOLTAGE_ANALYSIS')
sys.path.insert(0, '/home/casm/software/vishnu/bf_weights_generator')

from bf_weights_generator import (
    GeometricBeamformer,
    PhaseCenter,
    StationaryPointing,
    ArrayConfig,
    FrequencyConfig,
)

# =============================================================================
# SNAP ADC to Antenna Mapping
# =============================================================================
# This mapping comes from casm_adc_to_ant_map_5JAN26.csv
# Format: {snap_id: {adc_channel: (antenna_index, (x, y, z) in ENU meters)}}

SNAP_ADC_MAPPING = {
    2: {
        1: (10, (1.65, -10.5, 0.0)),
        2: (12, (-4.553, -5.5, -0.279)),
        3: (4, (1.65, 0.0, 0.0)),
        6: (6, (0.0, -10.5, 0.0)),
        8: (8, (0.825, -10.5, 0.0)),
        9: (11, (2.03, -10.5, 0.0)),
        11: (7, (0.38, -10.5, 0.0)),
    },
    # Add other SNAPs as needed:
    # 0: {...},
    # 1: {...},
}

# Common source coordinates (J2000)
KNOWN_SOURCES = {
    'casa': PhaseCenter(ra_deg=350.850, dec_deg=58.815, name='CasA'),
    'cyga': PhaseCenter(ra_deg=299.868, dec_deg=40.734, name='CygA'),
    'taua': PhaseCenter(ra_deg=83.633, dec_deg=22.015, name='TauA'),
    'vira': PhaseCenter(ra_deg=187.706, dec_deg=12.391, name='VirA'),
    'sun': PhaseCenter(ra_deg=0.0, dec_deg=0.0, name='Sun'),  # Placeholder
}


@dataclass
class AntennaMapping:
    """Container for antenna mapping from ADC channels."""
    adc_channels: List[int]      # ADC channel indices
    antenna_indices: List[int]   # Corresponding antenna indices
    positions_enu: np.ndarray    # ENU positions (n_ant, 3)
    snap_id: int                 # SNAP board ID

    @classmethod
    def from_snap(cls, snap_id: int) -> 'AntennaMapping':
        """Create mapping for a specific SNAP board."""
        if snap_id not in SNAP_ADC_MAPPING:
            raise ValueError(f"No mapping defined for SNAP {snap_id}")

        mapping = SNAP_ADC_MAPPING[snap_id]
        adc_channels = sorted(mapping.keys())
        antenna_indices = [mapping[ch][0] for ch in adc_channels]
        positions = np.array([mapping[ch][1] for ch in adc_channels])

        return cls(
            adc_channels=adc_channels,
            antenna_indices=antenna_indices,
            positions_enu=positions,
            snap_id=snap_id
        )

    @property
    def n_antennas(self) -> int:
        return len(self.adc_channels)

    def __repr__(self) -> str:
        return (f"AntennaMapping(snap={self.snap_id}, n_ant={self.n_antennas}, "
                f"adc_channels={self.adc_channels})")


def parse_utc_start(utc_str: str) -> float:
    """Parse UTC_START string to Unix timestamp."""
    # Format: "2026-01-21-01:16:44"
    dt = datetime.strptime(utc_str, "%Y-%m-%d-%H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def compute_stationary_delays(
    pointing: StationaryPointing,
    antenna_mapping: AntennaMapping,
    freq_config: Optional[FrequencyConfig] = None,
) -> Dict:
    """
    Compute geometric delays for a stationary beam at fixed Alt/Az.

    Parameters
    ----------
    pointing : StationaryPointing
        Target direction in Alt/Az or direction cosines
    antenna_mapping : AntennaMapping
        Antenna positions and indices
    freq_config : FrequencyConfig, optional
        Frequency configuration (default: CASM standard)

    Returns
    -------
    dict with keys:
        'delays_sec': np.ndarray (n_ant,) - geometric delays in seconds
        'delays_ns': np.ndarray (n_ant,) - geometric delays in nanoseconds
        'phases_rad': np.ndarray (n_ant, n_chan) - phase corrections
        'weights': np.ndarray (n_ant, n_chan) - complex beamformer weights
        'frequencies_mhz': np.ndarray (n_chan,) - frequency axis
    """
    if freq_config is None:
        freq_config = FrequencyConfig(
            n_chan=3072,
            total_bw_mhz=125.0,
            total_n_chan=4096,
            freq_end_voltage_mhz=468.75
        )

    # Create array config
    array_config = ArrayConfig(
        positions_enu=antenna_mapping.positions_enu,
        antenna_flags=np.ones(antenna_mapping.n_antennas, dtype=bool)
    )

    # Create beamformer
    bf = GeometricBeamformer(
        array_config=array_config,
        freq_config=freq_config,
        use_astropy=False  # Not needed for stationary beams
    )

    # Compute delays (in seconds)
    delays_sec = bf.compute_stationary_delays(pointing)

    # Compute phases
    phases_rad = bf.compute_stationary_phases(pointing)

    # Compute weights
    weights = np.exp(-1j * phases_rad).astype(np.complex64)

    return {
        'delays_sec': delays_sec,
        'delays_ns': delays_sec * 1e9,
        'phases_rad': phases_rad,
        'weights': weights,
        'frequencies_mhz': freq_config.get_frequencies_mhz(),
        'pointing': pointing,
        'antenna_mapping': antenna_mapping,
        'beam_type': 'stationary',
    }


def compute_geometric_delays(
    source: PhaseCenter,
    obs_time: float,
    antenna_mapping: AntennaMapping,
    freq_config: Optional[FrequencyConfig] = None,
) -> Dict:
    """
    Compute geometric delays for a source at a given time.

    Parameters
    ----------
    source : PhaseCenter
        Target source (RA/Dec in J2000)
    obs_time : float
        Unix timestamp of observation
    antenna_mapping : AntennaMapping
        Antenna positions and indices
    freq_config : FrequencyConfig, optional
        Frequency configuration (default: CASM standard)

    Returns
    -------
    dict with keys:
        'delays_sec': np.ndarray (n_ant,) - geometric delays in seconds
        'delays_ns': np.ndarray (n_ant,) - geometric delays in nanoseconds
        'phases_rad': np.ndarray (n_ant, n_chan) - phase corrections
        'weights': np.ndarray (n_ant, n_chan) - complex beamformer weights
        'frequencies_mhz': np.ndarray (n_chan,) - frequency axis
    """
    if freq_config is None:
        freq_config = FrequencyConfig(
            n_chan=3072,
            total_bw_mhz=125.0,
            total_n_chan=4096,
            freq_end_voltage_mhz=468.75
        )

    # Create array config
    array_config = ArrayConfig(
        positions_enu=antenna_mapping.positions_enu,
        antenna_flags=np.ones(antenna_mapping.n_antennas, dtype=bool)
    )

    # Create beamformer
    bf = GeometricBeamformer(
        array_config=array_config,
        freq_config=freq_config,
        use_astropy=True
    )

    # Compute delays (in seconds)
    delays_sec = bf.compute_tracking_delays(source, np.array([obs_time]))
    delays_sec = delays_sec[0]  # Remove time axis -> (n_ant,)

    # Compute phases
    phases_rad = bf.compute_tracking_phases(source, np.array([obs_time]))
    phases_rad = phases_rad[0]  # Remove time axis -> (n_ant, n_chan)

    # Compute weights
    weights = np.exp(-1j * phases_rad).astype(np.complex64)

    return {
        'delays_sec': delays_sec,
        'delays_ns': delays_sec * 1e9,
        'phases_rad': phases_rad,
        'weights': weights,
        'frequencies_mhz': freq_config.get_frequencies_mhz(),
        'source': source,
        'obs_time': obs_time,
        'antenna_mapping': antenna_mapping,
    }


def compute_incoherent_weights(
    antenna_mapping: AntennaMapping,
    freq_config: Optional[FrequencyConfig] = None,
) -> Dict:
    """
    Compute incoherent (unity) weights.

    Parameters
    ----------
    antenna_mapping : AntennaMapping
        Antenna positions and indices
    freq_config : FrequencyConfig, optional
        Frequency configuration

    Returns
    -------
    dict with 'weights' key containing unity weights
    """
    if freq_config is None:
        freq_config = FrequencyConfig(
            n_chan=3072,
            total_bw_mhz=125.0,
            total_n_chan=4096,
            freq_end_voltage_mhz=468.75
        )

    n_ant = antenna_mapping.n_antennas
    n_chan = freq_config.n_chan

    weights = np.ones((n_ant, n_chan), dtype=np.complex64)

    return {
        'weights': weights,
        'frequencies_mhz': freq_config.get_frequencies_mhz(),
        'antenna_mapping': antenna_mapping,
    }


def print_delay_table(result: Dict) -> None:
    """Print geometric delays in a readable table."""
    mapping = result['antenna_mapping']
    delays_ns = result['delays_ns']

    # Check if this is a stationary or tracking beam
    beam_type = result.get('beam_type', 'tracking')

    print("\n" + "=" * 70)
    if beam_type == 'stationary':
        pointing = result['pointing']
        print(f"GEOMETRIC DELAYS for STATIONARY BEAM")
        print(f"Alt = {pointing.alt_deg:.2f}°, Az = {pointing.az_deg:.2f}°")
        print(f"Direction cosines: l={pointing.l:.4f}, m={pointing.m:.4f}, n={pointing.n:.4f}")
    else:
        source = result['source']
        print(f"GEOMETRIC DELAYS for {source.name}")
        print(f"RA = {source.ra_deg:.4f}°, Dec = {source.dec_deg:.4f}°")
    print("=" * 70)
    print(f"\n{'ADC Ch':<8} {'Ant Idx':<10} {'Position (E,N,U) [m]':<30} {'Delay [ns]':<12}")
    print("-" * 70)

    for i, (adc_ch, ant_idx) in enumerate(zip(mapping.adc_channels, mapping.antenna_indices)):
        pos = mapping.positions_enu[i]
        delay = delays_ns[i]
        print(f"{adc_ch:<8} {ant_idx:<10} ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})    {delay:>10.3f}")

    print("-" * 70)
    print(f"Reference: Antenna with delay closest to zero")
    ref_idx = np.argmin(np.abs(delays_ns))
    print(f"  -> ADC channel {mapping.adc_channels[ref_idx]} "
          f"(Antenna {mapping.antenna_indices[ref_idx]})")
    print(f"\nDelay range: {delays_ns.min():.3f} to {delays_ns.max():.3f} ns")
    print(f"Max baseline delay: {delays_ns.max() - delays_ns.min():.3f} ns")
    print("=" * 70)


def save_delays_to_file(result: Dict, output_path: str) -> None:
    """Save delays and weights to NPZ file."""
    beam_type = result.get('beam_type', 'tracking')

    save_dict = {
        'delays_sec': result['delays_sec'],
        'delays_ns': result['delays_ns'],
        'phases_rad': result['phases_rad'],
        'weights': result['weights'],
        'frequencies_mhz': result['frequencies_mhz'],
        'adc_channels': result['antenna_mapping'].adc_channels,
        'antenna_indices': result['antenna_mapping'].antenna_indices,
        'positions_enu': result['antenna_mapping'].positions_enu,
        'beam_type': beam_type,
    }

    if beam_type == 'stationary':
        pointing = result['pointing']
        save_dict['alt_deg'] = pointing.alt_deg
        save_dict['az_deg'] = pointing.az_deg
        save_dict['l'] = pointing.l
        save_dict['m'] = pointing.m
        save_dict['n'] = pointing.n
        save_dict['pointing_name'] = pointing.name
    else:
        save_dict['source_ra_deg'] = result['source'].ra_deg
        save_dict['source_dec_deg'] = result['source'].dec_deg
        save_dict['source_name'] = result['source'].name
        save_dict['obs_time'] = result['obs_time']

    np.savez(output_path, **save_dict)
    print(f"\nSaved delays to: {output_path}")


# =============================================================================
# Example: Apply weights to DADA voltage data
# =============================================================================

def apply_weights_to_voltages(
    voltages: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Apply beamformer weights to voltage data.

    Parameters
    ----------
    voltages : np.ndarray
        Shape (n_time, n_chan, n_ant) - complex voltages
    weights : np.ndarray
        Shape (n_ant, n_chan) - complex weights

    Returns
    -------
    beamformed : np.ndarray
        Shape (n_time, n_chan) - beamformed voltages
    """
    # weights: (n_ant, n_chan) -> transpose to (n_chan, n_ant)
    # voltages: (n_time, n_chan, n_ant)
    # Result: sum over antennas of (weight * voltage)

    # Transpose weights for broadcasting
    w = weights.T  # (n_chan, n_ant)

    # Apply weights and sum over antennas
    beamformed = np.sum(voltages * w[np.newaxis, :, :], axis=2)

    return beamformed


def example_with_dada_file(
    dada_file: str,
    source: PhaseCenter,
    snap_id: int = 2,
    n_time: int = 8192,
    mode: str = 'coherent',
) -> Dict:
    """
    Complete example: load DADA file, compute weights, apply beamforming.

    Parameters
    ----------
    dada_file : str
        Path to DADA file
    source : PhaseCenter
        Target source (ignored if mode='incoherent')
    snap_id : int
        SNAP board to use
    n_time : int
        Number of time samples to load
    mode : str
        'coherent' or 'incoherent'

    Returns
    -------
    dict with beamformed data and metadata
    """
    from casm_io import read_dada_header, read_dada_data

    print(f"\n{'='*70}")
    print(f"CASM Beamformer Example")
    print(f"{'='*70}")
    print(f"DADA file: {dada_file}")
    print(f"Mode: {mode}")
    if mode == 'coherent':
        print(f"Source: {source.name} (RA={source.ra_deg}°, Dec={source.dec_deg}°)")

    # Step 1: Read header
    header = read_dada_header(dada_file)
    obs_time = parse_utc_start(header['UTC_START'])
    obs_datetime = datetime.fromtimestamp(obs_time, tz=timezone.utc)
    print(f"Observation time: {obs_datetime.isoformat()}")

    # Step 2: Get antenna mapping
    mapping = AntennaMapping.from_snap(snap_id)
    print(f"Antenna mapping: {mapping}")

    # Step 3: Compute weights
    if mode == 'coherent':
        result = compute_geometric_delays(source, obs_time, mapping)
        print_delay_table(result)
        weights = result['weights']
    else:
        result = compute_incoherent_weights(mapping)
        weights = result['weights']
        print("\nUsing incoherent (unity) weights")

    # Step 4: Load voltage data
    print(f"\nLoading {n_time} time samples from SNAP {snap_id}...")
    voltages_dict, _ = read_dada_data(dada_file, n_time=n_time, snaps=[snap_id])

    # Extract active ADC channels
    snap_data = voltages_dict[snap_id]  # (n_time, n_chan, 12)
    voltages = snap_data[:, :, mapping.adc_channels]  # (n_time, n_chan, n_ant)
    print(f"Voltages shape: {voltages.shape}")

    # Step 5: Apply beamforming
    print("\nApplying beamformer weights...")
    beamformed = apply_weights_to_voltages(voltages, weights)
    print(f"Beamformed shape: {beamformed.shape}")

    # Step 6: Compute power
    power = np.abs(beamformed) ** 2
    mean_power = power.mean()
    print(f"Mean beamformed power: {mean_power:.6f}")

    return {
        'beamformed': beamformed,
        'power': power,
        'weights': weights,
        'voltages': voltages,
        'header': header,
        'obs_time': obs_time,
        'mapping': mapping,
        'mode': mode,
        'source': source if mode == 'coherent' else None,
    }


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute geometric delays for CASM voltage data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute delays for Cas A (tracking beam)
  python compute_delays_for_dada.py --source casa

  # Compute delays for custom RA/Dec coordinates
  python compute_delays_for_dada.py --ra 350.85 --dec 58.815 --name MySource

  # Compute delays for stationary beam at Alt/Az
  python compute_delays_for_dada.py --alt 60 --az 180 --name Zenith60

  # Stationary beam at zenith
  python compute_delays_for_dada.py --alt 90 --az 0 --name Zenith

  # Process a DADA file with coherent beamforming
  python compute_delays_for_dada.py --dada /path/to/file.dada --source casa

  # Process with incoherent beamforming
  python compute_delays_for_dada.py --dada /path/to/file.dada --mode incoherent

  # Save delays to file
  python compute_delays_for_dada.py --source cyga --output delays.npz
  python compute_delays_for_dada.py --alt 45 --az 90 --output stationary_delays.npz
        """
    )

    # Source specification (tracking beams)
    parser.add_argument('--source', type=str, choices=list(KNOWN_SOURCES.keys()),
                        help='Known source name (casa, cyga, taua, vira)')
    parser.add_argument('--ra', type=float, help='Right Ascension in degrees (J2000)')
    parser.add_argument('--dec', type=float, help='Declination in degrees (J2000)')

    # Stationary beam specification (Alt/Az)
    parser.add_argument('--alt', type=float,
                        help='Altitude in degrees (0=horizon, 90=zenith) for stationary beam')
    parser.add_argument('--az', type=float,
                        help='Azimuth in degrees (0=North, 90=East) for stationary beam')

    # Common options
    parser.add_argument('--name', type=str, default='Custom', help='Source/pointing name')

    # Time specification (for tracking beams)
    parser.add_argument('--time', type=str,
                        help='Observation time (ISO format or Unix timestamp)')

    # DADA file processing
    parser.add_argument('--dada', type=str, help='Path to DADA file')
    parser.add_argument('--snap', type=int, default=2, help='SNAP board ID (default: 2)')
    parser.add_argument('--n-time', type=int, default=8192,
                        help='Number of time samples to process')

    # Beamforming mode
    parser.add_argument('--mode', type=str, choices=['coherent', 'incoherent'],
                        default='coherent', help='Beamforming mode')

    # Output
    parser.add_argument('--output', '-o', type=str, help='Output file for delays (.npz)')

    args = parser.parse_args()

    # Determine beam type: stationary (Alt/Az) or tracking (RA/Dec)
    is_stationary = args.alt is not None or args.az is not None
    is_tracking = args.source is not None or args.ra is not None or args.dec is not None

    if is_stationary and is_tracking:
        parser.error("Cannot specify both Alt/Az (stationary) and RA/Dec/source (tracking). Choose one.")

    # Determine source/pointing
    source = None
    pointing = None

    if is_stationary:
        if args.alt is None or args.az is None:
            parser.error("Must specify both --alt and --az for stationary beam")
        pointing = StationaryPointing(alt_deg=args.alt, az_deg=args.az, name=args.name)
    elif args.source:
        source = KNOWN_SOURCES[args.source]
    elif args.ra is not None and args.dec is not None:
        source = PhaseCenter(ra_deg=args.ra, dec_deg=args.dec, name=args.name)
    elif args.mode == 'incoherent':
        source = None
    else:
        parser.error("Must specify --source, --ra/--dec, or --alt/--az for coherent mode")

    # Get antenna mapping
    mapping = AntennaMapping.from_snap(args.snap)

    # Process DADA file if provided
    if args.dada:
        if is_stationary:
            # For stationary beams with DADA file, compute delays and apply
            print(f"\n{'='*70}")
            print(f"CASM Beamformer Example (Stationary Beam)")
            print(f"{'='*70}")
            print(f"DADA file: {args.dada}")
            print(f"Mode: {args.mode}")
            print(f"Pointing: Alt={pointing.alt_deg}°, Az={pointing.az_deg}°")

            result = compute_stationary_delays(pointing, mapping)
            print_delay_table(result)

            if args.output:
                save_delays_to_file(result, args.output)
        else:
            result = example_with_dada_file(
                dada_file=args.dada,
                source=source,
                snap_id=args.snap,
                n_time=args.n_time,
                mode=args.mode,
            )

            if args.output and args.mode == 'coherent':
                # Also save the delays
                delay_result = compute_geometric_delays(source, result['obs_time'], mapping)
                save_delays_to_file(delay_result, args.output)

    else:
        # Just compute and display delays
        if args.mode == 'incoherent':
            result = compute_incoherent_weights(mapping)
            print("\nIncoherent weights: all ones (unity)")
            print(f"Shape: {result['weights'].shape}")
        elif is_stationary:
            # Stationary beam - no time needed
            result = compute_stationary_delays(pointing, mapping)
            print_delay_table(result)

            if args.output:
                save_delays_to_file(result, args.output)
        else:
            # Tracking beam - needs observation time
            if args.time:
                try:
                    obs_time = float(args.time)
                except ValueError:
                    dt = datetime.fromisoformat(args.time.replace('Z', '+00:00'))
                    obs_time = dt.timestamp()
            else:
                # Use current time
                obs_time = datetime.now(timezone.utc).timestamp()
                print(f"Using current time: {datetime.fromtimestamp(obs_time, timezone.utc).isoformat()}")

            result = compute_geometric_delays(source, obs_time, mapping)
            print_delay_table(result)

            if args.output:
                save_delays_to_file(result, args.output)


if __name__ == "__main__":
    main()
