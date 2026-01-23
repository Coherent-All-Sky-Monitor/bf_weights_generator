#!/usr/bin/env python
"""
Basic usage examples for the CASM beamformer weights generator.

This script demonstrates:
1. Coherent beamforming weights for a single source
2. Coherent weights for multiple sources (multi-beam)
3. Incoherent beamforming weights
4. Saving and loading weights
5. Flagging antennas
6. Custom frequency configuration

Run with: python basic_usage.py
"""

import numpy as np
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bf_weights_generator import (
    GeometricBeamformer,
    PhaseCenter,
    ArrayConfig,
    FrequencyConfig,
    BeamMode,
    save_weights_hdf5,
    load_weights_hdf5,
    save_weights_npz,
    load_weights_npz,
    inspect_weights_file,
)


def example_coherent_single_source():
    """Example 1: Coherent beamforming for a single source."""
    print("\n" + "=" * 60)
    print("Example 1: Coherent beamforming for a single source")
    print("=" * 60)

    # Create beamformer with default CASM configuration
    bf = GeometricBeamformer()
    print(f"Beamformer: {bf}")

    # Define phase center (Cygnus A)
    cyg_a = PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA")
    print(f"Phase center: {cyg_a}")

    # Define observation time
    # Let's use a specific time: 2024-01-15 00:00:00 UTC
    start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    # Compute weights for 1 hour with 10-second cadence
    weights = bf.compute_weights(
        phase_centers=[cyg_a],
        start_time=start_time,
        duration_sec=3600,  # 1 hour
        cadence_sec=10.0    # 10 second cadence
    )

    print(f"Weights: {weights}")
    print(f"  Shape: {weights.shape}")
    print(f"  Memory: {weights.weights.nbytes / 1e6:.1f} MB")

    # Show weight values at first time, first antenna, first few channels
    print(f"\nSample weights (time=0, ant=0, first 5 channels):")
    print(f"  {weights.weights[0, 0, 0, :5]}")
    print(f"  Magnitudes: {np.abs(weights.weights[0, 0, 0, :5])}")
    print(f"  Phases (deg): {np.angle(weights.weights[0, 0, 0, :5]) * 180/np.pi}")

    return weights


def example_coherent_multi_beam():
    """Example 2: Coherent beamforming for multiple sources (multi-beam)."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-beam coherent beamforming")
    print("=" * 60)

    bf = GeometricBeamformer()

    # Define multiple phase centers
    sources = [
        PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA"),
        PhaseCenter(ra_deg=83.633, dec_deg=22.015, name="TauA"),
        PhaseCenter.from_hours(ra_hours=5.5755, dec_deg=-5.391, name="OrionA"),
    ]

    print("Phase centers:")
    for src in sources:
        print(f"  {src}")

    start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    # Compute weights with shorter duration for demonstration
    weights = bf.compute_weights(
        phase_centers=sources,
        start_time=start_time,
        duration_sec=600,   # 10 minutes
        cadence_sec=30.0    # 30 second cadence
    )

    print(f"\nWeights: {weights}")
    print(f"  Shape: {weights.shape}")
    print(f"  n_beams: {weights.n_beams}")

    return weights


def example_incoherent_beams():
    """Example 3: Incoherent beamforming."""
    print("\n" + "=" * 60)
    print("Example 3: Incoherent beamforming (8 identical beams)")
    print("=" * 60)

    bf = GeometricBeamformer()

    start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    # Compute incoherent weights
    weights = bf.compute_weights(
        start_time=start_time,
        duration_sec=3600,
        cadence_sec=1.0,
        mode='incoherent',
        n_beams=8
    )

    print(f"Weights: {weights}")
    print(f"  Mode: {weights.mode}")
    print(f"  All weights are unity: {np.allclose(weights.weights, 1.0)}")

    print("\nNote: For incoherent beamforming, these unity weights are placeholders.")
    print("At runtime, the actual weight is v.conj(), so beam = v.conj() * v = |v|^2")
    print("This gives total power (sum of squared magnitudes across antennas).")

    return weights


def example_flagging_antennas():
    """Example 4: Flagging bad antennas."""
    print("\n" + "=" * 60)
    print("Example 4: Flagging antennas")
    print("=" * 60)

    # Create array config and flag some antennas
    array_config = ArrayConfig()
    print(f"Before flagging: {array_config}")

    # Flag antennas 3 and 12
    array_config.flag_antennas([3, 12])
    print(f"After flagging [3, 12]: {array_config}")
    print(f"  Active antenna indices: {array_config.active_indices}")

    # Create beamformer with flagged array
    bf = GeometricBeamformer(array_config=array_config)

    cyg_a = PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA")
    start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    weights = bf.compute_weights(
        phase_centers=[cyg_a],
        start_time=start_time,
        duration_sec=60,
        cadence_sec=10.0
    )

    print(f"\nWeights with flagged antennas:")
    print(f"  Shape: {weights.shape}")
    print(f"  n_antennas: {weights.n_antennas} (was 13)")
    print(f"  Antenna indices in weights: {weights.antenna_indices}")

    return weights


def example_custom_frequency():
    """Example 5: Custom frequency configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Custom frequency configuration")
    print("=" * 60)

    # Default configuration
    default_freq = FrequencyConfig()
    print(f"Default: {default_freq}")
    default_freqs = default_freq.get_frequencies_mhz()
    print(f"  Freq range: {default_freqs[-1]:.3f} - {default_freqs[0]:.3f} MHz")

    # Custom configuration - e.g., only use 1024 channels
    custom_freq = FrequencyConfig(
        n_chan=1024,
        total_bw_mhz=125.0,
        total_n_chan=4096,
        freq_end_voltage_mhz=468.75
    )
    print(f"\nCustom: {custom_freq}")
    custom_freqs = custom_freq.get_frequencies_mhz()
    print(f"  Freq range: {custom_freqs[-1]:.3f} - {custom_freqs[0]:.3f} MHz")

    # Create beamformer with custom frequency config
    bf = GeometricBeamformer(freq_config=custom_freq)

    cyg_a = PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA")
    start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    weights = bf.compute_weights(
        phase_centers=[cyg_a],
        start_time=start_time,
        duration_sec=60,
        cadence_sec=10.0
    )

    print(f"\nWeights with custom freq config:")
    print(f"  Shape: {weights.shape}")
    print(f"  n_channels: {weights.n_channels}")

    return weights


def example_save_load():
    """Example 6: Saving and loading weights."""
    print("\n" + "=" * 60)
    print("Example 6: Saving and loading weights")
    print("=" * 60)

    # Generate some weights
    bf = GeometricBeamformer()
    cyg_a = PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA")
    start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    weights = bf.compute_weights(
        phase_centers=[cyg_a],
        start_time=start_time,
        duration_sec=60,
        cadence_sec=10.0
    )

    # Save to HDF5
    hdf5_path = "/tmp/test_weights.h5"
    try:
        save_weights_hdf5(weights, hdf5_path, overwrite=True)
        print(f"Saved to HDF5: {hdf5_path}")

        # Inspect file
        info = inspect_weights_file(hdf5_path)
        print(f"  File size: {info['file_size_mb']:.2f} MB")
        print(f"  Shape: {info['weights_shape']}")

        # Load weights
        loaded = load_weights_hdf5(hdf5_path)
        print(f"\nLoaded from HDF5:")
        print(f"  {loaded}")
        print(f"  Weights match: {np.allclose(weights.weights, loaded.weights)}")
    except ImportError as e:
        print(f"HDF5 not available: {e}")
        print("Install with: pip install h5py")

    # Save to NPZ (always available)
    npz_path = "/tmp/test_weights.npz"
    save_weights_npz(weights, npz_path)
    print(f"\nSaved to NPZ: {npz_path}")

    # Load from NPZ
    loaded_npz = load_weights_npz(npz_path)
    print(f"Loaded from NPZ:")
    print(f"  {loaded_npz}")
    print(f"  Weights match: {np.allclose(weights.weights, loaded_npz.weights)}")


def example_compute_delays():
    """Example 7: Computing geometric delays (for debugging/analysis)."""
    print("\n" + "=" * 60)
    print("Example 7: Computing geometric delays")
    print("=" * 60)

    bf = GeometricBeamformer()

    # Source at zenith (approximately) for OVRO at transit
    # For a source to be at zenith at OVRO (lat ~37.2), it needs dec ~ 37.2
    zenith_source = PhaseCenter(ra_deg=0.0, dec_deg=37.2, name="Zenith")

    # Time when RA=0 transits at OVRO (approximately)
    start_time = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    times = np.array([start_time])

    # Compute delays
    delays = bf.compute_delays(zenith_source, times)

    print(f"Geometric delays for source near zenith:")
    print(f"  Delays (ns): {delays[0] * 1e9}")
    print(f"  Max delay: {np.max(np.abs(delays)) * 1e9:.2f} ns")

    # For a source on the horizon (east)
    east_source = PhaseCenter(ra_deg=90.0, dec_deg=0.0, name="East")
    delays_east = bf.compute_delays(east_source, times)

    print(f"\nGeometric delays for source toward east:")
    print(f"  Delays (ns): {delays_east[0] * 1e9}")
    print(f"  Max delay: {np.max(np.abs(delays_east)) * 1e9:.2f} ns")


def main():
    """Run all examples."""
    print("CASM Beamformer Weights Generator - Examples")
    print("=" * 60)

    example_coherent_single_source()
    example_coherent_multi_beam()
    example_incoherent_beams()
    example_flagging_antennas()
    example_custom_frequency()
    example_save_load()
    example_compute_delays()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
