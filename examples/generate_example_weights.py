#!/usr/bin/env python
"""
Generate example beamformer weights for CASM.

This script demonstrates all four beamforming modes:
1. Stationary Coherent  - Fixed Alt/Az beams for FRB transit search
2. Stationary Incoherent - Fixed total power beams
3. Tracking Coherent    - RA/Dec beams that follow sources
4. Tracking Incoherent  - Total power with time axis

Usage:
    python generate_example_weights.py

Beamforming modes explained:
----------------------------
STATIONARY beams are fixed in the local horizon frame (Alt/Az).
- Weights are TIME-INDEPENDENT (computed once, applied continuously)
- The sky drifts through the beams as Earth rotates
- Ideal for FRB transit searches
- Shape: (n_beams, n_ant, n_chan)

TRACKING beams follow celestial sources (RA/Dec).
- Weights are TIME-DEPENDENT (must be updated as source moves)
- The beam tracks the source position
- Ideal for pulsar timing, source monitoring
- Shape: (n_beams, n_times, n_ant, n_chan)

COHERENT beams apply geometric phase corrections:
- w = exp(-2πi × f × τ)  where τ is the geometric delay
- Steers the beam toward a specific direction

INCOHERENT beams compute total power:
- beam = v* × v = |v|²
- No phase steering, weights are unity
- At runtime, multiply data by its conjugate
"""

import numpy as np
from datetime import datetime, timezone
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bf_weights_generator import (
    GeometricBeamformer,
    PhaseCenter,
    StationaryPointing,
    generate_beam_grid_altaz,
    generate_beam_grid_lm,
    save_weights_hdf5,
)


def example_stationary_coherent():
    """
    Example 1: Stationary Coherent Beams (FRB Transit Search)

    This is the primary mode for real-time FRB searches:
    - Beams are fixed in Alt/Az (local horizon frame)
    - Weights are time-independent
    - Sky drifts through the beams as Earth rotates
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Stationary Coherent Beams (FRB Transit Search)")
    print("=" * 70)

    bf = GeometricBeamformer()

    # Generate beam grid covering the visible sky
    # Default: Alt 30° to 90°, full azimuth, 4° spacing
    pointings = generate_beam_grid_altaz(
        alt_min_deg=30.0,
        alt_max_deg=90.0,
        spacing_deg=4.0
    )
    print(f"\nGenerated {len(pointings)} beam pointings")
    print(f"Sample pointings:")
    for p in pointings[:5]:
        print(f"  {p}")
    print(f"  ...")

    # Compute stationary weights
    weights = bf.compute_stationary_weights(
        pointings=pointings,
        mode='coherent'
    )

    print(f"\nStationary Coherent Weights:")
    print(f"  Type: {type(weights).__name__}")
    print(f"  Shape: {weights.shape}  (n_beams, n_ant, n_chan)")
    print(f"  NO time axis - weights are constant!")
    print(f"  Memory: {weights.weights.nbytes / 1e6:.1f} MB")

    # Save to file
    output_path = "/tmp/casm_stationary_coherent.h5"
    save_weights_hdf5(weights, output_path, overwrite=True)
    print(f"\nSaved to: {output_path}")

    return weights


def example_stationary_coherent_custom():
    """
    Example 1b: Custom Stationary Beam Pointings

    Instead of a grid, specify exact beam directions.
    """
    print("\n" + "-" * 70)
    print("EXAMPLE 1b: Custom Stationary Beam Pointings")
    print("-" * 70)

    bf = GeometricBeamformer()

    # Define specific beam directions
    pointings = [
        StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
        StationaryPointing(alt_deg=60.0, az_deg=0.0, name="north_60"),
        StationaryPointing(alt_deg=60.0, az_deg=90.0, name="east_60"),
        StationaryPointing(alt_deg=60.0, az_deg=180.0, name="south_60"),
        StationaryPointing(alt_deg=60.0, az_deg=270.0, name="west_60"),
        # Can also specify by direction cosines
        StationaryPointing(l=0.0, m=0.5, name="custom_lm"),
    ]

    print(f"\nCustom pointings:")
    for p in pointings:
        print(f"  {p}")
        print(f"    Direction cosines: l={p.l:.4f}, m={p.m:.4f}, n={p.n:.4f}")

    weights = bf.compute_stationary_weights(pointings=pointings, mode='coherent')
    print(f"\nWeights shape: {weights.shape}")

    return weights


def example_stationary_incoherent():
    """
    Example 2: Stationary Incoherent Beams

    Total power beams with no phase steering.
    All beams are identical (just summed antenna power).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Stationary Incoherent Beams")
    print("=" * 70)

    bf = GeometricBeamformer()

    # Compute incoherent weights
    weights = bf.compute_stationary_weights(
        mode='incoherent',
        n_beams=8  # 8 identical total-power beams
    )

    print(f"\nStationary Incoherent Weights:")
    print(f"  Shape: {weights.shape}  (n_beams, n_ant, n_chan)")
    print(f"  All weights are unity: {np.allclose(weights.weights, 1.0)}")
    print(f"\n  At runtime: beam = v* × v = |v|² (total power)")
    print(f"  All 8 beams are identical since there's no steering")

    output_path = "/tmp/casm_stationary_incoherent.h5"
    save_weights_hdf5(weights, output_path, overwrite=True)
    print(f"\nSaved to: {output_path}")

    return weights


def example_tracking_coherent():
    """
    Example 3: Tracking Coherent Beams

    Beams that follow celestial sources (RA/Dec).
    Weights are time-dependent.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Tracking Coherent Beams (Source Monitoring)")
    print("=" * 70)

    bf = GeometricBeamformer()

    # Define celestial sources to track
    sources = [
        PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA"),
        PhaseCenter(ra_deg=83.633, dec_deg=22.015, name="TauA"),
    ]

    print(f"\nTracking sources:")
    for src in sources:
        print(f"  {src}")

    # Observation parameters
    start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    weights = bf.compute_tracking_weights(
        phase_centers=sources,
        start_time=start_time,
        duration_sec=3600,   # 1 hour
        cadence_sec=10.0,    # Update weights every 10 seconds
        mode='coherent'
    )

    print(f"\nTracking Coherent Weights:")
    print(f"  Shape: {weights.shape}  (n_beams, n_times, n_ant, n_chan)")
    print(f"  HAS time axis - weights change as source moves!")
    print(f"  Memory: {weights.weights.nbytes / 1e6:.1f} MB")

    # Show how phases change over time for one antenna
    phases_t0 = np.angle(weights.weights[0, 0, -1, 0]) * 180 / np.pi
    phases_t1 = np.angle(weights.weights[0, -1, -1, 0]) * 180 / np.pi
    print(f"\n  Phase change over 1 hour (ant 12, chan 0):")
    print(f"    t=0: {phases_t0:.1f}°")
    print(f"    t=1h: {phases_t1:.1f}°")
    print(f"    Δφ = {phases_t1 - phases_t0:.1f}°")

    output_path = "/tmp/casm_tracking_coherent.h5"
    save_weights_hdf5(weights, output_path, overwrite=True)
    print(f"\nSaved to: {output_path}")

    return weights


def example_tracking_incoherent():
    """
    Example 4: Tracking Incoherent Beams

    Total power beams with time axis (for compatibility with tracking pipelines).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Tracking Incoherent Beams")
    print("=" * 70)

    bf = GeometricBeamformer()

    start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    weights = bf.compute_tracking_weights(
        start_time=start_time,
        duration_sec=3600,
        cadence_sec=1.0,
        mode='incoherent',
        n_beams=8
    )

    print(f"\nTracking Incoherent Weights:")
    print(f"  Shape: {weights.shape}  (n_beams, n_times, n_ant, n_chan)")
    print(f"  All weights are unity: {np.allclose(weights.weights, 1.0)}")
    print(f"  Memory: {weights.weights.nbytes / 1e9:.2f} GB")

    output_path = "/tmp/casm_tracking_incoherent.h5"
    save_weights_hdf5(weights, output_path, overwrite=True)
    print(f"\nSaved to: {output_path}")

    return weights


def example_beam_grid_comparison():
    """
    Example 5: Compare beam grid generation methods
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Beam Grid Generation Methods")
    print("=" * 70)

    # Alt/Az grid
    altaz_grid = generate_beam_grid_altaz(
        alt_min_deg=30.0,
        alt_max_deg=90.0,
        spacing_deg=4.0
    )

    # Direction cosine (l,m) grid
    lm_grid = generate_beam_grid_lm(
        l_min=-0.8, l_max=0.8,
        m_min=-0.8, m_max=0.8,
        spacing=0.07  # ~4° in direction cosine units
    )

    print(f"\nAlt/Az grid:")
    print(f"  Number of beams: {len(altaz_grid)}")
    print(f"  Altitude range: {min(p.alt_deg for p in altaz_grid):.1f}° - {max(p.alt_deg for p in altaz_grid):.1f}°")
    print(f"  Naturally handles pole crowding (fewer beams near zenith)")

    print(f"\n(l,m) grid:")
    print(f"  Number of beams: {len(lm_grid)}")
    print(f"  l range: {min(p.l for p in lm_grid):.2f} - {max(p.l for p in lm_grid):.2f}")
    print(f"  m range: {min(p.m for p in lm_grid):.2f} - {max(p.m for p in lm_grid):.2f}")
    print(f"  Uniform grid in direction cosine space")


def main():
    """Run all examples."""
    print("=" * 70)
    print("CASM Geometric Beamformer Weight Generator")
    print("All Four Modes Demonstration")
    print("=" * 70)

    # Run all examples
    stationary_coherent = example_stationary_coherent()
    example_stationary_coherent_custom()
    stationary_incoherent = example_stationary_incoherent()
    tracking_coherent = example_tracking_coherent()
    tracking_incoherent = example_tracking_incoherent()
    example_beam_grid_comparison()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Generated weight files:

    1. STATIONARY COHERENT (FRB Transit Search):
       /tmp/casm_stationary_coherent.h5
       Shape: (n_beams, n_ant, n_chan) - NO time axis
       Use case: Real-time FRB search with fixed beam grid

    2. STATIONARY INCOHERENT:
       /tmp/casm_stationary_incoherent.h5
       Shape: (n_beams, n_ant, n_chan) - NO time axis
       Use case: Total power monitoring

    3. TRACKING COHERENT (Source Monitoring):
       /tmp/casm_tracking_coherent.h5
       Shape: (n_beams, n_times, n_ant, n_chan) - HAS time axis
       Use case: Pulsar timing, tracking specific sources

    4. TRACKING INCOHERENT:
       /tmp/casm_tracking_incoherent.h5
       Shape: (n_beams, n_times, n_ant, n_chan) - HAS time axis
       Use case: Total power with time-stamped output

    Key differences:
    ┌─────────────┬─────────────────┬─────────────────────┐
    │             │ COHERENT        │ INCOHERENT          │
    ├─────────────┼─────────────────┼─────────────────────┤
    │ STATIONARY  │ Phase steering  │ Total power         │
    │ (Alt/Az)    │ Time-independent│ Time-independent    │
    │             │ For FRB search  │ All beams identical │
    ├─────────────┼─────────────────┼─────────────────────┤
    │ TRACKING    │ Phase steering  │ Total power         │
    │ (RA/Dec)    │ Time-dependent  │ Time-dependent      │
    │             │ Source tracking │ With timestamps     │
    └─────────────┴─────────────────┴─────────────────────┘
    """)


if __name__ == "__main__":
    main()
