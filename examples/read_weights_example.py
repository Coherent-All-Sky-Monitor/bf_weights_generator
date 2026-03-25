#!/usr/bin/env python3
"""
Example: How to read and extract data from CASM int8 weights HDF5 files.

Demonstrates reading beam pointings, weights, frequency axis, antenna layout,
and SNAP mapping from the HDF5 output.

Usage:
    python examples/read_weights_example.py weights_512beam/int8_weights_512beam_feb14_svd.h5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Read and inspect CASM int8 weights HDF5 file.",
    )
    parser.add_argument("weights", help="Path to int8 weights HDF5 file")
    args = parser.parse_args()

    f = h5py.File(args.weights, "r")

    # =====================================================================
    # 1. ROOT METADATA
    # =====================================================================
    print("=" * 60)
    print("1. FILE METADATA")
    print("=" * 60)
    print(f"  File:           {args.weights}")
    print(f"  Format:         {f.attrs['format_type']}")
    print(f"  Version:        {f.attrs['version']}")
    print(f"  Created:        {f.attrs['created_utc']}")
    print(f"  N beams:        {f.attrs['n_beams']}")
    print(f"  N channels:     {f.attrs['n_channels']}")
    print(f"  N polarizations:{f.attrs['n_pol']}")
    print(f"  N antennas:     {f.attrs['n_antennas']} (64 SNAP slots)")
    print(f"  Scale factor:   {f.attrs['scale_factor']}")

    # =====================================================================
    # 2. WEIGHTS — shape (2, n_chan, 2, n_beams, 64)
    #    axis 0: 0=real, 1=imag
    #    axis 1: frequency channel
    #    axis 2: polarization (0=A, 1=B, both identical)
    #    axis 3: beam index
    #    axis 4: SNAP packet index (snap_board * 12 + adc_channel)
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("2. WEIGHTS ARRAY")
    print("=" * 60)
    w = f["weights_int8"]  # don't load into memory yet
    print(f"  Shape:  {w.shape}")
    print(f"  Dtype:  {w.dtype}")
    print(f"  Layout: (real/imag, channels, pol, beams, antennas)")

    # Load into memory
    w_int8 = w[:]
    scale = f.attrs["scale_factor"]  # 127.0

    # Extract a single beam (beam 0, pol A) as complex64
    beam0_real = w_int8[0, :, 0, 0, :].astype(np.float32)  # (n_chan, 64)
    beam0_imag = w_int8[1, :, 0, 0, :].astype(np.float32)
    beam0_complex = (beam0_real + 1j * beam0_imag) / scale  # unit amplitude
    print(f"\n  Example: beam 0, pol A → complex64 shape {beam0_complex.shape}")
    print(f"  Max |w|: {np.max(np.abs(beam0_complex)):.4f}")

    # Extract all beams as complex64: (n_beams, 64, n_chan)
    all_real = w_int8[0, :, 0, :, :].astype(np.float32)  # (n_chan, n_beams, 64)
    all_imag = w_int8[1, :, 0, :, :].astype(np.float32)
    all_complex = (all_real + 1j * all_imag) / scale
    all_complex = all_complex.transpose(1, 2, 0)  # (n_beams, 64, n_chan)
    print(f"  All beams complex64: {all_complex.shape} = (beams, ants, chans)")

    # =====================================================================
    # 3. FREQUENCY AXIS
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("3. FREQUENCY AXIS")
    print("=" * 60)
    freqs_hz = f["frequencies_hz"][:]
    freqs_mhz = freqs_hz / 1e6
    print(f"  N channels: {len(freqs_hz)}")
    print(f"  Range:      {freqs_mhz[0]:.2f} → {freqs_mhz[-1]:.2f} MHz")
    if freqs_mhz[0] > freqs_mhz[-1]:
        print(f"  Order:      descending (high → low)")
    else:
        print(f"  Order:      ascending (low → high)")
    print(f"  Chan BW:    {abs(freqs_mhz[1] - freqs_mhz[0]) * 1e3:.1f} kHz")
    print(f"  Total BW:   {abs(freqs_mhz[0] - freqs_mhz[-1]):.2f} MHz")

    # Freq config attributes
    fc = f["freq_config"]
    print(f"\n  Freq config:")
    print(f"    n_chan:              {fc.attrs['n_chan']}")
    print(f"    total_n_chan:        {fc.attrs['total_n_chan']}")
    print(f"    total_bw_mhz:       {fc.attrs['total_bw_mhz']}")
    print(f"    freq_end_voltage:   {fc.attrs['freq_end_voltage_mhz']} MHz")

    # =====================================================================
    # 4. BEAM POINTINGS (Alt/Az)
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("4. BEAM POINTINGS")
    print("=" * 60)
    alt_deg = f["pointings/alt_deg"][:]
    az_deg = f["pointings/az_deg"][:]
    names = json.loads(f["pointings"].attrs["names"])

    print(f"  N beams:    {len(alt_deg)}")
    print(f"  Alt range:  {alt_deg.min():.1f}° – {alt_deg.max():.1f}°")
    print(f"  Az range:   {az_deg.min():.1f}° – {az_deg.max():.1f}°")
    print(f"\n  First 16 beams:")
    print(f"  {'Name':<12s} {'Alt (°)':>8s} {'Az (°)':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}")
    for i in range(min(16, len(alt_deg))):
        print(f"  {names[i]:<12s} {alt_deg[i]:>8.1f} {az_deg[i]:>8.1f}")
    if len(alt_deg) > 10:
        print(f"  ... ({len(alt_deg) - 16} more)")

    # =====================================================================
    # 5. ARRAY CONFIG — antenna positions and SNAP mapping
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("5. ARRAY CONFIGURATION")
    print("=" * 60)
    ac = f["array_config"]
    positions = ac["positions_enu"][:]        # (64, 3) ENU in meters
    active_mask = ac["active_mask"][:]        # (64,) bool
    snap2ant = ac["snap_to_ant64"][:]         # (64,) SNAP pkt idx → ant64
    ant2snap = ac["ant64_to_snap"][:]         # (64,) ant64 → SNAP pkt idx
    pos_ids = json.loads(ac.attrs["pos_ids"])  # 64 strings
    csv_path = ac.attrs.get("csv_path", "unknown")

    n_active = int(np.sum(active_mask))
    active_indices = np.where(active_mask)[0]

    print(f"  Layout CSV:     {csv_path}")
    print(f"  Active ants:    {n_active} / 64")
    print(f"  Active ant64s:  {list(active_indices)}")

    # Active antenna table
    print(f"\n  {'Ant64':>5s} {'Pos ID':<10s} {'E (m)':>7s} {'N (m)':>7s} "
          f"{'U (m)':>7s} {'Pkt Idx':>8s}")
    print(f"  {'-'*5} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for ai in active_indices:
        snap_idx = ant2snap[ai]
        snap_str = str(snap_idx) if snap_idx >= 0 else "n/a"
        print(f"  {ai:>5d} {pos_ids[ai]:<10s} {positions[ai,0]:>7.2f} "
              f"{positions[ai,1]:>7.2f} {positions[ai,2]:>7.3f} {snap_str:>8s}")

    # SNAP packet index mapping
    print(f"\n  SNAP packet index = board × 12 + ADC channel")
    print(f"\n  {'Pkt Idx':>7s} {'SNAP':>5s} {'ADC':>3s} → {'Ant64':>5s} {'Pos ID':<10s}")
    print(f"  {'-'*7} {'-'*5} {'-'*3}   {'-'*5} {'-'*10}")
    for si in range(64):
        ai = snap2ant[si]
        if ai >= 0 and active_mask[ai]:
            board = si // 12
            adc = si % 12
            print(f"  {si:>7d} {board:>5d} {adc:>3d} → {ai:>5d} {pos_ids[ai]:<10s}")

    # =====================================================================
    # 6. PRACTICAL EXAMPLES
    # =====================================================================
    print(f"\n{'=' * 60}")
    print("6. PRACTICAL EXAMPLES")
    print("=" * 60)

    # a) Which channels have non-zero weights?
    first_active_snap = next(
        si for si in range(64)
        if snap2ant[si] >= 0 and active_mask[snap2ant[si]]
    )
    beam0_ant = (w_int8[0, :, 0, 0, first_active_snap] != 0) | \
                (w_int8[1, :, 0, 0, first_active_snap] != 0)
    n_nonzero = int(np.sum(beam0_ant))
    print(f"\n  a) Non-zero channels (beam 0, pkt idx {first_active_snap}):")
    print(f"     {n_nonzero} / {len(freqs_hz)} channels")
    if n_nonzero < len(freqs_hz):
        print(f"     ({len(freqs_hz) - n_nonzero} zeroed — "
              f"flagged by calibration)")

    # b) Which SNAP packet indices have non-zero weights?
    print(f"\n  b) Active SNAP packet indices (non-zero weights):")
    active_snaps = [
        si for si in range(64)
        if np.any(w_int8[:, :, 0, :, si] != 0)
    ]
    print(f"     {len(active_snaps)} inputs: {active_snaps}")

    # c) Phase at center frequency for each antenna
    mid_ch = len(freqs_hz) // 2
    print(f"\n  c) Phase at center freq ({freqs_mhz[mid_ch]:.1f} MHz), beam 0:")
    print(f"     {'Pkt Idx':>7s} {'Pos ID':<10s} {'|w|':>6s} {'Phase (°)':>10s}")
    print(f"     {'-'*7} {'-'*10} {'-'*6} {'-'*10}")
    for si in active_snaps:
        ai = snap2ant[si]
        val = beam0_complex[mid_ch, si]
        amp = abs(val)
        phase = np.angle(val, deg=True)
        if ai >= 0 and pos_ids[ai]:
            pid = pos_ids[ai]
        else:
            pid = f"pkt_{si}"
        print(f"     {si:>7d} {pid:<10s} {amp:>6.3f} {phase:>+10.1f}")

    # d) Extract weights for a specific beam and frequency slice
    beam_idx = 0
    freq_start_mhz, freq_end_mhz = 430.0, 440.0
    if freqs_mhz[0] > freqs_mhz[-1]:
        ch_mask = (freqs_mhz <= freq_start_mhz + 10) & (freqs_mhz >= freq_start_mhz)
    else:
        ch_mask = (freqs_mhz >= freq_start_mhz) & (freqs_mhz <= freq_end_mhz)
    n_sel = int(np.sum(ch_mask))
    print(f"\n  d) Frequency slice {freq_start_mhz}–{freq_end_mhz} MHz:")
    print(f"     {n_sel} channels selected")
    beam_slice = all_complex[beam_idx, :, ch_mask]  # (64, n_sel)
    print(f"     Beam {beam_idx} weights shape: {beam_slice.shape} = (ants, chans)")

    f.close()
    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
