#!/usr/bin/env python3
"""
Inspect SNAP int8 weights HDF5 file.

Prints the SNAP input mapping, showing which board/ADC channels have
non-zero weights and their corresponding antenna positions.

Usage:
    python examples/inspect_snap_weights.py weights_512beam/int8_weights_512beam_feb14_svd.h5
    python examples/inspect_snap_weights.py weights_512beam/int8_weights_512beam_feb14_svd.h5 --layout casm_antenna_layout_current.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bf_weights_generator import Array64Config


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SNAP int8 weights HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "weights", help="Int8 weights HDF5 file",
    )
    parser.add_argument(
        "--layout", "-l", default="casm_antenna_layout_current.csv",
        help="Antenna layout CSV for SNAP mapping (output layout)",
    )
    args = parser.parse_args()

    import h5py

    # Load weights
    f = h5py.File(args.weights, "r")
    w = f["weights_int8"][:]  # (2, n_chan, 2, n_beams, 64)
    freqs = f["frequencies_hz"][:]
    alt = f["pointings/alt_deg"][:]
    az = f["pointings/az_deg"][:]
    names = json.loads(f["pointings"].attrs["names"])
    scale = f.attrs["scale_factor"]
    n_beams = f.attrs["n_beams"]
    n_chan = f.attrs["n_channels"]

    # Stored array config (compute layout)
    stored_csv = f["array_config"].attrs.get("csv_path", "unknown")
    f.close()

    # Load output layout for SNAP mapping
    csv_path = Path(args.layout)
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / args.layout
    current = Array64Config.from_csv(str(csv_path))

    # ── File summary ──────────────────────────────────────────────────────
    print(f"File: {args.weights}")
    print(f"Weights shape: {w.shape}")
    print(f"  axis 0: real/imag (2)")
    print(f"  axis 1: channels ({n_chan})")
    print(f"  axis 2: polarization (2, identical)")
    print(f"  axis 3: beams ({n_beams})")
    print(f"  axis 4: SNAP input index (64)")
    print(f"Scale factor: {scale}")
    print(f"Freq range: {freqs[0]/1e6:.2f} → {freqs[-1]/1e6:.2f} MHz")
    print(f"Compute layout (stored): {stored_csv}")
    print(f"Output layout (SNAP map): {csv_path.name}")
    print()

    # ── Beam pointings ────────────────────────────────────────────────────
    print(f"Beams: {n_beams}")
    print(f"  Alt range: {alt.min():.1f}° – {alt.max():.1f}°")
    print(f"  First 5: ", end="")
    for i in range(min(5, n_beams)):
        print(f"{names[i]}(alt={alt[i]:.1f}°,az={az[i]:.1f}°) ", end="")
    print()
    print()

    # ── SNAP input mapping ────────────────────────────────────────────────
    print(f"Packet Index = SNAP board × 12 + ADC channel")
    print()
    print(f"{'Pkt Idx':>7s} {'SNAP':>5s} {'ADC':>3s}  {'Ant64':>5s} "
          f"{'Pos ID':<10s} {'E(m)':>6s} {'N(m)':>6s}  {'Non-zero':>8s}")
    print("-" * 70)

    n_nonzero = 0
    for snap_idx in range(64):
        snap_board = snap_idx // 12
        adc_ch = snap_idx % 12
        ant64 = current.snap_to_ant64[snap_idx]

        has_data = np.any(w[:, :, 0, :, snap_idx] != 0)
        if has_data:
            n_nonzero += 1

        if ant64 >= 0 and current.active_mask[ant64]:
            pid = current.pos_ids[ant64]
            e = current.positions_enu[ant64, 0]
            n = current.positions_enu[ant64, 1]
            flag = "YES" if has_data else "---"
            print(f"{snap_idx:>7d} {snap_board:>5d} {adc_ch:>3d}  {ant64:>5d} "
                  f"{pid:<10s} {e:>6.2f} {n:>6.2f}  {flag:>8s}")
        elif has_data:
            print(f"{snap_idx:>7d} {snap_board:>5d} {adc_ch:>3d}  {'n/a':>5s} "
                  f"{'':.<10s} {'':>6s} {'':>6s}  {'YES':>8s}  *** unmapped")

    print()
    print(f"SNAP inputs with non-zero weights: {n_nonzero} / 64")
    print(f"Active antennas in layout:         {current.n_active} / 64")

    # ── Channel stats ─────────────────────────────────────────────────────
    # Check how many channels have non-zero weights (beam 0, first active input)
    first_active = next(i for i in range(64) if np.any(w[:, :, 0, 0, i] != 0))
    beam0_real = w[0, :, 0, 0, first_active]
    beam0_imag = w[1, :, 0, 0, first_active]
    nonzero_chan = np.sum((beam0_real != 0) | (beam0_imag != 0))
    print(f"\nNon-zero channels (beam 0, SNAP input {first_active}): "
          f"{nonzero_chan} / {n_chan}")


if __name__ == "__main__":
    main()
