"""Public ``inspect_snap_weights`` API.

Sanity-check a SNAP int8 weights HDF5 file: print the SNAP input
mapping, which board/ADC channels carry non-zero weights, and which
antennas they correspond to. Promoted from
``examples/inspect_snap_weights.py`` for the compose-friendly notebook
flow.

Usage in a notebook
-------------------

>>> from bf_weights_generator import inspect_snap_weights
>>> info = inspect_snap_weights('weights.h5')          # uses canonical layout
>>> info['n_nonzero_snap_inputs']                       # int

Usage from the CLI
------------------

::

    casm-bf-inspect weights.h5 --layout my_layout.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from bf_weights_generator.snap_weights import Array64Config


def inspect_snap_weights(weights, layout=None, *, print_table=True):
    """Print and return a SNAP-input summary of a weights HDF5 file.

    Parameters
    ----------
    weights : str or Path
        Path to the int8 weights HDF5.
    layout : str or Path, optional
        Antenna layout CSV (output SNAP mapping). Defaults to the
        canonical resolver in
        :func:`casm_io.constants.resolve_layout_path`.
    print_table : bool
        Print human-readable table to stdout. Default True.

    Returns
    -------
    info : dict
        Keys: ``weights_shape``, ``n_chan``, ``n_beams``, ``scale``,
        ``freq_range_hz``, ``compute_layout`` (path stored in HDF5),
        ``output_layout`` (path used for the SNAP mapping),
        ``n_nonzero_snap_inputs`` (int <= 64),
        ``n_active_layout`` (int <= 64),
        ``snap_table`` (list of dicts with snap_idx, snap_board,
        adc_ch, ant64, pos_id, east_m, north_m, has_data).
    """
    weights = Path(weights)
    with h5py.File(weights, "r") as f:
        w = f["weights_int8"][:]
        freqs = f["frequencies_hz"][:]
        alt = f["pointings/alt_deg"][:]
        az = f["pointings/az_deg"][:]
        names = json.loads(f["pointings"].attrs["names"])
        scale = float(f.attrs["scale_factor"])
        n_beams = int(f.attrs["n_beams"])
        n_chan = int(f.attrs["n_channels"])
        stored_csv = str(f["array_config"].attrs.get("csv_path", "unknown"))

    # Resolve output layout via canonical resolver if not given.
    if layout is None:
        from casm_io.constants import resolve_layout_path
        layout_path = resolve_layout_path(None)
    else:
        layout_path = Path(layout)
    output_cfg = Array64Config.from_csv(str(layout_path))

    snap_table = []
    n_nonzero = 0
    for snap_idx in range(64):
        snap_board = snap_idx // 12
        adc_ch = snap_idx % 12
        ant64 = int(output_cfg.snap_to_ant64[snap_idx])
        has_data = bool(np.any(w[:, :, 0, :, snap_idx] != 0))
        if has_data:
            n_nonzero += 1
        row = {
            "snap_idx": snap_idx,
            "snap_board": snap_board,
            "adc_ch": adc_ch,
            "ant64": ant64,
            "has_data": has_data,
        }
        if ant64 >= 0 and bool(output_cfg.active_mask[ant64]):
            row["pos_id"] = str(output_cfg.pos_ids[ant64])
            row["east_m"] = float(output_cfg.positions_enu[ant64, 0])
            row["north_m"] = float(output_cfg.positions_enu[ant64, 1])
        snap_table.append(row)

    info = {
        "weights_shape": tuple(w.shape),
        "n_chan": n_chan,
        "n_beams": n_beams,
        "scale": scale,
        "freq_range_hz": (float(freqs[0]), float(freqs[-1])),
        "compute_layout": stored_csv,
        "output_layout": str(layout_path),
        "n_nonzero_snap_inputs": n_nonzero,
        "n_active_layout": int(output_cfg.n_active),
        "snap_table": snap_table,
        "beam_names": names,
        "beam_alt_deg": alt.tolist(),
        "beam_az_deg": az.tolist(),
    }

    if print_table:
        _print_summary(info)
    return info


def _print_summary(info):
    print(f"Weights shape: {info['weights_shape']}")
    print(f"  axes: (real/imag, channels, polarization, beams, SNAP-input)")
    print(f"Scale factor: {info['scale']}")
    print(f"Freq range: {info['freq_range_hz'][0]/1e6:.2f} → "
          f"{info['freq_range_hz'][1]/1e6:.2f} MHz")
    print(f"Compute layout (stored): {info['compute_layout']}")
    print(f"Output layout (SNAP map): {info['output_layout']}")
    print()
    print(f"Beams: {info['n_beams']}")
    print(f"  alt range: {min(info['beam_alt_deg']):.1f}° – "
          f"{max(info['beam_alt_deg']):.1f}°")
    print(f"  az  range: {min(info['beam_az_deg']):.1f}° – "
          f"{max(info['beam_az_deg']):.1f}°")
    if info['n_beams'] <= 8:
        print("  detail (alt°, az°):")
        for n, alt, az in zip(info['beam_names'],
                              info['beam_alt_deg'],
                              info['beam_az_deg']):
            print(f"    {n!r:<22s}  ({alt:6.2f}, {az:6.2f})")
    print()
    print(f"{'PktIdx':>7s} {'SNAP':>5s} {'ADC':>3s}  {'Ant64':>5s} "
          f"{'Pos ID':<10s} {'E(m)':>6s} {'N(m)':>6s}  {'Non-zero':>8s}")
    print("-" * 70)
    for row in info["snap_table"]:
        if "pos_id" in row:
            flag = "YES" if row["has_data"] else "---"
            print(f"{row['snap_idx']:>7d} {row['snap_board']:>5d} "
                  f"{row['adc_ch']:>3d}  {row['ant64']:>5d} "
                  f"{row['pos_id']:<10s} {row['east_m']:>6.2f} "
                  f"{row['north_m']:>6.2f}  {flag:>8s}")
        elif row["has_data"]:
            print(f"{row['snap_idx']:>7d} {row['snap_board']:>5d} "
                  f"{row['adc_ch']:>3d}  {'n/a':>5s} "
                  f"{'':<10s} {'':>6s} {'':>6s}  {'YES':>8s}  *** unmapped")
    print()
    print(f"SNAP inputs with non-zero weights: "
          f"{info['n_nonzero_snap_inputs']} / 64")
    print(f"Active antennas in layout:         "
          f"{info['n_active_layout']} / 64")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect a SNAP int8 weights HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("weights", help="Int8 weights HDF5 file")
    parser.add_argument(
        "--layout", "-l", default=None,
        help="Antenna layout CSV for SNAP mapping (output layout). "
             "Default: $CASM_LAYOUT_CSV / $CASM_LAYOUT_DIR/current.",
    )
    args = parser.parse_args(argv)
    inspect_snap_weights(args.weights, layout=args.layout)


if __name__ == "__main__":
    main()
