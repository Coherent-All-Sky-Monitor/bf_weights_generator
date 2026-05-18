import h5py
import json
import numpy as np

filepath = "/home/casm/software/vishnu/beamforming_weights/20261003/int8_weights_512beam_mar10_svd.h5"

with h5py.File(filepath, "r") as f:
    # Load int8 weights: shape (2, n_chan, 2, n_beams, 64)
    # axis 0: 0=real, 1=imaginary
    w_int8 = f["weights_int8"][:]
    scale = f.attrs["scale_factor"]  # 127.0

    # Reconstruct complex weights for pol A, all beams
    # shape: (n_chan, n_beams, 64)
    w_complex = (w_int8[0, :, 0, :, :] + 1j * w_int8[1, :, 0, :, :]) / scale

    # Phase in degrees: (n_chan, n_beams, 64)
    phase_deg = np.angle(w_complex, deg=True)

    # Array layout info
    pos_ids = json.loads(f["array_config"].attrs["pos_ids"])
    active_mask = f["array_config/active_mask"][:]
    snap_to_ant64 = f["array_config/snap_to_ant64"][:]
    freqs_hz = f["frequencies_hz"][:]

    # --- Per-antenna: non-zero phase check ---
    has_nonzero_phase = np.any(phase_deg != 0.0, axis=(0, 1))  # per SNAP slot

    print("=== Antennas with non-zero phase weights ===")
    print(f"{'SNAP idx':<10} {'Ant64 idx':<10} {'Pos ID':<12} {'Row':<8} {'Col':<8} {'Non-zero phase'}")
    print("-" * 65)
    for snap_idx in range(64):
        ant64_idx = snap_to_ant64[snap_idx]
        if ant64_idx < 0:
            continue
        pos_id = pos_ids[ant64_idx]
        if not pos_id:
            continue
        row, col = pos_id.split("_")
        nonzero = has_nonzero_phase[snap_idx]
        print(f"{snap_idx:<10} {ant64_idx:<10} {pos_id:<12} {row:<8} {col:<8} {nonzero}")

    # --- Per-channel: check which are all-zero vs have data ---
    # A channel is "zero" if all weights (all beams, all antennas) are zero
    chan_power = np.sum(np.abs(w_complex) ** 2, axis=(1, 2))  # shape: (n_chan,)
    chan_has_data = chan_power > 0.0

    n_total = len(freqs_hz)
    n_active = np.sum(chan_has_data)
    n_zero = n_total - n_active

    print(f"\n=== Frequency channel summary ===")
    print(f"Total channels:    {n_total}")
    print(f"Active (non-zero): {n_active}")
    print(f"Zero (empty):      {n_zero}")

    if n_active > 0:
        active_freqs = freqs_hz[chan_has_data]
        print(f"Active freq range: {active_freqs.min()/1e6:.3f} - {active_freqs.max()/1e6:.3f} MHz")

    # Show contiguous active/zero regions
    print(f"\n=== Channel regions ===")
    changes = np.diff(chan_has_data.astype(int))
    boundaries = np.where(changes != 0)[0] + 1
    regions = np.split(np.arange(n_total), boundaries)

    for region in regions:
        if len(region) == 0:
            continue
        start, end = region[0], region[-1]
        status = "DATA" if chan_has_data[start] else "ZERO"
        freq_start = freqs_hz[start] / 1e6
        freq_end = freqs_hz[end] / 1e6
        lo, hi = min(freq_start, freq_end), max(freq_start, freq_end)
        print(f"  Chans {start:>4}-{end:>4} ({len(region):>4} chans) | {lo:.3f} - {hi:.3f} MHz | {status}")
