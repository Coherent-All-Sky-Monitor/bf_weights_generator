#!/usr/bin/env python3
"""Quick snippet to read CASM int8 weights HDF5 file."""

import h5py
import json
import numpy as np

f = h5py.File("/home/casm/software/vishnu/beamforming_weights/int8_weights_512beam_feb19_svd.h5", "r")

# --- Weights: int8, shape (2, 3072, 2, 512, 64) ---
# axes: (real/imag, channel, pol, beam, packet_index)
w = f["weights_int8"][:]
print(f"Weights shape: {w.shape}")

# Convert beam 0, pol A to complex64
beam0 = (w[0, :, 0, 0, :] + 1j * w[1, :, 0, 0, :]) / 127.0  # (3072, 64)

# --- Frequencies: float64, shape (3072,) ---
freqs_mhz = f["frequencies_hz"][:] / 1e6
print(f"Freq range: {freqs_mhz[0]:.2f} → {freqs_mhz[-1]:.2f} MHz")

# --- Beam pointings: alt/az in degrees ---
alt = f["pointings/alt_deg"][:]
az = f["pointings/az_deg"][:]
names = json.loads(f["pointings"].attrs["names"])
print(f"Beams: {len(alt)}, alt range: {alt.min():.1f}° – {alt.max():.1f}°")

# --- Array config ---
positions = f["array_config/positions_enu"][:]     # (64, 3) ENU meters
active = f["array_config/active_mask"][:]          # (64,) bool
snap2ant = f["array_config/snap_to_ant64"][:]      # packet_index → ant64 slot
ant2snap = f["array_config/ant64_to_snap"][:]      # ant64 slot → packet_index
pos_ids = json.loads(f["array_config"].attrs["pos_ids"])

print(f"Active antennas: {active.sum()} / 64")

# --- Print active SNAP inputs ---
for si in range(64):
    if np.any(w[:, :, 0, :, si] != 0):
        board, adc = si // 12, si % 12
        ai = snap2ant[si]
        pid = pos_ids[ai] if (ai >= 0 and pos_ids[ai]) else f"pkt_{si}"
        print(f"  pkt {si:2d}  SNAP {board}  ADC {adc:2d}  → ant64 {ai:2d}  {pid}")

f.close()
