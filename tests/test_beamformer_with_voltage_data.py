#!/usr/bin/env python3
"""
Test beamformer weights with real CASM voltage data.

This script:
1. Loads voltage data from a DADA file (SNAP 2 only)
2. Maps ADC channels to antenna positions
3. Computes tracking coherent weights for Cassiopeia A
4. Computes incoherent weights
5. Applies beamforming and compares results
"""

import sys
import numpy as np
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to disk
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.insert(0, '/home/casm/software/vishnu/OVRO_DATA_EXPERIMENTS_DEC_2025/VOLTAGE_ANALYSIS')
sys.path.insert(0, '/home/casm/software/vishnu/bf_weights_generator')

from casm_io import read_dada_header, print_dada_header, read_dada_data
from bf_weights_generator import (
    GeometricBeamformer,
    PhaseCenter,
    ArrayConfig,
    FrequencyConfig,
    BeamMode
)

# =============================================================================
# SNAP 2 ADC Channel to Antenna Mapping
# =============================================================================
# From casm_adc_to_ant_map_5JAN26.csv
# Format: {adc_channel: (antenna_index, (x, y, z) in ENU meters)}
SNAP2_ADC_TO_ANTENNA = {
    1: (10, (1.65, -10.5, 0.0)),
    2: (12, (-4.553, -5.5, -0.279)),
    3: (4, (1.65, 0.0, 0.0)),
    6: (6, (0.0, -10.5, 0.0)),
    8: (8, (0.825, -10.5, 0.0)),
    9: (11, (2.03, -10.5, 0.0)),
    11: (7, (0.38, -10.5, 0.0)),
}

# ADC channels with antennas (sorted)
ACTIVE_ADC_CHANNELS = sorted(SNAP2_ADC_TO_ANTENNA.keys())
N_ANTENNAS = len(ACTIVE_ADC_CHANNELS)

# =============================================================================
# Cassiopeia A coordinates (J2000)
# =============================================================================
CAS_A_RA_DEG = 350.85    # 23h 23m 24s
CAS_A_DEC_DEG = 58.815   # +58° 48' 54"


def get_snap2_antenna_positions():
    """
    Get antenna positions for SNAP 2 in the order of ADC channels.

    Returns
    -------
    positions : np.ndarray
        Shape (n_ant, 3) ENU positions in meters
    adc_channels : list
        ADC channel indices corresponding to each row
    antenna_indices : list
        Original antenna indices corresponding to each row
    """
    positions = []
    adc_channels = []
    antenna_indices = []

    for adc_ch in ACTIVE_ADC_CHANNELS:
        ant_idx, pos = SNAP2_ADC_TO_ANTENNA[adc_ch]
        positions.append(pos)
        adc_channels.append(adc_ch)
        antenna_indices.append(ant_idx)

    return np.array(positions), adc_channels, antenna_indices


def extract_snap2_voltages(voltages_dict, adc_channels):
    """
    Extract voltages for active ADC channels from SNAP 2.

    Parameters
    ----------
    voltages_dict : dict
        Dictionary from read_dada_data with snap_id -> voltages
    adc_channels : list
        List of ADC channels to extract

    Returns
    -------
    voltages : np.ndarray
        Shape (n_time, n_chan, n_ant) - reordered for beamforming
    """
    snap2_data = voltages_dict[2]  # Shape: (n_time, n_chan, n_inputs=12)

    # Extract only the active ADC channels
    # voltages_dict[2] has shape (n_time, n_chan, 12)
    # We want shape (n_time, n_chan, n_ant)
    voltages = snap2_data[:, :, adc_channels]

    return voltages


def compute_observation_time(header):
    """
    Compute Unix timestamp from DADA header UTC_START.

    Parameters
    ----------
    header : dict
        DADA header dictionary

    Returns
    -------
    unix_time : float
        Unix timestamp (seconds since 1970-01-01)
    """
    # Parse UTC_START format: "2026-01-21-01:16:44"
    utc_start_str = header.get('UTC_START', '2026-01-21-01:16:44')

    # Parse the datetime string
    dt = datetime.strptime(utc_start_str, "%Y-%m-%d-%H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)

    unix_time = dt.timestamp()

    return unix_time


def apply_beamformer_weights(voltages, weights):
    """
    Apply beamformer weights to voltage data.

    For tracking beams:
        weights shape: (n_beams, n_times, n_ant, n_chan)
        voltages shape: (n_time, n_chan, n_ant)

    For stationary beams:
        weights shape: (n_beams, n_ant, n_chan)

    Parameters
    ----------
    voltages : np.ndarray
        Voltage data, shape (n_time, n_chan, n_ant)
    weights : np.ndarray
        Beamformer weights

    Returns
    -------
    beamformed : np.ndarray
        Beamformed output
    """
    n_time, n_chan, n_ant = voltages.shape

    if weights.ndim == 4:
        # Tracking weights: (n_beams, n_times, n_ant, n_chan)
        n_beams, n_weight_times, _, _ = weights.shape

        # For now, use weights at each corresponding time
        # If n_weight_times < n_time, we need to interpolate or repeat
        if n_weight_times == 1:
            # Use same weights for all times
            w = weights[:, 0, :, :]  # (n_beams, n_ant, n_chan)
            # Beamform: sum over antennas of (weight * voltage)
            # voltages: (n_time, n_chan, n_ant) -> transpose to (n_time, n_ant, n_chan)
            v = voltages.transpose(0, 2, 1)  # (n_time, n_ant, n_chan)
            # w: (n_beams, n_ant, n_chan)
            # Result: (n_beams, n_time, n_chan)
            beamformed = np.einsum('bac,tac->btc', w, v)
        else:
            # Multiple time steps in weights - need to match
            raise NotImplementedError("Multi-time tracking weights not yet implemented")

    elif weights.ndim == 3:
        # Stationary/simple weights: (n_beams, n_ant, n_chan)
        v = voltages.transpose(0, 2, 1)  # (n_time, n_ant, n_chan)
        beamformed = np.einsum('bac,tac->btc', weights, v)

    else:
        raise ValueError(f"Unexpected weights shape: {weights.shape}")

    return beamformed


def compute_power_spectrum(beamformed, n_avg=1024):
    """
    Compute power spectrum from beamformed voltages.

    Parameters
    ----------
    beamformed : np.ndarray
        Shape (n_beams, n_time, n_chan)
    n_avg : int
        Number of time samples to average

    Returns
    -------
    power : np.ndarray
        Shape (n_beams, n_time_avg, n_chan)
    """
    n_beams, n_time, n_chan = beamformed.shape

    # Compute power (|V|^2)
    power = np.abs(beamformed) ** 2

    # Average over time blocks
    n_blocks = n_time // n_avg
    if n_blocks > 0:
        power = power[:, :n_blocks * n_avg, :]
        power = power.reshape(n_beams, n_blocks, n_avg, n_chan)
        power = power.mean(axis=2)

    return power


def main():
    """Main test function."""

    # =========================================================================
    # Configuration
    # =========================================================================
    dada_file = "/data/casm/voltage_dumps/2026-01-21-01:16:44_0000000000000000.000000.dada"
    n_time_samples = 8192  # Use fewer samples for quick testing

    print("=" * 70)
    print("CASM Beamformer Test with Voltage Data")
    print("=" * 70)

    # =========================================================================
    # Step 1: Read DADA header
    # =========================================================================
    print("\n[1] Reading DADA header...")
    header = read_dada_header(dada_file)
    print_dada_header(header)

    # Get observation time from UTC_START
    obs_unix_time = compute_observation_time(header)
    obs_datetime = datetime.fromtimestamp(obs_unix_time, tz=timezone.utc)
    print(f"\nObservation time (from UTC_START): {obs_datetime.isoformat()}")
    print(f"Unix timestamp: {obs_unix_time:.3f}")

    # =========================================================================
    # Step 2: Load voltage data (SNAP 2 only)
    # =========================================================================
    print("\n[2] Loading voltage data (SNAP 2 only)...")
    voltages_dict, _ = read_dada_data(dada_file, n_time=n_time_samples, snaps=[2])

    # Get antenna positions and extract relevant voltages
    positions_enu, adc_channels, antenna_indices = get_snap2_antenna_positions()
    voltages = extract_snap2_voltages(voltages_dict, adc_channels)

    print(f"\nExtracted voltages shape: {voltages.shape}")
    print(f"  (n_time={voltages.shape[0]}, n_chan={voltages.shape[1]}, n_ant={voltages.shape[2]})")
    print(f"\nAntenna mapping:")
    for i, (adc_ch, ant_idx) in enumerate(zip(adc_channels, antenna_indices)):
        pos = positions_enu[i]
        print(f"  ADC ch {adc_ch:2d} -> Antenna {ant_idx:2d} @ ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) m")

    # =========================================================================
    # Step 3: Setup beamformer
    # =========================================================================
    print("\n[3] Setting up beamformer...")

    # Create array config with SNAP 2 antenna positions
    array_config = ArrayConfig(
        positions_enu=positions_enu,
        antenna_flags=np.ones(N_ANTENNAS, dtype=bool)
    )
    print(f"Array config: {array_config}")

    # Use fixed frequency config: 468.75 MHz to 375 MHz with 3072 channels
    n_chan = 3072
    freq_config = FrequencyConfig(
        n_chan=n_chan,
        total_bw_mhz=125.0,
        total_n_chan=4096,
        freq_end_voltage_mhz=468.75
    )

    bf_freqs_mhz = freq_config.get_frequencies_mhz()
    print(f"Frequency config: {n_chan} channels")
    print(f"  Range: {bf_freqs_mhz[0]:.3f} MHz (high) to {bf_freqs_mhz[-1]:.3f} MHz (low)")
    print(f"  Channel BW: {freq_config.chan_bw_mhz * 1000:.3f} kHz")

    # Create beamformer
    bf = GeometricBeamformer(
        array_config=array_config,
        freq_config=freq_config,
        use_astropy=True
    )

    # =========================================================================
    # Step 4: Compute tracking coherent weights for Cassiopeia A
    # =========================================================================
    print("\n[4] Computing tracking coherent weights for Cassiopeia A...")

    cas_a = PhaseCenter(ra_deg=CAS_A_RA_DEG, dec_deg=CAS_A_DEC_DEG, name="CasA")
    print(f"  Phase center: RA={cas_a.ra_deg:.3f}°, Dec={cas_a.dec_deg:.3f}°")

    # Compute weights for a single time (observation start)
    # For a short observation, we can use a single set of weights
    tracking_weights = bf.compute_tracking_weights(
        phase_centers=[cas_a],
        unix_times=np.array([obs_unix_time]),
        mode='coherent'
    )

    print(f"  Tracking weights shape: {tracking_weights.weights.shape}")
    print(f"  (n_beams={tracking_weights.n_beams}, n_times={tracking_weights.n_times}, "
          f"n_ant={tracking_weights.n_antennas}, n_chan={tracking_weights.n_channels})")

    # =========================================================================
    # Step 5: Compute incoherent weights (1 beam)
    # =========================================================================
    print("\n[5] Computing incoherent weights (1 beam)...")

    incoherent_weights = bf.compute_tracking_weights(
        phase_centers=None,
        unix_times=np.array([obs_unix_time]),
        mode='incoherent',
        n_beams=1
    )

    print(f"  Incoherent weights shape: {incoherent_weights.weights.shape}")

    # Verify incoherent weights are all ones
    assert np.allclose(incoherent_weights.weights, 1.0), "Incoherent weights should be all ones!"
    print("  ✓ Verified: all weights are 1+0j")

    # =========================================================================
    # Step 6: Apply beamforming
    # =========================================================================
    print("\n[6] Applying beamformer weights...")

    # Coherent beam on Cas A
    print("  Computing coherent beam on Cas A...")
    coherent_beam = apply_beamformer_weights(voltages, tracking_weights.weights)
    print(f"    Coherent beam shape: {coherent_beam.shape}")

    # Incoherent beams (all identical since weights are ones)
    print("  Computing incoherent beams...")
    incoherent_beams = apply_beamformer_weights(voltages, incoherent_weights.weights)
    print(f"    Incoherent beams shape: {incoherent_beams.shape}")

    # =========================================================================
    # Step 7: Compute and compare power spectra
    # =========================================================================
    print("\n[7] Computing power spectra...")

    n_avg = 1024  # Average over 1024 time samples

    coherent_power = compute_power_spectrum(coherent_beam, n_avg=n_avg)
    incoherent_power = compute_power_spectrum(incoherent_beams, n_avg=n_avg)

    print(f"  Coherent power shape: {coherent_power.shape}")
    print(f"  Incoherent power shape: {incoherent_power.shape}")

    # Average over time to get spectrum
    coherent_spectrum = coherent_power.mean(axis=1)  # (n_beams, n_chan)
    incoherent_spectrum = incoherent_power.mean(axis=1)  # (n_beams, n_chan)

    print(f"\n  Coherent beam (Cas A) mean power: {coherent_spectrum[0].mean():.6f}")
    print(f"  Incoherent beam 0 mean power: {incoherent_spectrum[0].mean():.6f}")

    # =========================================================================
    # Step 8: Generate diagnostic plots
    # =========================================================================
    print("\n[8] Generating diagnostic plots...")

    output_dir = '/home/casm/software/vishnu/bf_weights_generator/tests'

    # Use the beamformer frequencies (high to low)
    plot_freqs = bf_freqs_mhz

    # --- Figure 1: Spectrum comparison ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Coherent beam spectrum
    ax = axes1[0, 0]
    ax.plot(plot_freqs, 10*np.log10(coherent_spectrum[0] + 1e-10), 'b-', lw=0.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'Coherent Beam on Cassiopeia A\nRA={CAS_A_RA_DEG:.2f}°, Dec={CAS_A_DEC_DEG:.2f}°')
    ax.grid(True, alpha=0.3)

    # Plot 2: Incoherent beam spectrum
    ax = axes1[0, 1]
    ax.plot(plot_freqs, 10*np.log10(incoherent_spectrum[0] + 1e-10), 'r-', lw=0.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Incoherent Beam (Total Power)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Comparison overlay
    ax = axes1[1, 0]
    ax.plot(plot_freqs, 10*np.log10(coherent_spectrum[0] + 1e-10), 'b-', lw=0.5, label='Coherent (Cas A)')
    ax.plot(plot_freqs, 10*np.log10(incoherent_spectrum[0] + 1e-10), 'r-', lw=0.5, alpha=0.7, label='Incoherent')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Coherent vs Incoherent Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Ratio (coherent / incoherent)
    ax = axes1[1, 1]
    ratio = coherent_spectrum[0] / (incoherent_spectrum[0] + 1e-10)
    ax.plot(plot_freqs, 10*np.log10(ratio + 1e-10), 'g-', lw=0.5)
    ax.axhline(y=0, color='k', linestyle='--', lw=0.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Ratio (dB)')
    ax.set_title('Coherent / Incoherent Ratio')
    ax.grid(True, alpha=0.3)

    fig1.suptitle(f'CASM Beamformer Test - {obs_datetime.strftime("%Y-%m-%d %H:%M:%S")} UTC', fontsize=12)
    plt.tight_layout()
    fig1_path = f'{output_dir}/beamformer_test_spectra.png'
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig1_path}")
    plt.close(fig1)

    # --- Figure 2: Time series ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Compute time axis
    tsamp_us = float(header.get('TSAMP', 32.768))
    n_time_plot = coherent_beam.shape[1]
    time_ms = np.arange(n_time_plot) * tsamp_us / 1000.0

    # Coherent beam time series (band-averaged)
    coherent_timeseries = np.abs(coherent_beam[0]) ** 2
    coherent_ts_avg = coherent_timeseries.mean(axis=1)

    # Incoherent beam time series (band-averaged)
    incoherent_timeseries = np.abs(incoherent_beams[0]) ** 2
    incoherent_ts_avg = incoherent_timeseries.mean(axis=1)

    # Plot 1: Coherent time series
    ax = axes2[0, 0]
    ax.plot(time_ms, 10*np.log10(coherent_ts_avg + 1e-10), 'b-', lw=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Coherent Beam Time Series (Band-Averaged)')
    ax.grid(True, alpha=0.3)

    # Plot 2: Incoherent time series
    ax = axes2[0, 1]
    ax.plot(time_ms, 10*np.log10(incoherent_ts_avg + 1e-10), 'r-', lw=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Incoherent Beam Time Series (Band-Averaged)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Dynamic spectrum (coherent) - waterfall
    ax = axes2[1, 0]
    # Downsample for visualization
    ds_time = 64
    ds_freq = 8
    coherent_ds = coherent_timeseries[::ds_time, ::ds_freq]
    extent = [plot_freqs[0], plot_freqs[-1], time_ms[-1], time_ms[0]]
    im = ax.imshow(10*np.log10(coherent_ds + 1e-10), aspect='auto', extent=extent,
                   cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Coherent Beam Dynamic Spectrum')
    plt.colorbar(im, ax=ax, label='Power (dB)')

    # Plot 4: Dynamic spectrum (incoherent) - waterfall
    ax = axes2[1, 1]
    incoherent_ds = incoherent_timeseries[::ds_time, ::ds_freq]
    im = ax.imshow(10*np.log10(incoherent_ds + 1e-10), aspect='auto', extent=extent,
                   cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Incoherent Beam Dynamic Spectrum')
    plt.colorbar(im, ax=ax, label='Power (dB)')

    fig2.suptitle(f'CASM Beamformer Test - Time Domain', fontsize=12)
    plt.tight_layout()
    fig2_path = f'{output_dir}/beamformer_test_timeseries.png'
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig2_path}")
    plt.close(fig2)

    # --- Figure 3: Weights inspection ---
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))

    # Get weights for inspection
    coh_weights = tracking_weights.weights[0, 0, :, :]  # (n_ant, n_chan)

    # Plot weight phases for each antenna
    ax = axes3[0, 0]
    for ant_i in range(coh_weights.shape[0]):
        ax.plot(plot_freqs, np.angle(coh_weights[ant_i, :]), lw=0.5,
                label=f'Ant {antenna_indices[ant_i]}')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Coherent Weight Phases')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot weight amplitudes (should be ~1 for coherent)
    ax = axes3[0, 1]
    for ant_i in range(coh_weights.shape[0]):
        ax.plot(plot_freqs, np.abs(coh_weights[ant_i, :]), lw=0.5,
                label=f'Ant {antenna_indices[ant_i]}')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Coherent Weight Amplitudes')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot geometric delays
    ax = axes3[0, 2]
    # Compute delays from phase slope
    for ant_i in range(coh_weights.shape[0]):
        phases = np.unwrap(np.angle(coh_weights[ant_i, :]))
        ax.plot(plot_freqs, phases, lw=0.5, label=f'Ant {antenna_indices[ant_i]}')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Unwrapped Phase (rad)')
    ax.set_title('Unwrapped Phases (shows delay)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot per-antenna power (coherent beam)
    ax = axes3[1, 0]
    per_ant_power = np.abs(voltages) ** 2
    per_ant_power_avg = per_ant_power.mean(axis=(0, 1))  # Average over time and freq
    ax.bar(range(len(antenna_indices)), per_ant_power_avg)
    ax.set_xticks(range(len(antenna_indices)))
    ax.set_xticklabels([f'{idx}' for idx in antenna_indices])
    ax.set_xlabel('Antenna Index')
    ax.set_ylabel('Mean Power')
    ax.set_title('Per-Antenna Mean Power')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot antenna positions
    ax = axes3[1, 1]
    ax.scatter(positions_enu[:, 0], positions_enu[:, 1], s=100, c='blue', marker='o')
    for i, ant_idx in enumerate(antenna_indices):
        ax.annotate(f'{ant_idx}', (positions_enu[i, 0], positions_enu[i, 1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Antenna Positions (ENU)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot cross-correlation matrix (snapshot)
    ax = axes3[1, 2]
    # Pick middle time sample, average over frequency bands
    mid_t = voltages.shape[0] // 2
    freq_avg_range = slice(n_chan//4, 3*n_chan//4)  # Middle 50% of band
    v_snap = voltages[mid_t, freq_avg_range, :].mean(axis=0)  # (n_ant,)
    xcorr = np.outer(v_snap, np.conj(v_snap))
    xcorr_phase = np.angle(xcorr)
    im = ax.imshow(xcorr_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_xticks(range(len(antenna_indices)))
    ax.set_yticks(range(len(antenna_indices)))
    ax.set_xticklabels([f'{idx}' for idx in antenna_indices])
    ax.set_yticklabels([f'{idx}' for idx in antenna_indices])
    ax.set_xlabel('Antenna')
    ax.set_ylabel('Antenna')
    ax.set_title('Cross-correlation Phase (snapshot)')
    plt.colorbar(im, ax=ax, label='Phase (rad)')

    fig3.suptitle('Beamformer Weights & Array Diagnostics', fontsize=12)
    plt.tight_layout()
    fig3_path = f'{output_dir}/beamformer_test_weights.png'
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig3_path}")
    plt.close(fig3)

    # =========================================================================
    # Step 9: Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Observation: {obs_datetime.isoformat()}")
    print(f"Source: Cassiopeia A (RA={CAS_A_RA_DEG}°, Dec={CAS_A_DEC_DEG}°)")
    print(f"Antennas used: {N_ANTENNAS} (from SNAP 2)")
    print(f"Frequency channels: {n_chan}")
    print(f"Time samples: {n_time_samples}")
    print(f"\nCoherent beam mean power: {coherent_spectrum[0].mean():.6f}")
    print(f"Incoherent beam mean power: {incoherent_spectrum[0].mean():.6f}")
    print(f"Ratio (coherent/incoherent): {coherent_spectrum[0].mean() / incoherent_spectrum[0].mean():.4f}")
    print("=" * 70)

    return {
        'coherent_spectrum': coherent_spectrum,
        'incoherent_spectrum': incoherent_spectrum,
        'coherent_beam': coherent_beam,
        'incoherent_beams': incoherent_beams,
        'freqs_mhz': bf_freqs_mhz,
        'header': header,
    }


if __name__ == "__main__":
    results = main()
