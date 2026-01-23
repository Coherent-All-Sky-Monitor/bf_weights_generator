#!/usr/bin/env python3
"""
Cable Delay Calibration via Cross-Correlation

This script:
1. Loads voltage data from a DADA file
2. Applies geometric delays (fringe stopping toward Cas A)
3. Computes cross-correlations (visibilities) for all baselines
4. Flags RFI using robust sigma clipping (MAD-based)
5. Fits phase vs frequency to extract residual cable delays
6. Solves for per-antenna cable delays
7. Re-tests beamforming with calibrated weights
8. Generates diagnostic plots and saves visibilities

Usage:
    python calibrate_cable_delays.py
"""

import sys
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import warnings

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add paths
sys.path.insert(0, '/home/casm/software/vishnu/OVRO_DATA_EXPERIMENTS_DEC_2025/VOLTAGE_ANALYSIS')
sys.path.insert(0, '/home/casm/software/vishnu/bf_weights_generator')

from casm_io import read_dada_header, read_dada_data
from bf_weights_generator import (
    GeometricBeamformer,
    PhaseCenter,
    ArrayConfig,
    FrequencyConfig,
)

# =============================================================================
# Configuration
# =============================================================================

DADA_FILE = "/data/casm/voltage_dumps/2026-01-21-01:16:44_0000000000000000.000000.dada"
OUTPUT_DIR = "/home/casm/software/vishnu/bf_weights_generator/tests"

# Cas A coordinates (J2000)
CAS_A = PhaseCenter(ra_deg=350.850, dec_deg=58.815, name='CasA')

# SNAP 2 ADC to antenna mapping
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
N_ANTENNAS = len(ACTIVE_ADC_CHANNELS)

# RFI flagging parameters
RFI_SIGMA_THRESHOLD = 5.0  # Flag channels > 5 sigma from median
RFI_WINDOW_SIZE = 32       # Window for local median estimation
MIN_UNFLAGGED_FRAC = 0.5   # Require at least 50% unflagged channels

# Delay fitting parameters
PHASE_UNWRAP_THRESHOLD = np.pi  # Jump threshold for unwrapping


# =============================================================================
# Helper Functions
# =============================================================================

def parse_utc_start(header: Dict) -> float:
    """Parse UTC_START to Unix timestamp."""
    utc_str = header.get('UTC_START', '2026-01-21-01:16:44')
    dt = datetime.strptime(utc_str, "%Y-%m-%d-%H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def get_antenna_mapping():
    """Get antenna positions and mapping."""
    positions = []
    adc_channels = []
    antenna_indices = []

    for adc_ch in ACTIVE_ADC_CHANNELS:
        ant_idx, pos = SNAP2_ADC_TO_ANTENNA[adc_ch]
        positions.append(pos)
        adc_channels.append(adc_ch)
        antenna_indices.append(ant_idx)

    return np.array(positions), adc_channels, antenna_indices


def mad(data: np.ndarray, axis: int = None) -> np.ndarray:
    """Median Absolute Deviation - robust measure of spread."""
    med = np.nanmedian(data, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(data - med), axis=axis)


def robust_zscore(data: np.ndarray, axis: int = None) -> np.ndarray:
    """Compute robust z-score using MAD."""
    med = np.nanmedian(data, axis=axis, keepdims=True)
    mad_val = mad(data, axis=axis)
    # Scale factor to make MAD comparable to std for normal distribution
    mad_scaled = mad_val * 1.4826
    # Avoid division by zero
    mad_scaled = np.maximum(mad_scaled, 1e-10)
    return (data - med) / mad_scaled


# =============================================================================
# RFI Flagging
# =============================================================================

def numpy_median_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """Simple median filter using numpy (no scipy dependency)."""
    n = len(data)
    result = np.zeros_like(data)
    half_win = window_size // 2

    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        result[i] = np.median(data[start:end])

    return result


def flag_rfi_channels(
    spectrum: np.ndarray,
    sigma_threshold: float = RFI_SIGMA_THRESHOLD,
    window_size: int = RFI_WINDOW_SIZE,
) -> Tuple[np.ndarray, Dict]:
    """
    Flag RFI channels using robust sigma clipping.

    Uses a sliding window median to handle bandpass shape,
    then flags outliers using MAD-based z-scores.

    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum, shape (n_chan,) or (n_baseline, n_chan)
    sigma_threshold : float
        Flag channels with |z-score| > threshold
    window_size : int
        Window size for local median estimation

    Returns
    -------
    flags : np.ndarray (bool)
        True = flagged (bad), False = good
    stats : dict
        Flagging statistics
    """
    if spectrum.ndim == 1:
        spectrum = spectrum[np.newaxis, :]

    n_baseline, n_chan = spectrum.shape

    # Compute local median using sliding window (numpy implementation)
    avg_spectrum = np.mean(spectrum, axis=0)
    local_median = numpy_median_filter(avg_spectrum, window_size)
    local_median = local_median[np.newaxis, :]  # Add baseline dimension for broadcasting

    # Compute residual (spectrum - local_median)
    residual = spectrum - local_median

    # Compute MAD-based z-score for each channel
    # Average across baselines first for better statistics
    avg_residual = np.mean(residual, axis=0)
    z_scores = robust_zscore(avg_residual)

    # Flag high outliers (RFI) and low outliers (dropouts)
    flags = np.abs(z_scores) > sigma_threshold

    # Also flag channels with very low power (potential dropouts)
    avg_spectrum = np.mean(spectrum, axis=0)
    low_power_threshold = np.nanmedian(avg_spectrum) * 0.1
    flags |= avg_spectrum < low_power_threshold

    # Flag edge channels (often bad)
    n_edge = 10
    flags[:n_edge] = True
    flags[-n_edge:] = True

    stats = {
        'n_flagged': np.sum(flags),
        'n_total': n_chan,
        'frac_flagged': np.sum(flags) / n_chan,
        'sigma_threshold': sigma_threshold,
    }

    return flags.flatten(), stats


def identify_rfi_bands(flags: np.ndarray, freqs_mhz: np.ndarray) -> List[Tuple[float, float]]:
    """Identify contiguous RFI bands from flags."""
    rfi_bands = []
    in_rfi = False
    start_idx = 0

    for i, flag in enumerate(flags):
        if flag and not in_rfi:
            in_rfi = True
            start_idx = i
        elif not flag and in_rfi:
            in_rfi = False
            if i - start_idx > 1:  # Only report bands > 1 channel
                rfi_bands.append((freqs_mhz[start_idx], freqs_mhz[i-1]))

    if in_rfi:
        rfi_bands.append((freqs_mhz[start_idx], freqs_mhz[-1]))

    return rfi_bands


# =============================================================================
# Cross-Correlation and Visibility Computation
# =============================================================================

def compute_visibilities(
    voltages: np.ndarray,
    n_time_avg: int = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute cross-correlation visibilities for all baselines.

    Parameters
    ----------
    voltages : np.ndarray
        Shape (n_time, n_chan, n_ant) - complex voltages
    n_time_avg : int, optional
        Number of time samples to average (default: all)

    Returns
    -------
    visibilities : np.ndarray
        Shape (n_baseline, n_chan) - complex visibilities
    baselines : list of tuples
        List of (ant_i, ant_j) pairs for each baseline
    """
    n_time, n_chan, n_ant = voltages.shape

    if n_time_avg is None:
        n_time_avg = n_time

    # Generate all baseline pairs
    baselines = list(combinations(range(n_ant), 2))
    n_baseline = len(baselines)

    # Compute visibilities
    visibilities = np.zeros((n_baseline, n_chan), dtype=np.complex128)

    for b_idx, (i, j) in enumerate(baselines):
        # V_ij = V_i × V_j*
        vis = voltages[:n_time_avg, :, i] * np.conj(voltages[:n_time_avg, :, j])
        # Average over time
        visibilities[b_idx, :] = np.mean(vis, axis=0)

    return visibilities, baselines


def compute_autocorrelations(voltages: np.ndarray) -> np.ndarray:
    """Compute auto-correlations (power) for each antenna."""
    n_time, n_chan, n_ant = voltages.shape

    autos = np.zeros((n_ant, n_chan), dtype=np.float64)
    for i in range(n_ant):
        autos[i, :] = np.mean(np.abs(voltages[:, :, i])**2, axis=0)

    return autos


# =============================================================================
# Phase Fitting and Delay Extraction
# =============================================================================

def fit_delay_fft(
    visibility: np.ndarray,
    chan_bw_hz: float,
    flags: np.ndarray = None,
    oversample: int = 16,
) -> Tuple[float, float, float]:
    """
    Fit delay using FFT of cross-spectrum (more robust than phase unwrapping).

    The delay is found by inverse FFT of the visibility, which gives the
    cross-correlation in the lag domain. The peak location gives the delay.

    Parameters
    ----------
    visibility : np.ndarray
        Complex visibility spectrum, shape (n_chan,)
    chan_bw_hz : float
        Channel bandwidth in Hz
    flags : np.ndarray, optional
        Boolean flags (True = bad)
    oversample : int
        Oversampling factor for finer delay resolution

    Returns
    -------
    delay_sec : float
        Fitted delay in seconds
    delay_err : float
        Uncertainty estimate (based on peak width)
    snr : float
        Signal-to-noise ratio of the peak
    """
    n_chan = len(visibility)

    if flags is None:
        flags = np.zeros(n_chan, dtype=bool)

    # Zero out flagged channels
    vis_clean = visibility.copy()
    vis_clean[flags] = 0

    # Oversample FFT for finer delay resolution
    n_fft = n_chan * oversample

    # Inverse FFT to get cross-correlation vs lag
    xcorr = np.fft.ifft(vis_clean, n=n_fft)
    xcorr = np.fft.fftshift(xcorr)

    # Compute lag axis
    # Lag resolution = 1 / (n_chan * chan_bw_hz)
    total_bw_hz = n_chan * chan_bw_hz
    lag_resolution = 1.0 / total_bw_hz / oversample
    lags = (np.arange(n_fft) - n_fft // 2) * lag_resolution

    # Find peak
    xcorr_amp = np.abs(xcorr)
    peak_idx = np.argmax(xcorr_amp)
    peak_lag = lags[peak_idx]

    # Refine peak with quadratic interpolation
    if 1 < peak_idx < n_fft - 1:
        y0, y1, y2 = xcorr_amp[peak_idx - 1], xcorr_amp[peak_idx], xcorr_amp[peak_idx + 1]
        if y0 < y1 > y2:  # Valid peak
            delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
            peak_lag = lags[peak_idx] + delta * lag_resolution

    # Estimate SNR
    peak_val = xcorr_amp[peak_idx]
    # Exclude region around peak for noise estimate
    mask = np.abs(np.arange(n_fft) - peak_idx) > n_fft // 10
    noise_std = np.std(xcorr_amp[mask])
    snr = peak_val / noise_std if noise_std > 0 else np.inf

    # Estimate delay error from peak width
    # Rough estimate: error ~ 1 / (SNR * bandwidth)
    delay_err = 1.0 / (snr * total_bw_hz) if snr > 0 else np.nan

    return peak_lag, delay_err, snr


def fit_phase_vs_frequency(
    phases: np.ndarray,
    freqs_hz: np.ndarray,
    weights: np.ndarray = None,
    flags: np.ndarray = None,
) -> Tuple[float, float, float, float]:
    """
    Fit phase vs frequency to extract delay.

    phase(f) = 2π × f × delay + offset

    Parameters
    ----------
    phases : np.ndarray
        Unwrapped phases in radians, shape (n_chan,)
    freqs_hz : np.ndarray
        Frequencies in Hz, shape (n_chan,)
    weights : np.ndarray, optional
        Weights for fitting (default: uniform)
    flags : np.ndarray, optional
        Boolean flags (True = bad, exclude from fit)

    Returns
    -------
    delay_sec : float
        Fitted delay in seconds
    offset_rad : float
        Fitted phase offset in radians
    delay_err : float
        Uncertainty in delay (seconds)
    chi2_red : float
        Reduced chi-squared of fit
    """
    if flags is None:
        flags = np.zeros(len(phases), dtype=bool)

    if weights is None:
        weights = np.ones(len(phases))

    # Mask flagged channels
    good = ~flags & np.isfinite(phases) & (weights > 0)

    if np.sum(good) < 10:
        return np.nan, np.nan, np.nan, np.nan

    f_good = freqs_hz[good]
    p_good = phases[good]
    w_good = weights[good]

    # Weighted linear fit: phase = 2π × f × delay + offset
    # Rewrite as: phase = a × f + b, where a = 2π × delay

    # Weighted least squares
    W = np.diag(w_good)
    A = np.column_stack([f_good, np.ones_like(f_good)])

    # (A^T W A)^{-1} A^T W y
    ATA = A.T @ W @ A
    ATy = A.T @ W @ p_good

    try:
        coeffs = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    slope, intercept = coeffs
    delay_sec = slope / (2 * np.pi)
    offset_rad = intercept

    # Compute residuals and chi-squared
    residuals = p_good - (slope * f_good + intercept)
    chi2 = np.sum(w_good * residuals**2)
    dof = len(p_good) - 2
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # Compute parameter uncertainties
    try:
        cov = np.linalg.inv(ATA) * chi2_red
        delay_err = np.sqrt(cov[0, 0]) / (2 * np.pi)
    except np.linalg.LinAlgError:
        delay_err = np.nan

    return delay_sec, offset_rad, delay_err, chi2_red


def unwrap_phase_robust(phases: np.ndarray, flags: np.ndarray = None) -> np.ndarray:
    """
    Robust phase unwrapping that handles flagged channels.

    Parameters
    ----------
    phases : np.ndarray
        Wrapped phases in radians
    flags : np.ndarray, optional
        Boolean flags (True = bad)

    Returns
    -------
    unwrapped : np.ndarray
        Unwrapped phases
    """
    if flags is None:
        flags = np.zeros(len(phases), dtype=bool)

    unwrapped = phases.copy()
    good_idx = np.where(~flags)[0]

    if len(good_idx) < 2:
        return unwrapped

    # Unwrap only the good channels
    good_phases = phases[good_idx]
    good_unwrapped = np.unwrap(good_phases)

    # Interpolate to fill flagged channels (for continuity)
    unwrapped[good_idx] = good_unwrapped

    # Linear interpolation for flagged channels
    for i in range(len(phases)):
        if flags[i]:
            # Find nearest good neighbors
            left_idx = good_idx[good_idx < i]
            right_idx = good_idx[good_idx > i]

            if len(left_idx) > 0 and len(right_idx) > 0:
                l = left_idx[-1]
                r = right_idx[0]
                # Linear interpolation
                unwrapped[i] = unwrapped[l] + (unwrapped[r] - unwrapped[l]) * (i - l) / (r - l)
            elif len(left_idx) > 0:
                unwrapped[i] = unwrapped[left_idx[-1]]
            elif len(right_idx) > 0:
                unwrapped[i] = unwrapped[right_idx[0]]

    return unwrapped


# =============================================================================
# Antenna Delay Solving
# =============================================================================

def solve_antenna_delays(
    baseline_delays: Dict[Tuple[int, int], float],
    baseline_errors: Dict[Tuple[int, int], float],
    n_antennas: int,
    reference_ant: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for per-antenna delays from baseline delays.

    Baseline delay: d_ij = d_i - d_j
    This is a linear system that we solve with weighted least squares.

    Parameters
    ----------
    baseline_delays : dict
        {(i, j): delay_ij} in seconds
    baseline_errors : dict
        {(i, j): error_ij} in seconds
    n_antennas : int
        Number of antennas
    reference_ant : int
        Reference antenna (delay = 0)

    Returns
    -------
    antenna_delays : np.ndarray
        Per-antenna delays in seconds, shape (n_antennas,)
    antenna_errors : np.ndarray
        Uncertainties in antenna delays
    """
    baselines = list(baseline_delays.keys())
    n_baselines = len(baselines)

    # Build design matrix A and data vector b
    # For baseline (i, j): d_ij = d_i - d_j
    # With reference antenna r: d_r = 0

    # Remove reference antenna from unknowns
    ant_list = [a for a in range(n_antennas) if a != reference_ant]
    n_unknowns = len(ant_list)
    ant_to_idx = {a: i for i, a in enumerate(ant_list)}

    A = np.zeros((n_baselines, n_unknowns))
    b = np.zeros(n_baselines)
    w = np.zeros(n_baselines)

    for k, (i, j) in enumerate(baselines):
        if i in ant_to_idx:
            A[k, ant_to_idx[i]] = 1.0
        if j in ant_to_idx:
            A[k, ant_to_idx[j]] = -1.0

        b[k] = baseline_delays[(i, j)]
        err = baseline_errors.get((i, j), 1e-9)
        w[k] = 1.0 / (err**2) if err > 0 else 1.0

    # Weighted least squares: (A^T W A)^{-1} A^T W b
    W = np.diag(w)
    ATA = A.T @ W @ A
    ATb = A.T @ W @ b

    try:
        antenna_delays_reduced = np.linalg.solve(ATA, ATb)
        cov = np.linalg.inv(ATA)
        antenna_errors_reduced = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in delay solving, using pseudo-inverse")
        antenna_delays_reduced = np.linalg.lstsq(A, b, rcond=None)[0]
        antenna_errors_reduced = np.full(n_unknowns, np.nan)

    # Reconstruct full antenna delay array
    antenna_delays = np.zeros(n_antennas)
    antenna_errors = np.zeros(n_antennas)

    for a, idx in ant_to_idx.items():
        antenna_delays[a] = antenna_delays_reduced[idx]
        antenna_errors[a] = antenna_errors_reduced[idx]

    return antenna_delays, antenna_errors


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_rfi_flagging(
    spectrum: np.ndarray,
    flags: np.ndarray,
    freqs_mhz: np.ndarray,
    output_path: str,
):
    """Plot spectrum with RFI flags highlighted."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: Full spectrum with flags
    ax = axes[0]
    ax.plot(freqs_mhz, 10*np.log10(spectrum + 1e-10), 'b-', lw=0.5, label='Spectrum')
    ax.scatter(freqs_mhz[flags], 10*np.log10(spectrum[flags] + 1e-10),
               c='red', s=5, label=f'Flagged ({np.sum(flags)}/{len(flags)})')
    ax.set_ylabel('Power (dB)')
    ax.set_title('RFI Flagging: Average Bandpass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: Good channels only
    ax = axes[1]
    good_freqs = freqs_mhz[~flags]
    good_spec = spectrum[~flags]
    ax.plot(good_freqs, 10*np.log10(good_spec + 1e-10), 'g-', lw=0.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'After RFI Flagging: {np.sum(~flags)} good channels')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_delay_fft_all_baselines(
    visibilities: np.ndarray,
    baselines: List[Tuple[int, int]],
    chan_bw_hz: float,
    flags: np.ndarray,
    baseline_delays: Dict,
    antenna_indices: List[int],
    output_path: str,
    oversample: int = 16,
):
    """Plot cross-correlation vs lag for all baselines showing delay peaks."""
    n_baselines = len(baselines)
    n_cols = 3
    n_rows = (n_baselines + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()

    n_chan = visibilities.shape[1]
    n_fft = n_chan * oversample
    total_bw_hz = n_chan * chan_bw_hz
    lag_resolution = 1.0 / total_bw_hz / oversample
    lags_ns = (np.arange(n_fft) - n_fft // 2) * lag_resolution * 1e9

    for b_idx, (i, j) in enumerate(baselines):
        ax = axes[b_idx]

        # Compute cross-correlation
        vis_clean = visibilities[b_idx, :].copy()
        vis_clean[flags] = 0
        xcorr = np.fft.ifft(vis_clean, n=n_fft)
        xcorr = np.fft.fftshift(xcorr)
        xcorr_amp = np.abs(xcorr)

        # Normalize
        xcorr_amp = xcorr_amp / xcorr_amp.max()

        # Plot
        ax.plot(lags_ns, xcorr_amp, 'b-', lw=0.5)

        # Mark fitted delay
        delay = baseline_delays.get((i, j), 0)
        if np.isfinite(delay):
            ax.axvline(x=delay*1e9, color='r', linestyle='--', lw=2,
                       label=f'τ = {delay*1e9:.2f} ns')

        ant_i = antenna_indices[i]
        ant_j = antenna_indices[j]
        ax.set_title(f'Baseline {ant_i}-{ant_j}')
        ax.set_xlabel('Lag (ns)')
        ax.set_ylabel('Cross-correlation (normalized)')
        ax.set_xlim(-200, 200)  # Focus on reasonable delay range
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_baselines, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Cross-Correlation vs Lag (FFT Method) - All Baselines', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_phase_vs_freq_all_baselines(
    visibilities: np.ndarray,
    baselines: List[Tuple[int, int]],
    freqs_mhz: np.ndarray,
    flags: np.ndarray,
    baseline_delays: Dict,
    baseline_offsets: Dict,
    antenna_indices: List[int],
    output_path: str,
):
    """Plot phase vs frequency for all baselines with fits."""
    n_baselines = len(baselines)
    n_cols = 3
    n_rows = (n_baselines + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()

    freqs_hz = freqs_mhz * 1e6

    for b_idx, (i, j) in enumerate(baselines):
        ax = axes[b_idx]

        phases = np.angle(visibilities[b_idx, :])

        # Plot wrapped phase (don't unwrap - it's unreliable)
        good = ~flags
        ax.scatter(freqs_mhz[good], phases[good], s=1, alpha=0.5, c='blue', label='Data')
        ax.scatter(freqs_mhz[flags], phases[flags], s=1, alpha=0.3, c='red', label='Flagged')

        # Plot expected phase from fitted delay (wrapped)
        delay = baseline_delays.get((i, j), 0)
        offset = baseline_offsets.get((i, j), 0)

        if np.isfinite(delay):
            fit_phase = 2 * np.pi * freqs_hz * delay + offset
            fit_phase_wrapped = np.angle(np.exp(1j * fit_phase))
            ax.plot(freqs_mhz, fit_phase_wrapped, 'g-', lw=1, alpha=0.7,
                    label=f'Fit: τ={delay*1e9:.2f} ns')

        ant_i = antenna_indices[i]
        ant_j = antenna_indices[j]
        ax.set_title(f'Baseline {ant_i}-{ant_j}')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Phase (rad)')
        ax.set_ylim(-np.pi, np.pi)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_baselines, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Phase vs Frequency (Wrapped) - All Baselines', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_antenna_delays(
    antenna_delays_ns: np.ndarray,
    antenna_errors_ns: np.ndarray,
    antenna_indices: List[int],
    adc_channels: List[int],
    output_path: str,
):
    """Plot per-antenna cable delays."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(antenna_indices))

    ax.bar(x, antenna_delays_ns, yerr=antenna_errors_ns, capsize=5,
           color='steelblue', edgecolor='black', alpha=0.7)

    ax.axhline(y=0, color='k', linestyle='--', lw=0.5)

    # Labels
    labels = [f'Ant {ant}\n(ADC {adc})' for ant, adc in zip(antenna_indices, adc_channels)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Antenna')
    ax.set_ylabel('Cable Delay (ns)')
    ax.set_title('Per-Antenna Cable Delays (Relative to Reference)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (delay, err) in enumerate(zip(antenna_delays_ns, antenna_errors_ns)):
        ax.text(i, delay + np.sign(delay)*2, f'{delay:.2f}',
                ha='center', va='bottom' if delay >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_beamformer_comparison(
    coherent_before: float,
    coherent_after: float,
    incoherent: float,
    output_path: str,
):
    """Plot beamformer power comparison before/after calibration."""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ['Incoherent\n(baseline)', 'Coherent\n(before cal)', 'Coherent\n(after cal)']
    values = [incoherent, coherent_before, coherent_after]
    colors = ['gray', 'orange', 'green']

    bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.7)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12)

    # Compute improvements
    improvement_before = coherent_before / incoherent
    improvement_after = coherent_after / incoherent

    ax.set_ylabel('Mean Power')
    ax.set_title(f'Beamformer Power Comparison\n'
                 f'Before cal: {improvement_before:.2f}x | After cal: {improvement_after:.2f}x')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_visibility_amplitude_phase(
    visibilities: np.ndarray,
    baselines: List[Tuple[int, int]],
    freqs_mhz: np.ndarray,
    flags: np.ndarray,
    antenna_indices: List[int],
    output_path: str,
):
    """Plot visibility amplitude and phase for all baselines."""
    n_baselines = len(baselines)

    fig = plt.figure(figsize=(16, 4*n_baselines))
    gs = GridSpec(n_baselines, 2, figure=fig)

    for b_idx, (i, j) in enumerate(baselines):
        ant_i = antenna_indices[i]
        ant_j = antenna_indices[j]

        vis = visibilities[b_idx, :]
        amp = np.abs(vis)
        phase = np.angle(vis)

        # Amplitude
        ax1 = fig.add_subplot(gs[b_idx, 0])
        ax1.plot(freqs_mhz[~flags], amp[~flags], 'b-', lw=0.5)
        ax1.scatter(freqs_mhz[flags], amp[flags], c='red', s=3, alpha=0.5)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Baseline {ant_i}-{ant_j}: Amplitude')
        ax1.grid(True, alpha=0.3)
        if b_idx == n_baselines - 1:
            ax1.set_xlabel('Frequency (MHz)')

        # Phase
        ax2 = fig.add_subplot(gs[b_idx, 1])
        ax2.plot(freqs_mhz[~flags], phase[~flags], 'b.', ms=1, alpha=0.5)
        ax2.scatter(freqs_mhz[flags], phase[flags], c='red', s=3, alpha=0.5)
        ax2.set_ylabel('Phase (rad)')
        ax2.set_ylim(-np.pi, np.pi)
        ax2.set_title(f'Baseline {ant_i}-{ant_j}: Phase (wrapped)')
        ax2.grid(True, alpha=0.3)
        if b_idx == n_baselines - 1:
            ax2.set_xlabel('Frequency (MHz)')

    plt.suptitle('Visibility Amplitude and Phase (All Baselines)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Main Calibration Pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("CASM Cable Delay Calibration")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1] Loading voltage data...")

    header = read_dada_header(DADA_FILE)
    obs_time = parse_utc_start(header)
    obs_datetime = datetime.fromtimestamp(obs_time, tz=timezone.utc)
    print(f"  Observation: {obs_datetime.isoformat()}")
    print(f"  Source: {CAS_A.name} (RA={CAS_A.ra_deg}°, Dec={CAS_A.dec_deg}°)")

    # Load voltage data
    n_time_samples = 65536  # Use more samples for better SNR
    voltages_dict, _ = read_dada_data(DADA_FILE, n_time=n_time_samples, snaps=[2])

    # Get antenna mapping
    positions_enu, adc_channels, antenna_indices = get_antenna_mapping()

    # Extract active ADC channels
    snap_data = voltages_dict[2]
    voltages = snap_data[:, :, adc_channels]
    print(f"  Voltages shape: {voltages.shape} (time, chan, ant)")

    # Setup frequency config
    n_chan = voltages.shape[1]
    freq_config = FrequencyConfig(
        n_chan=n_chan,
        total_bw_mhz=125.0,
        total_n_chan=4096,
        freq_end_voltage_mhz=468.75
    )
    freqs_mhz = freq_config.get_frequencies_mhz()
    freqs_hz = freqs_mhz * 1e6

    print(f"  Frequency range: {freqs_mhz[0]:.2f} - {freqs_mhz[-1]:.2f} MHz")

    # =========================================================================
    # Step 2: Apply geometric delays (fringe stopping)
    # =========================================================================
    print("\n[2] Applying geometric delays (fringe stopping toward Cas A)...")

    array_config = ArrayConfig(
        positions_enu=positions_enu,
        antenna_flags=np.ones(N_ANTENNAS, dtype=bool)
    )

    bf = GeometricBeamformer(
        array_config=array_config,
        freq_config=freq_config,
        use_astropy=True
    )

    # Compute geometric weights
    geo_weights = bf.compute_tracking_weights(
        phase_centers=[CAS_A],
        unix_times=np.array([obs_time]),
        mode='coherent'
    )
    geo_w = geo_weights.weights[0, 0, :, :]  # (n_ant, n_chan)

    # Apply geometric correction to voltages
    # voltages: (n_time, n_chan, n_ant)
    # weights: (n_ant, n_chan) -> transpose for broadcasting
    voltages_geo = voltages * geo_w.T[np.newaxis, :, :]

    print(f"  Applied geometric delays to {N_ANTENNAS} antennas")

    # Compute geometric delays for reference
    geo_delays = bf.compute_tracking_delays(CAS_A, np.array([obs_time]))[0]
    print("\n  Geometric delays (ns):")
    for i, (adc_ch, ant_idx) in enumerate(zip(adc_channels, antenna_indices)):
        print(f"    ADC {adc_ch:2d} (Ant {ant_idx:2d}): {geo_delays[i]*1e9:8.3f} ns")

    # =========================================================================
    # Step 3: Compute visibilities
    # =========================================================================
    print("\n[3] Computing cross-correlations (visibilities)...")

    visibilities, baselines = compute_visibilities(voltages_geo)
    autocorrs = compute_autocorrelations(voltages_geo)

    print(f"  Computed {len(baselines)} baselines")

    # =========================================================================
    # Step 4: RFI flagging
    # =========================================================================
    print("\n[4] Flagging RFI...")

    # Use visibility amplitudes for flagging
    vis_amp = np.abs(visibilities)
    avg_vis_amp = np.mean(vis_amp, axis=0)

    flags, flag_stats = flag_rfi_channels(avg_vis_amp)

    print(f"  Flagged {flag_stats['n_flagged']}/{flag_stats['n_total']} channels "
          f"({flag_stats['frac_flagged']*100:.1f}%)")

    # Identify RFI bands
    rfi_bands = identify_rfi_bands(flags, freqs_mhz)
    if rfi_bands:
        print("  Major RFI bands:")
        for f1, f2 in rfi_bands[:5]:  # Show first 5
            print(f"    {f1:.2f} - {f2:.2f} MHz")

    # Plot RFI flagging
    plot_rfi_flagging(
        avg_vis_amp,
        flags,
        freqs_mhz,
        f"{OUTPUT_DIR}/calibration_rfi_flagging.png"
    )

    # =========================================================================
    # Step 5: Fit delays using FFT method (more robust than phase unwrapping)
    # =========================================================================
    print("\n[5] Fitting residual delays using FFT cross-correlation method...")

    baseline_delays = {}
    baseline_offsets = {}
    baseline_errors = {}
    baseline_snrs = {}

    chan_bw_hz = freq_config.chan_bw_hz
    MIN_SNR_THRESHOLD = 15.0  # Minimum SNR to trust a baseline

    for b_idx, (i, j) in enumerate(baselines):
        # Use FFT-based delay fitting (more robust)
        delay, delay_err, snr = fit_delay_fft(
            visibilities[b_idx, :], chan_bw_hz, flags
        )

        # Also compute phase offset at center frequency
        center_idx = n_chan // 2
        offset = np.angle(visibilities[b_idx, center_idx])

        baseline_delays[(i, j)] = delay
        baseline_offsets[(i, j)] = offset
        baseline_errors[(i, j)] = delay_err if np.isfinite(delay_err) else 1e-9
        baseline_snrs[(i, j)] = snr

        ant_i = antenna_indices[i]
        ant_j = antenna_indices[j]
        status = "OK" if snr > MIN_SNR_THRESHOLD else "LOW SNR"
        print(f"  Baseline {ant_i:2d}-{ant_j:2d}: τ = {delay*1e9:8.3f} ns, SNR = {snr:6.1f}  {status}")

    # Identify bad antennas based on consistently low SNR
    print("\n  Checking for bad antennas (low SNR on all baselines)...")
    antenna_snr_counts = {i: [] for i in range(N_ANTENNAS)}
    for (i, j), snr in baseline_snrs.items():
        antenna_snr_counts[i].append(snr)
        antenna_snr_counts[j].append(snr)

    bad_antennas = []
    for ant_idx, snrs in antenna_snr_counts.items():
        median_snr = np.median(snrs)
        if median_snr < MIN_SNR_THRESHOLD:
            bad_antennas.append(ant_idx)
            print(f"    Antenna {antenna_indices[ant_idx]} (ADC {adc_channels[ant_idx]}): "
                  f"median SNR = {median_snr:.1f} -> FLAGGED")

    # Remove bad baselines from delay solving
    if bad_antennas:
        print(f"\n  Excluding baselines with flagged antennas...")
        for (i, j) in list(baseline_delays.keys()):
            if i in bad_antennas or j in bad_antennas:
                del baseline_delays[(i, j)]
                del baseline_errors[(i, j)]
                print(f"    Removed baseline {antenna_indices[i]}-{antenna_indices[j]}")

    # Plot FFT cross-correlation (delay fitting)
    plot_delay_fft_all_baselines(
        visibilities, baselines, chan_bw_hz, flags,
        baseline_delays, antenna_indices,
        f"{OUTPUT_DIR}/calibration_delay_fft.png"
    )

    # Plot phase vs frequency
    plot_phase_vs_freq_all_baselines(
        visibilities, baselines, freqs_mhz, flags,
        baseline_delays, baseline_offsets, antenna_indices,
        f"{OUTPUT_DIR}/calibration_phase_vs_freq.png"
    )

    # Plot visibility amplitude and phase
    plot_visibility_amplitude_phase(
        visibilities, baselines, freqs_mhz, flags, antenna_indices,
        f"{OUTPUT_DIR}/calibration_visibilities.png"
    )

    # =========================================================================
    # Step 6: Solve for per-antenna cable delays
    # =========================================================================
    print("\n[6] Solving for per-antenna cable delays...")

    # Choose reference antenna (must not be a bad antenna)
    reference_ant = 0
    if reference_ant in bad_antennas:
        for i in range(N_ANTENNAS):
            if i not in bad_antennas:
                reference_ant = i
                break

    # Only solve for good antennas
    n_good_antennas = N_ANTENNAS - len(bad_antennas)
    print(f"  Using {n_good_antennas}/{N_ANTENNAS} good antennas")

    antenna_delays, antenna_errors = solve_antenna_delays(
        baseline_delays, baseline_errors, N_ANTENNAS, reference_ant
    )

    # Set bad antennas to NaN
    for bad_ant in bad_antennas:
        antenna_delays[bad_ant] = np.nan
        antenna_errors[bad_ant] = np.nan

    antenna_delays_ns = antenna_delays * 1e9
    antenna_errors_ns = antenna_errors * 1e9

    print(f"\n  Per-antenna cable delays (reference: Ant {antenna_indices[reference_ant]}):")
    print(f"  {'ADC':>5} {'Ant':>5} {'Delay (ns)':>12} {'Error (ns)':>12} {'Status':>10}")
    print(f"  {'-'*55}")
    for i, (adc_ch, ant_idx) in enumerate(zip(adc_channels, antenna_indices)):
        status = "FLAGGED" if i in bad_antennas else "OK"
        if np.isnan(antenna_delays_ns[i]):
            print(f"  {adc_ch:5d} {ant_idx:5d} {'N/A':>12} {'N/A':>12} {status:>10}")
        else:
            print(f"  {adc_ch:5d} {ant_idx:5d} {antenna_delays_ns[i]:12.3f} {antenna_errors_ns[i]:12.3f} {status:>10}")

    # Plot antenna delays
    plot_antenna_delays(
        antenna_delays_ns, antenna_errors_ns, antenna_indices, adc_channels,
        f"{OUTPUT_DIR}/calibration_antenna_delays.png"
    )

    # =========================================================================
    # Step 7: Apply cable delay corrections and re-test beamforming
    # =========================================================================
    print("\n[7] Testing beamforming with cable delay corrections...")

    # Create antenna flag mask (True = good antenna)
    good_antenna_mask = np.ones(N_ANTENNAS, dtype=bool)
    for bad_ant in bad_antennas:
        good_antenna_mask[bad_ant] = False

    print(f"  Using {np.sum(good_antenna_mask)}/{N_ANTENNAS} good antennas for beamforming")

    # Compute cable delay correction weights
    # Set bad antennas to zero delay (their weights will be zeroed anyway)
    antenna_delays_clean = np.nan_to_num(antenna_delays, nan=0.0)
    cable_phases = 2 * np.pi * freqs_hz[np.newaxis, :] * antenna_delays_clean[:, np.newaxis]
    cable_weights = np.exp(-1j * cable_phases).astype(np.complex64)

    # Zero out bad antennas
    for bad_ant in bad_antennas:
        cable_weights[bad_ant, :] = 0

    # Total calibrated weights = geometric × cable correction
    calibrated_weights = geo_w * cable_weights  # (n_ant, n_chan)

    # Also zero out bad antennas in geometric-only weights for fair comparison
    geo_w_masked = geo_w.copy()
    for bad_ant in bad_antennas:
        geo_w_masked[bad_ant, :] = 0

    # Beamform with different weight sets
    # 1. Incoherent (baseline) - using all good antennas
    v = voltages.transpose(0, 2, 1)  # (n_time, n_ant, n_chan)

    incoherent_weights = np.ones((N_ANTENNAS, n_chan), dtype=np.complex64)
    for bad_ant in bad_antennas:
        incoherent_weights[bad_ant, :] = 0

    incoherent_beam = np.sum(v * incoherent_weights[np.newaxis, :, :], axis=1)
    incoherent_power = np.mean(np.abs(incoherent_beam[:, ~flags])**2)

    # 2. Coherent (geometric only - before calibration)
    coherent_before_beam = np.sum(v * geo_w_masked[np.newaxis, :, :], axis=1)
    coherent_before_power = np.mean(np.abs(coherent_before_beam[:, ~flags])**2)

    # 3. Coherent (geometric + cable - after calibration)
    coherent_after_beam = np.sum(v * calibrated_weights[np.newaxis, :, :], axis=1)
    coherent_after_power = np.mean(np.abs(coherent_after_beam[:, ~flags])**2)

    print(f"\n  Results (RFI-flagged channels excluded):")
    print(f"  {'Mode':<25} {'Mean Power':>12} {'Ratio':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Incoherent (baseline)':<25} {incoherent_power:12.4f} {1.0:10.2f}x")
    print(f"  {'Coherent (geo only)':<25} {coherent_before_power:12.4f} "
          f"{coherent_before_power/incoherent_power:10.2f}x")
    print(f"  {'Coherent (geo + cable)':<25} {coherent_after_power:12.4f} "
          f"{coherent_after_power/incoherent_power:10.2f}x")

    improvement = (coherent_after_power - coherent_before_power) / coherent_before_power * 100
    print(f"\n  Improvement from cable calibration: {improvement:+.1f}%")

    # Plot comparison
    plot_beamformer_comparison(
        coherent_before_power, coherent_after_power, incoherent_power,
        f"{OUTPUT_DIR}/calibration_beamformer_comparison.png"
    )

    # =========================================================================
    # Step 8: Save results
    # =========================================================================
    print("\n[8] Saving results...")

    # Save visibilities
    vis_file = f"{OUTPUT_DIR}/calibration_visibilities.npz"
    np.savez(
        vis_file,
        visibilities=visibilities,
        baselines=np.array(baselines),
        autocorrelations=autocorrs,
        frequencies_mhz=freqs_mhz,
        flags=flags,
        antenna_indices=np.array(antenna_indices),
        adc_channels=np.array(adc_channels),
        obs_time=obs_time,
    )
    print(f"  Saved visibilities: {vis_file}")

    # Save calibration solutions
    cal_file = f"{OUTPUT_DIR}/calibration_solutions.npz"

    # Build baseline delay arrays only for remaining baselines
    remaining_baselines = list(baseline_delays.keys())
    np.savez(
        cal_file,
        antenna_delays_sec=antenna_delays,
        antenna_delays_ns=antenna_delays_ns,
        antenna_errors_ns=antenna_errors_ns,
        geometric_delays_sec=geo_delays,
        geometric_delays_ns=geo_delays * 1e9,
        total_delays_ns=np.where(good_antenna_mask, (geo_delays + np.nan_to_num(antenna_delays)) * 1e9, np.nan),
        baseline_delays_sec=np.array([baseline_delays[b] for b in remaining_baselines]),
        baseline_errors_sec=np.array([baseline_errors[b] for b in remaining_baselines]),
        baselines=np.array(remaining_baselines),
        antenna_indices=np.array(antenna_indices),
        adc_channels=np.array(adc_channels),
        reference_antenna=reference_ant,
        reference_antenna_index=antenna_indices[reference_ant],
        calibrated_weights=calibrated_weights,
        cable_weights=cable_weights,
        geometric_weights=geo_w,
        frequencies_mhz=freqs_mhz,
        rfi_flags=flags,
        good_antenna_mask=good_antenna_mask,
        bad_antennas=np.array(bad_antennas),
        bad_antenna_indices=np.array([antenna_indices[i] for i in bad_antennas]),
        source_ra_deg=CAS_A.ra_deg,
        source_dec_deg=CAS_A.dec_deg,
        source_name=CAS_A.name,
        obs_time=obs_time,
    )
    print(f"  Saved calibration: {cal_file}")

    # =========================================================================
    # Summary
    # =========================================================================
    n_good = np.sum(good_antenna_mask)
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Observation: {obs_datetime.isoformat()}")
    print(f"Calibrator: {CAS_A.name}")
    print(f"Antennas: {n_good}/{N_ANTENNAS} good")
    if bad_antennas:
        bad_ant_names = [f"{antenna_indices[i]} (ADC {adc_channels[i]})" for i in bad_antennas]
        print(f"  Flagged antennas: {', '.join(bad_ant_names)}")
    print(f"Reference antenna: {antenna_indices[reference_ant]} (ADC {adc_channels[reference_ant]})")
    print(f"RFI flagged: {np.sum(flags)}/{len(flags)} channels ({np.sum(flags)/len(flags)*100:.1f}%)")
    print(f"\nBeamforming improvement:")
    print(f"  Before cable cal: {coherent_before_power/incoherent_power:.2f}x incoherent")
    print(f"  After cable cal:  {coherent_after_power/incoherent_power:.2f}x incoherent")
    print(f"  Cable cal gain:   {improvement:+.1f}%")
    expected_gain = n_good  # For coherent phased array on point source
    print(f"  Expected max (N_ant): {expected_gain:.1f}x")
    print("=" * 70)

    return {
        'antenna_delays_ns': antenna_delays_ns,
        'antenna_errors_ns': antenna_errors_ns,
        'calibrated_weights': calibrated_weights,
        'incoherent_power': incoherent_power,
        'coherent_before_power': coherent_before_power,
        'coherent_after_power': coherent_after_power,
    }


if __name__ == "__main__":
    results = main()
