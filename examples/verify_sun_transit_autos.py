#!/usr/bin/env python3
"""
Verify Sun Transit Signal — Auto vs Cross Diagnostic.

Separates auto-correlation (total power) from cross-correlation (coherent)
contributions to determine whether beam 0's enhancement during the 2026-03-12
sun transit is real sidelobe response or auto-correlation leakage.

Produces a 4-panel diagnostic figure:
  1. Auto-correlation total power vs time (+ sun altitude)
  2. Cross-only beamformed power vs time for each beam direction
  3. Normalized cross power (cross / auto) vs time
  4. Auto power per antenna vs time

Usage:
    python examples/verify_sun_transit_autos.py
    python examples/verify_sun_transit_autos.py --obs 2026-03-12-16:32:52
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time
    import astropy.units as u
except ImportError:
    print("Error: astropy is required. Install with: pip install astropy",
          file=sys.stderr)
    sys.exit(1)

from casm_calibrator.visibility import VisibilityLoader
from casm_calibrator.rfi import RFIMask
from casm_io.correlator.mapping import AntennaMapping
from casm_io.constants import C_LIGHT_M_S, OVRO_LAT_DEG, OVRO_LON_DEG, OVRO_ELEV_M

from examples.check_source_visibility import load_beams_from_hdf5

# ── Constants ────────────────────────────────────────────────────────────────
DUMPED_BEAMS = [0, 81, 147, 186, 229, 272, 366]

BEAM_COLORS = {
    0:   "#999999",  # gray — zenith, far from sun
    81:  "#4363d8",  # blue
    147: "#42d4f4",  # cyan
    186: "#3cb44b",  # green
    229: "#e6194b",  # red — on-source
    272: "#f58231",  # orange
    366: "#911eb4",  # purple
}

PST = ZoneInfo("America/Los_Angeles")
OVRO_LOCATION = EarthLocation(
    lat=OVRO_LAT_DEG * u.deg,
    lon=OVRO_LON_DEG * u.deg,
    height=OVRO_ELEV_M * u.m,
)


# ── Helper functions ─────────────────────────────────────────────────────────

def altaz_to_lmn(alt_deg, az_deg):
    """Convert alt/az (degrees) to direction cosines (l, m, n) in ENU frame."""
    alt_rad = np.deg2rad(alt_deg)
    az_rad = np.deg2rad(az_deg)
    l = np.cos(alt_rad) * np.sin(az_rad)   # East
    m = np.cos(alt_rad) * np.cos(az_rad)   # North
    n = np.sin(alt_rad)                     # Up
    return np.array([l, m, n])


def load_beam_directions(weights_path, beam_indices):
    """Load beam alt/az from HDF5, return (n_dir, 3) direction cosines."""
    beam_alt, beam_az, _, _, _ = load_beams_from_hdf5(weights_path)
    lmn = np.zeros((len(beam_indices), 3))
    for i, bi in enumerate(beam_indices):
        lmn[i] = altaz_to_lmn(beam_alt[bi], beam_az[bi])
    return lmn, beam_alt[beam_indices], beam_az[beam_indices]


def sun_altaz_track(time_unix):
    """Sun alt/az at each timestamp. Returns alt_deg (T,), az_deg (T,)."""
    times = Time(time_unix, format="unix")
    altaz_frame = AltAz(obstime=times, location=OVRO_LOCATION)
    sun_coords = get_sun(times).transform_to(altaz_frame)
    return sun_coords.alt.deg, sun_coords.az.deg


def compute_auto_power(vis, rfi_mask):
    """Diagonal of vis matrix, averaged over good channels.

    Parameters
    ----------
    vis : np.ndarray, shape (T, F, n_ant, n_ant)
    rfi_mask : np.ndarray, shape (F,), bool (True=good)

    Returns
    -------
    total : np.ndarray, shape (T,)
        Sum of all antenna autos, averaged over good channels.
    per_ant : np.ndarray, shape (T, n_ant)
        Per-antenna auto power, averaged over good channels.
    """
    n_ant = vis.shape[2]
    # Extract diagonal: vis[:, :, i, i] for all i
    diag_idx = np.arange(n_ant)
    autos = vis[:, :, diag_idx, diag_idx].real  # (T, F, n_ant)

    # Average over good channels
    good_autos = autos[:, rfi_mask, :]  # (T, F_good, n_ant)
    per_ant = good_autos.mean(axis=1)   # (T, n_ant)
    total = per_ant.sum(axis=1)         # (T,)
    return total, per_ant


def compute_cross_beamform_power(vis, freqs_hz, positions, directions_lmn,
                                  rfi_mask):
    """Vectorized cross-only beamforming.

    Parameters
    ----------
    vis : np.ndarray, shape (T, F, n_ant, n_ant)
    freqs_hz : np.ndarray, shape (F,)
    positions : np.ndarray, shape (n_ant, 3) ENU meters
    directions_lmn : np.ndarray, shape (n_dir, 3)
    rfi_mask : np.ndarray, shape (F,), bool (True=good)

    Returns
    -------
    power : np.ndarray, shape (T, n_dir)
    """
    n_time = vis.shape[0]
    n_ant = vis.shape[2]
    n_dir = directions_lmn.shape[0]

    # Cross-baseline indices
    ii, jj = np.triu_indices(n_ant, k=1)  # upper triangle, excluding diagonal
    baselines = positions[jj] - positions[ii]  # (n_bl, 3)

    # Geometric delays: tau = baseline . direction / c
    tau = baselines @ directions_lmn.T / C_LIGHT_M_S  # (n_bl, n_dir)

    # Good frequencies
    freqs_good = freqs_hz[rfi_mask]  # (F_good,)
    n_good = len(freqs_good)

    # Phasors: exp(-2j*pi*freq*tau) → (n_bl, F_good, n_dir)
    # tau: (n_bl, n_dir), freqs_good: (F_good,)
    phase = -2.0 * np.pi * freqs_good[np.newaxis, :, np.newaxis] * \
        tau[:, np.newaxis, :]  # (n_bl, F_good, n_dir)
    phasor = np.exp(1j * phase)  # (n_bl, F_good, n_dir)

    power = np.zeros((n_time, n_dir))
    for t in range(n_time):
        # Extract cross-correlations for good channels
        v_cross = vis[t][rfi_mask][:, ii, jj]  # (F_good, n_bl)

        # Beamform: sum over baselines and channels
        # v_cross: (F_good, n_bl), phasor: (n_bl, F_good, n_dir)
        # Want: sum_f sum_b v_cross[f,b] * phasor[b,f,d]
        bf = np.einsum("fb,bfd->d", v_cross, phasor)  # (n_dir,)
        power[t] = 2.0 * bf.real / n_good

    return power


def make_offsource_direction(sun_alt_transit, sun_az_transit, az_offset_deg=10.0):
    """Create off-source control direction: same alt as sun at transit, az+offset."""
    return altaz_to_lmn(sun_alt_transit, sun_az_transit + az_offset_deg)


def load_cal_gains(npz_path, ant_ids):
    """Load calibration gains from NPZ, matched to ant_ids ordering.

    Parameters
    ----------
    npz_path : str
        Path to SVD calibration .npz file.
    ant_ids : np.ndarray
        Antenna IDs from the visibility matrix (1-indexed).

    Returns
    -------
    gains : np.ndarray, shape (n_ant, n_chan), complex64
    flags : np.ndarray, shape (n_chan,), bool (True=good)
    """
    d = np.load(npz_path, allow_pickle=True)
    gains = d["gains"]        # (n_ant_cal, n_chan)
    cal_ant_ids = d["ant_ids"]
    cal_freqs = d["freqs_hz"] if "freqs_hz" in d else d["freqs_mhz"] * 1e6
    flags = d["flags"]        # (n_chan,) bool, True=good

    # Ensure ascending frequency order
    if len(cal_freqs) > 1 and cal_freqs[1] < cal_freqs[0]:
        gains = gains[:, ::-1]
        flags = flags[::-1]

    # Match antenna ordering to visibility matrix
    cal_id_to_idx = {int(aid): i for i, aid in enumerate(cal_ant_ids)}
    n_ant = len(ant_ids)
    n_chan = gains.shape[1]
    g_out = np.ones((n_ant, n_chan), dtype=np.complex64)
    matched = 0
    for i, aid in enumerate(ant_ids):
        if int(aid) in cal_id_to_idx:
            g_out[i] = gains[cal_id_to_idx[int(aid)]]
            matched += 1

    print(f"  Matched {matched}/{n_ant} antennas to calibration gains")
    print(f"  Good cal channels: {np.sum(flags)}/{len(flags)}")
    return g_out, flags


def apply_calibration(vis, gains):
    """Apply gain calibration to visibility matrix in-place.

    V_cal[t, f, i, j] = conj(g[i, f]) * V[t, f, i, j] * g[j, f]

    Parameters
    ----------
    vis : np.ndarray, shape (T, F, n_ant, n_ant)
    gains : np.ndarray, shape (n_ant, n_chan)

    Returns
    -------
    vis_cal : np.ndarray, same shape as vis
    """
    g_conj = np.conj(gains)  # (n_ant, n_chan)
    vis_cal = vis.copy()
    n_chan = vis.shape[1]
    n_ant = gains.shape[0]
    for f in range(n_chan):
        # g_row broadcasts over T (axis 0) and columns (axis 2)
        g_row = g_conj[:, f].reshape(1, n_ant, 1)   # (1, n_ant, 1)
        # g_col broadcasts over T (axis 0) and rows (axis 1)
        g_col = gains[:, f].reshape(1, 1, n_ant)     # (1, 1, n_ant)
        vis_cal[:, f, :, :] = g_row * vis[:, f, :, :] * g_col
    return vis_cal


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_diagnostic_plot(time_unix, auto_total, auto_per_ant,
                         cross_power, beam_indices, sun_alt, sun_az,
                         output_path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    n_beams = len(beam_indices)
    # n_dir = n_beams + 1 (off-source is last)

    # Convert unix times to PST datetimes
    dt_pst = [datetime.fromtimestamp(t, tz=PST) for t in time_unix]

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    ax_auto, ax_cross, ax_ratio, ax_per_ant = axes

    # ── Panel 1: Auto-correlation total power ────────────────────────────
    ax_auto.plot(dt_pst, auto_total, "k-", linewidth=1.5, label="Total auto power")
    ax_auto.set_ylabel("Auto power (arb. units)", fontsize=11)
    ax_auto.set_title("Panel 1 — Auto-correlation Total Power", fontsize=13,
                      fontweight="bold")
    ax_auto.legend(loc="upper left", fontsize=9)
    ax_auto.grid(True, linestyle="--", alpha=0.3)

    # Secondary y-axis: sun altitude
    ax_sun = ax_auto.twinx()
    ax_sun.plot(dt_pst, sun_alt, color="gold", linewidth=2.0, linestyle="--",
                alpha=0.8, label="Sun altitude")
    ax_sun.set_ylabel("Sun altitude (deg)", fontsize=11, color="goldenrod")
    ax_sun.tick_params(axis="y", labelcolor="goldenrod")
    ax_sun.legend(loc="upper right", fontsize=9)

    # ── Panel 2: Cross-only beamformed power ─────────────────────────────
    for i, bi in enumerate(beam_indices):
        color = BEAM_COLORS.get(bi, "black")
        ax_cross.plot(dt_pst, cross_power[:, i], color=color, linewidth=1.5,
                      label=f"Beam {bi}", alpha=0.9)
    # Off-source control (last column)
    ax_cross.plot(dt_pst, cross_power[:, -1], color="gray", linewidth=1.5,
                  linestyle="--", label="Off-source", alpha=0.7)
    ax_cross.set_ylabel("Cross-only beamformed power", fontsize=11)
    ax_cross.set_title("Panel 2 — Cross-only Beamformed Power", fontsize=13,
                       fontweight="bold")
    ax_cross.legend(loc="upper right", fontsize=8, ncol=3)
    ax_cross.grid(True, linestyle="--", alpha=0.3)

    # ── Panel 3: Normalized cross power (cross / auto) ───────────────────
    # Normalize each beam's cross power by total auto power
    auto_safe = np.where(auto_total > 0, auto_total, 1.0)
    for i, bi in enumerate(beam_indices):
        color = BEAM_COLORS.get(bi, "black")
        ratio = cross_power[:, i] / auto_safe
        ax_ratio.plot(dt_pst, ratio, color=color, linewidth=1.5,
                      label=f"Beam {bi}", alpha=0.9)
    # Off-source control
    ratio_off = cross_power[:, -1] / auto_safe
    ax_ratio.plot(dt_pst, ratio_off, color="gray", linewidth=1.5,
                  linestyle="--", label="Off-source", alpha=0.7)
    ax_ratio.set_ylabel("Cross / Auto ratio", fontsize=11)
    ax_ratio.set_title("Panel 3 — Normalized Cross Power (Cross / Auto)",
                       fontsize=13, fontweight="bold")
    ax_ratio.legend(loc="upper right", fontsize=8, ncol=3)
    ax_ratio.grid(True, linestyle="--", alpha=0.3)

    # ── Panel 4: Auto power per antenna ──────────────────────────────────
    n_ant = auto_per_ant.shape[1]
    cmap = plt.cm.tab20
    for a in range(n_ant):
        ax_per_ant.plot(dt_pst, auto_per_ant[:, a], linewidth=1.0,
                        color=cmap(a / n_ant), label=f"Ant {a}", alpha=0.8)
    ax_per_ant.set_ylabel("Auto power per antenna", fontsize=11)
    ax_per_ant.set_xlabel("Time (PST)", fontsize=11)
    ax_per_ant.set_title("Panel 4 — Auto Power per Antenna", fontsize=13,
                         fontweight="bold")
    ax_per_ant.legend(loc="upper right", fontsize=7, ncol=4)
    ax_per_ant.grid(True, linestyle="--", alpha=0.3)

    # Format x-axis
    ax_per_ant.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=PST))
    ax_per_ant.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    fig.autofmt_xdate(rotation=45)

    title = "Sun Transit Auto vs Cross Diagnostic — 2026-03-12 (OVRO)"
    if title_suffix:
        title += f"\n{title_suffix}"
    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(time_unix, auto_total, cross_power, beam_indices):
    """Print summary table with peak cross power per beam."""
    print("\n" + "=" * 72)
    print("Summary: Peak Cross Power per Beam")
    print("=" * 72)
    print(f"{'Beam':>6s}  {'Peak Cross':>12s}  {'Time (PST)':>14s}  "
          f"{'Cross/Auto':>12s}")
    print("-" * 72)

    auto_safe = np.where(auto_total > 0, auto_total, 1.0)

    for i, bi in enumerate(beam_indices):
        cross = cross_power[:, i]
        idx_peak = np.argmax(np.abs(cross))
        peak_val = cross[idx_peak]
        peak_time = datetime.fromtimestamp(time_unix[idx_peak], tz=PST)
        ratio = peak_val / auto_safe[idx_peak]
        label = f"Beam {bi}"
        print(f"{label:>6s}  {peak_val:12.4f}  {peak_time.strftime('%H:%M:%S'):>14s}  "
              f"{ratio:12.6f}")

    # Off-source
    cross_off = cross_power[:, -1]
    idx_peak = np.argmax(np.abs(cross_off))
    peak_val = cross_off[idx_peak]
    peak_time = datetime.fromtimestamp(time_unix[idx_peak], tz=PST)
    ratio = peak_val / auto_safe[idx_peak]
    print(f"{'Off':>6s}  {peak_val:12.4f}  {peak_time.strftime('%H:%M:%S'):>14s}  "
          f"{ratio:12.6f}")
    print("=" * 72)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify sun transit signal: auto vs cross diagnostic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="/mnt/nvme3/data/casm/visibilities_64ant/",
        help="Visibility data directory",
    )
    parser.add_argument(
        "--obs", default="2026-03-12-16:32:52",
        help="Observation ID",
    )
    parser.add_argument(
        "--layout",
        default=str(Path.home() / "software/dev/antenna_layouts"
                    / "antenna_layout_current.csv"),
        help="Antenna layout CSV path",
    )
    parser.add_argument(
        "--weights",
        default=str(Path.home() / "software/dev/casm_calibrator/results"
                    / "int8_weights_512beam_mar10_svd.h5"),
        help="HDF5 weights file with beam pointings",
    )
    parser.add_argument(
        "--beam-indices", nargs="+", type=int, default=DUMPED_BEAMS,
        help="Beam indices to analyze",
    )
    parser.add_argument(
        "--rfi-mask-range", nargs=2, type=float, action="append",
        default=None, metavar=("LOW", "HIGH"),
        help="RFI frequency range to mask in MHz (repeatable)",
    )
    parser.add_argument(
        "--cal",
        default=str(Path.home() / "software/dev/casm_calibrator/results"
                    / "svd_weights_mar10.npz"),
        help="SVD calibration .npz file (set to 'none' to skip)",
    )
    parser.add_argument(
        "--az-offset", type=float, default=10.0,
        help="Azimuth offset in degrees for off-source control",
    )
    parser.add_argument(
        "-o", "--output",
        default="examples/verify_sun_transit_autos_20260312.png",
        help="Output PNG path (uncalibrated)",
    )
    parser.add_argument(
        "--output-cal",
        default="examples/verify_sun_transit_autos_cal_20260312.png",
        help="Output PNG path (calibrated)",
    )
    args = parser.parse_args()

    # Default RFI mask if not specified
    if args.rfi_mask_range is None:
        rfi_ranges = [(375.0, 390.0)]
    else:
        rfi_ranges = [tuple(r) for r in args.rfi_mask_range]

    beam_indices = args.beam_indices

    # ── Load antenna mapping ─────────────────────────────────────────────
    print(f"Loading antenna mapping: {args.layout}")
    mapping = AntennaMapping.load(args.layout)

    # ── Load visibility data ─────────────────────────────────────────────
    print(f"Loading visibilities: {args.obs}")
    print(f"  Data dir: {args.data_dir}")
    loader = VisibilityLoader(mapping)
    vm = loader.load(args.data_dir, args.obs)
    print(f"  Shape: {vm.vis.shape}  "
          f"({vm.vis.shape[0]} times, {vm.vis.shape[1]} chans, "
          f"{vm.vis.shape[2]} ants)")
    print(f"  Time range: "
          f"{datetime.fromtimestamp(vm.time_unix[0], tz=PST).strftime('%H:%M:%S')} - "
          f"{datetime.fromtimestamp(vm.time_unix[-1], tz=PST).strftime('%H:%M:%S')} PST")

    # ── RFI mask ─────────────────────────────────────────────────────────
    print(f"RFI mask ranges: {rfi_ranges}")
    rfi_mask_obj = RFIMask(bad_ranges_mhz=rfi_ranges)
    rfi_mask = rfi_mask_obj(vm.freq_mhz)
    n_good = np.sum(rfi_mask)
    print(f"  Good channels: {n_good} / {len(vm.freq_mhz)}")

    # ── Load beam directions ─────────────────────────────────────────────
    print(f"Loading beam directions from: {args.weights}")
    directions_lmn, beam_alt, beam_az = load_beam_directions(
        args.weights, beam_indices,
    )
    print(f"  {len(beam_indices)} beams loaded:")
    for i, bi in enumerate(beam_indices):
        print(f"    Beam {bi:3d}: alt={beam_alt[i]:.2f} az={beam_az[i]:.2f}")

    # ── Sun track ────────────────────────────────────────────────────────
    print("Computing sun track...")
    sun_alt, sun_az = sun_altaz_track(vm.time_unix)
    i_transit = np.argmax(sun_alt)
    t_transit = datetime.fromtimestamp(vm.time_unix[i_transit], tz=PST)
    print(f"  Sun transit: alt={sun_alt[i_transit]:.2f} "
          f"az={sun_az[i_transit]:.2f} at {t_transit.strftime('%H:%M:%S')} PST")

    # ── Off-source control direction ─────────────────────────────────────
    off_lmn = make_offsource_direction(
        sun_alt[i_transit], sun_az[i_transit], args.az_offset,
    )
    print(f"  Off-source control: sun transit alt, az+{args.az_offset}")

    # Combine beam directions + off-source
    all_directions = np.vstack([directions_lmn, off_lmn[np.newaxis, :]])

    # ── Compute auto power ───────────────────────────────────────────────
    print("Computing auto-correlation power...")
    auto_total, auto_per_ant = compute_auto_power(vm.vis, rfi_mask)
    print(f"  Auto power range: {auto_total.min():.2f} - {auto_total.max():.2f}")

    # ── Compute cross-only beamformed power ──────────────────────────────
    freqs_hz = vm.freq_mhz * 1e6
    print(f"Computing cross-only beamformed power "
          f"({vm.vis.shape[0]} times x {len(all_directions)} dirs)...")
    cross_power = compute_cross_beamform_power(
        vm.vis, freqs_hz, vm.positions_enu, all_directions, rfi_mask,
    )
    print("  Done.")

    # ── Summary (uncalibrated) ───────────────────────────────────────────
    print("\n*** UNCALIBRATED ***")
    print_summary(vm.time_unix, auto_total, cross_power, beam_indices)

    # ── Plot (uncalibrated) ───────────────────────────────────────────────
    output_path = Path(args.output)
    print(f"\nGenerating uncalibrated diagnostic plot: {output_path}")
    make_diagnostic_plot(
        vm.time_unix, auto_total, auto_per_ant,
        cross_power, beam_indices, sun_alt, sun_az,
        output_path, title_suffix="(Uncalibrated)",
    )

    # ── Calibrated analysis ──────────────────────────────────────────────
    if args.cal.lower() != "none":
        print(f"\n{'='*72}")
        print("CALIBRATED ANALYSIS")
        print(f"{'='*72}")
        print(f"Loading calibration: {args.cal}")
        gains, cal_flags = load_cal_gains(args.cal, vm.ant_ids)

        # Combine RFI mask with calibration flags
        cal_rfi_mask = rfi_mask & cal_flags
        n_good_cal = np.sum(cal_rfi_mask)
        print(f"  Combined good channels (RFI + cal): {n_good_cal} / {len(vm.freq_mhz)}")

        print("Applying calibration to visibilities...")
        vis_cal = apply_calibration(vm.vis, gains)

        print("Computing calibrated auto power...")
        auto_total_cal, auto_per_ant_cal = compute_auto_power(vis_cal, cal_rfi_mask)
        print(f"  Auto power range: {auto_total_cal.min():.2f} - "
              f"{auto_total_cal.max():.2f}")

        print(f"Computing calibrated cross-only beamformed power "
              f"({vis_cal.shape[0]} times x {len(all_directions)} dirs)...")
        cross_power_cal = compute_cross_beamform_power(
            vis_cal, freqs_hz, vm.positions_enu, all_directions, cal_rfi_mask,
        )
        print("  Done.")

        print("\n*** CALIBRATED ***")
        print_summary(vm.time_unix, auto_total_cal, cross_power_cal, beam_indices)

        output_cal_path = Path(args.output_cal)
        print(f"\nGenerating calibrated diagnostic plot: {output_cal_path}")
        make_diagnostic_plot(
            vm.time_unix, auto_total_cal, auto_per_ant_cal,
            cross_power_cal, beam_indices, sun_alt, sun_az,
            output_cal_path, title_suffix="(Calibrated — SVD gains applied)",
        )
    else:
        print("\nSkipping calibrated analysis (--cal none).")


if __name__ == "__main__":
    main()
