#!/usr/bin/env python3
"""
Check whether astronomical sources pass through beam positions.

Computes the Alt/Az track of a source over a date range at OVRO and checks
which beams (if any) the source passes through. Can load beam positions from
an HDF5 weights file or generate them on the fly.

Built-in sources: Sun, Tau-A (Crab), Cas-A, Cyg-A, Vir-A

Usage:
    # Check Sun over next 14 days against weights file
    python examples/check_source_visibility.py \\
        --source Sun --days 14 \\
        --weights weights_256beam/int8_weights_256beam_feb14_svd.h5

    # Check multiple sources
    python examples/check_source_visibility.py \\
        --source Sun Tau-A Cas-A --days 14 \\
        --weights weights_256beam/int8_weights_256beam_feb14_svd.h5

    # Custom date range
    python examples/check_source_visibility.py \\
        --source Sun --start 2026-02-23 --end 2026-03-09 \\
        --weights weights_256beam/int8_weights_256beam_feb14_svd.h5

    # Generate beams on the fly (no weights file needed)
    python examples/check_source_visibility.py \\
        --source Sun --days 7 --n-beams 256
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
    from astropy.time import Time
    import astropy.units as u
except ImportError:
    print("Error: astropy is required. Install with: pip install astropy",
          file=sys.stderr)
    sys.exit(1)

from bf_weights_generator.config import OVRO_LAT_DEG, OVRO_LON_DEG, OVRO_ALT_M


# ── Known sources ─────────────────────────────────────────────────────────────
KNOWN_SOURCES = {
    "sun":   None,  # special handling via get_sun()
    "tau-a": SkyCoord(ra="05h34m31.94s", dec="+22d00m52.2s"),
    "crab":  SkyCoord(ra="05h34m31.94s", dec="+22d00m52.2s"),  # alias
    "cas-a": SkyCoord(ra="23h23m24.00s", dec="+58d48m54.0s"),
    "cyg-a": SkyCoord(ra="19h59m28.36s", dec="+40d44m02.1s"),
    "vir-a": SkyCoord(ra="12h30m49.42s", dec="+12d23m28.0s"),
    "b0329+54": SkyCoord(ra="03h32m59.4096s", dec="+54d34m43.329s"),
}

OVRO = EarthLocation(
    lat=OVRO_LAT_DEG * u.deg,
    lon=OVRO_LON_DEG * u.deg,
    height=OVRO_ALT_M * u.m,
)


def compute_source_track(source_name, start_utc, end_utc, dt_minutes=5.0):
    """
    Compute Alt/Az track of a source at OVRO.

    Returns
    -------
    times : astropy.time.Time
    alt_deg, az_deg : np.ndarray
        Altitude and azimuth in degrees.
    """
    key = source_name.lower().replace(" ", "-")
    is_sun = key == "sun"

    if not is_sun and key not in KNOWN_SOURCES:
        raise ValueError(
            f"Unknown source: {source_name!r}. "
            f"Known: {', '.join(k for k in KNOWN_SOURCES if k != 'crab')}"
        )

    # Time grid
    t_start = Time(start_utc)
    t_end = Time(end_utc)
    n_steps = max(1, int((t_end - t_start).sec / (dt_minutes * 60)))
    times = t_start + np.linspace(0, (t_end - t_start).sec, n_steps) * u.s

    altaz_frame = AltAz(obstime=times, location=OVRO)

    if is_sun:
        coords = get_sun(times).transform_to(altaz_frame)
    else:
        coords = KNOWN_SOURCES[key].transform_to(altaz_frame)

    return times, coords.alt.deg, coords.az.deg


def load_beams_from_hdf5(filepath):
    """Load beam alt/az and FWHM info from int8 weights HDF5."""
    import h5py

    with h5py.File(filepath, "r") as f:
        alt = f["pointings/alt_deg"][:]
        az = f["pointings/az_deg"][:]
        names = json.loads(f["pointings"].attrs["names"])

        # Get array config for FWHM computation
        positions = f["array_config/positions_enu"][:]
        active = f["array_config/active_mask"][:]

    from bf_weights_generator import compute_beam_fwhm
    active_pos = positions[active]
    fwhm_ew, fwhm_ns = compute_beam_fwhm(active_pos)

    return alt, az, names, fwhm_ew, fwhm_ns


def check_beam_hits(
    source_alt, source_az, beam_alt, beam_az, fwhm_ew, fwhm_ns, times,
):
    """
    Check which beams the source passes through.

    Uses elliptical beam model: a source is "in" a beam if its angular
    distance in the E-W and N-S directions are both within FWHM/2.

    Returns
    -------
    hits : list of dict
        Each dict: beam_idx, beam_name, entry_time, exit_time, min_dist_deg,
                   peak_alt, peak_az
    """
    n_beams = len(beam_alt)
    half_ew = fwhm_ew / 2.0
    half_ns = fwhm_ns / 2.0

    hits = []

    for bi in range(n_beams):
        b_alt = beam_alt[bi]
        b_az = beam_az[bi]

        # Angular separation in alt (N-S) and az (E-W, corrected for alt)
        d_alt = source_alt - b_alt
        d_az = ((source_az - b_az + 180) % 360) - 180  # wrap to [-180, 180]
        cos_alt = np.cos(np.deg2rad(b_alt))
        d_az_sky = d_az * cos_alt  # project to true angle on sky

        # Elliptical distance (normalized)
        r_ew = d_az_sky / half_ew
        r_ns = d_alt / half_ns
        r = np.sqrt(r_ew**2 + r_ns**2)

        in_beam = r < 1.0
        if not np.any(in_beam):
            continue

        # Find contiguous intervals
        indices = np.where(in_beam)[0]
        # Split into groups of consecutive indices
        splits = np.where(np.diff(indices) > 1)[0] + 1
        groups = np.split(indices, splits)

        for grp in groups:
            i_entry = grp[0]
            i_exit = grp[-1]
            i_peak = grp[np.argmin(r[grp])]

            hits.append({
                "beam_idx": bi,
                "entry_time": times[i_entry],
                "exit_time": times[i_exit],
                "min_dist_deg": float(r[i_peak] * max(half_ew, half_ns)),
                "peak_alt": float(source_alt[i_peak]),
                "peak_az": float(source_az[i_peak]),
                "duration_min": float((times[i_exit] - times[i_entry]).sec / 60),
            })

    return hits


def plot_source_track(
    source_name, source_alt, source_az, times,
    beam_alt, beam_az, fwhm_ew, fwhm_ns, hits, output_path,
):
    """Plot source track on Alt/Az with beam positions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # ── Left: Rectangular Alt vs Az ───────────────────────────────────────
    ax = axes[0]

    # Only show source when above horizon
    above = source_alt > 0
    if not np.any(above):
        ax.text(0.5, 0.5, f"{source_name} never rises",
                transform=ax.transAxes, ha='center', fontsize=14)
    else:
        # Beam ellipses
        norm = mcolors.Normalize(vmin=min(beam_alt), vmax=max(beam_alt))
        cmap = plt.cm.viridis

        for i in range(len(beam_alt)):
            b_alt, b_az = beam_alt[i], beam_az[i]
            cos_alt = np.cos(np.deg2rad(b_alt))
            w_az = fwhm_ew / cos_alt if cos_alt > 0.01 else 360.0
            color = cmap(norm(b_alt))
            ell = Ellipse(
                (b_az, b_alt), width=w_az, height=fwhm_ns, angle=0,
                facecolor=color, edgecolor='gray', linewidth=0.5, alpha=0.35,
            )
            ax.add_patch(ell)

        # Highlight hit beams
        hit_beams = set(h["beam_idx"] for h in hits)
        for bi in hit_beams:
            b_alt, b_az = beam_alt[bi], beam_az[bi]
            cos_alt = np.cos(np.deg2rad(b_alt))
            w_az = fwhm_ew / cos_alt if cos_alt > 0.01 else 360.0
            ell = Ellipse(
                (b_az, b_alt), width=w_az, height=fwhm_ns, angle=0,
                facecolor='red', edgecolor='red', linewidth=1.5, alpha=0.5,
            )
            ax.add_patch(ell)

        # Source track (color by time)
        # Split track at az wraps for cleaner lines
        above_idx = np.where(above)[0]
        d_az = np.abs(np.diff(source_az[above_idx]))
        breaks = np.where(d_az > 90)[0] + 1
        segments = np.split(above_idx, breaks)

        for seg in segments:
            if len(seg) < 2:
                continue
            ax.plot(source_az[seg], source_alt[seg], 'k-', linewidth=1.5, alpha=0.7)

        # Mark entry/exit of hit beams
        for h in hits:
            ax.plot(h["peak_az"], h["peak_alt"], 'r*', markersize=10, zorder=10)

    ax.set_xlim(-5, 365)
    ax.set_ylim(0, 95)
    ax.set_xlabel("Azimuth (°)", fontsize=12)
    ax.set_ylabel("Altitude (°)", fontsize=12)
    ax.set_title(f"{source_name} track — Alt vs Az", fontsize=13, fontweight='bold')
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.set_xticklabels(['N\n0', 'NE\n45', 'E\n90', 'SE\n135',
                         'S\n180', 'SW\n225', 'W\n270', 'NW\n315', 'N\n360'])
    ax.grid(True, linestyle='--', alpha=0.3)

    # ── Right: Alt vs Time (24h wrapped) ──────────────────────────────────
    ax2 = axes[1]

    # Convert to hours since midnight UTC of first day
    t0 = times[0]
    hours = np.array([(t - t0).sec / 3600 for t in times])
    # Wrap to 24h for overlay of multiple days
    hours_wrapped = hours % 24.0
    days = (hours / 24).astype(int)
    unique_days = np.unique(days)

    cmap_days = plt.cm.coolwarm
    norm_days = mcolors.Normalize(vmin=unique_days[0], vmax=unique_days[-1])

    for d in unique_days:
        mask = (days == d) & above
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        color = cmap_days(norm_days(d))
        date_str = (t0 + d * 24 * 3600 * u.s).datetime.strftime("%m/%d")
        ax2.plot(hours_wrapped[idx], source_alt[idx],
                 color=color, linewidth=0.8, alpha=0.7, label=date_str)

    # Shade beam altitude range
    if len(beam_alt) > 0:
        ax2.axhspan(min(beam_alt) - fwhm_ns/2, max(beam_alt) + fwhm_ns/2,
                     color='green', alpha=0.1, label='Beam coverage')

    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 95)
    ax2.set_xlabel("UTC Hour", fontsize=12)
    ax2.set_ylabel("Altitude (°)", fontsize=12)
    ax2.set_title(f"{source_name} — Altitude vs Time (daily overlay)",
                  fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Only show legend if few days, otherwise use colorbar
    if len(unique_days) <= 10:
        ax2.legend(fontsize=8, loc='upper right', ncol=2)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap_days, norm=norm_days)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, shrink=0.7, pad=0.02)
        cbar.set_label('Day offset', fontsize=10)

    plt.suptitle(
        f"{source_name} visibility at OVRO — "
        f"{times[0].datetime.strftime('%Y-%m-%d')} to "
        f"{times[-1].datetime.strftime('%Y-%m-%d')}",
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Check if astronomical sources pass through beam positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", nargs="+", required=True,
        help="Source name(s): Sun, Tau-A, Cas-A, Cyg-A, Vir-A",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Number of days from today (alternative to --start/--end)",
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date (YYYY-MM-DD UTC). Default: today.",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (YYYY-MM-DD UTC). Default: start + --days.",
    )
    parser.add_argument(
        "--dt", type=float, default=2.0,
        help="Time step in minutes",
    )

    beam_group = parser.add_argument_group("Beam source (choose one)")
    beam_group.add_argument(
        "--weights", "-w", default=None,
        help="Int8 weights HDF5 file to load beam positions from",
    )
    beam_group.add_argument(
        "--n-beams", type=int, default=None,
        help="Generate beams on the fly (requires --output-layout)",
    )
    beam_group.add_argument(
        "--spacing", choices=["fwhm", "nyquist"], default="fwhm",
        help="Beam spacing when generating on the fly",
    )
    beam_group.add_argument(
        "--output-layout", default="casm_antenna_layout_current.csv",
        help="Antenna layout CSV (for on-the-fly beam generation)",
    )
    beam_group.add_argument(
        "--alt-min", type=float, default=30.0,
        help="Min altitude for on-the-fly beam generation",
    )

    parser.add_argument(
        "--output-dir", "-d", default=".",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation (text output only)",
    )
    args = parser.parse_args()

    # ── Resolve date range ────────────────────────────────────────────────
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(
            tzinfo=timezone.utc)
    else:
        start_dt = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0)

    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(
            tzinfo=timezone.utc)
    elif args.days:
        end_dt = start_dt + timedelta(days=args.days)
    else:
        end_dt = start_dt + timedelta(days=14)

    print(f"Date range: {start_dt.strftime('%Y-%m-%d')} to "
          f"{end_dt.strftime('%Y-%m-%d')} ({(end_dt - start_dt).days} days)")

    # ── Load or generate beams ────────────────────────────────────────────
    if args.weights:
        print(f"Loading beams from: {args.weights}")
        beam_alt, beam_az, beam_names, fwhm_ew, fwhm_ns = \
            load_beams_from_hdf5(args.weights)
        print(f"  {len(beam_alt)} beams, FWHM: {fwhm_ew:.2f}° × {fwhm_ns:.2f}°")
    elif args.n_beams:
        from bf_weights_generator import Array64Config, compute_beam_fwhm
        from bf_weights_generator.weights import generate_beam_grid_altaz

        csv_path = args.output_layout
        p = Path(csv_path)
        if not p.exists():
            p = Path(__file__).parent.parent / csv_path
        arr = Array64Config.from_csv(str(p))
        pos = arr.active_positions
        fwhm_ew, fwhm_ns = compute_beam_fwhm(pos)
        sp_ew = fwhm_ew / (2.0 if args.spacing == "nyquist" else 1.0)
        sp_ns = fwhm_ns / (2.0 if args.spacing == "nyquist" else 1.0)
        beams = generate_beam_grid_altaz(
            alt_min_deg=args.alt_min, spacing_ew_deg=sp_ew, spacing_ns_deg=sp_ns,
        )
        if len(beams) > args.n_beams:
            beams.sort(key=lambda b: (-b.alt_deg, b.az_deg))
            beams = beams[:args.n_beams]
        beam_alt = np.array([b.alt_deg for b in beams])
        beam_az = np.array([b.az_deg for b in beams])
        beam_names = [f"beam_{i:03d}" for i in range(len(beams))]
        print(f"  Generated {len(beams)} beams, "
              f"FWHM: {fwhm_ew:.2f}° × {fwhm_ns:.2f}°")
    else:
        parser.error("Must specify --weights or --n-beams")

    beam_alt_range = (float(beam_alt.min()), float(beam_alt.max()))
    print(f"  Beam alt range: {beam_alt_range[0]:.1f}° – {beam_alt_range[1]:.1f}°")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each source ───────────────────────────────────────────────
    for source_name in args.source:
        print(f"\n{'='*60}")
        print(f"Source: {source_name}")
        print(f"{'='*60}")

        times, src_alt, src_az = compute_source_track(
            source_name, start_dt, end_dt, dt_minutes=args.dt,
        )

        # Basic visibility info
        above_horizon = src_alt > 0
        if np.any(above_horizon):
            alt_max = float(src_alt[above_horizon].max())
            alt_min_vis = float(src_alt[above_horizon].min())
            print(f"  Visible alt range: {alt_min_vis:.1f}° – {alt_max:.1f}° "
                  f"(max altitude)")
        else:
            print(f"  {source_name} never rises above horizon in this period.")
            continue

        # Check if source altitude overlaps with beam coverage
        if alt_max < beam_alt_range[0] - fwhm_ns / 2:
            print(f"  Source max alt ({alt_max:.1f}°) is below beam coverage "
                  f"({beam_alt_range[0]:.1f}°). No hits possible.")
        elif alt_min_vis > beam_alt_range[1] + fwhm_ns / 2:
            print(f"  Source min visible alt ({alt_min_vis:.1f}°) is above beam "
                  f"coverage ({beam_alt_range[1]:.1f}°). No hits possible.")

        # Check beam hits
        hits = check_beam_hits(
            src_alt, src_az, beam_alt, beam_az, fwhm_ew, fwhm_ns, times,
        )

        if hits:
            print(f"\n  {len(hits)} beam crossing(s) found:")
            print(f"  {'Beam':<10s} {'Entry (UTC)':<22s} {'Exit (UTC)':<22s} "
                  f"{'Duration':>10s} {'Peak Alt':>9s} {'Peak Az':>9s}")
            print(f"  {'-'*10} {'-'*22} {'-'*22} {'-'*10} {'-'*9} {'-'*9}")
            for h in sorted(hits, key=lambda x: x["entry_time"].datetime):
                entry = h["entry_time"].datetime.strftime("%Y-%m-%d %H:%M")
                exit_ = h["exit_time"].datetime.strftime("%Y-%m-%d %H:%M")
                dur = f"{h['duration_min']:.0f} min"
                print(f"  beam_{h['beam_idx']:03d}   {entry:<22s} {exit_:<22s} "
                      f"{dur:>10s} {h['peak_alt']:>8.1f}° {h['peak_az']:>8.1f}°")

            # Summary by unique beams hit
            unique_beams = sorted(set(h["beam_idx"] for h in hits))
            print(f"\n  Unique beams hit: {len(unique_beams)} / {len(beam_alt)}")
            print(f"  Beam indices: {unique_beams}")
        else:
            print(f"\n  No beam crossings found.")

        # Plot
        if not args.no_plot:
            plot_file = out_dir / f"source_track_{source_name.lower().replace(' ', '_')}.png"
            plot_source_track(
                source_name, src_alt, src_az, times,
                beam_alt, beam_az, fwhm_ew, fwhm_ns, hits, plot_file,
            )


if __name__ == "__main__":
    main()
