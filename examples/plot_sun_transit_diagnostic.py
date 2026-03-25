#!/usr/bin/env python3
"""
Diagnostic plot for the 2026-03-12 sun transit coherent beamforming test.

Produces a 3-panel figure:
  1. Zenithal sky map with all 512 beam FWHM ellipses, 7 dumped beams
     highlighted, and the sun track overlaid with time ticks.
  2. Angular distance from sun to each dumped beam center vs local time.
  3. Expected Gaussian beam response vs local time.

Usage:
    python examples/plot_sun_transit_diagnostic.py \\
        weights_512beam/int8_weights_512beam_feb19_svd.h5 \\
        -o examples/sun_transit_diagnostic_20260312.png
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import astropy.units as u
    from astropy.time import Time
except ImportError:
    print("Error: astropy is required. Install with: pip install astropy",
          file=sys.stderr)
    sys.exit(1)

from examples.check_source_visibility import (
    compute_source_track,
    load_beams_from_hdf5,
)

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
DATE = "2026-03-12"
TIME_START_PST = "09:00"
TIME_END_PST = "15:00"


def angular_distance_to_beam(
    sun_alt, sun_az, beam_alt_deg, beam_az_deg, fwhm_ew, fwhm_ns,
):
    """
    Compute angular distance from the sun to a beam center over time,
    using the same elliptical metric as check_beam_hits.

    Returns angular distance in degrees (using the larger FWHM as scale).
    """
    d_alt = sun_alt - beam_alt_deg
    d_az = ((sun_az - beam_az_deg + 180) % 360) - 180
    cos_alt = np.cos(np.deg2rad(beam_alt_deg))
    d_az_sky = d_az * cos_alt

    # Elliptical normalized distance
    r_ew = d_az_sky / (fwhm_ew / 2.0)
    r_ns = d_alt / (fwhm_ns / 2.0)
    r_norm = np.sqrt(r_ew**2 + r_ns**2)

    # Convert back to degrees using the larger half-FWHM as scale
    r_deg = r_norm * max(fwhm_ew, fwhm_ns) / 2.0
    return r_deg, r_norm


def gaussian_beam_response(r_norm):
    """Expected Gaussian beam response: exp(-4 ln2 * (r/FWHM)^2).

    r_norm is already in units of FWHM/2 (half-width), so r_norm=1 means
    the source is at the FWHM edge. The Gaussian is:
        exp(-4 ln2 * (r / FWHM)^2)
    Since r_norm = r / (FWHM/2), we have r/FWHM = r_norm/2, so:
        exp(-4 ln2 * (r_norm/2)^2) = exp(-ln2 * r_norm^2)
    """
    return np.exp(-np.log(2) * r_norm**2)


def make_diagnostic_plot(
    beam_alt, beam_az, fwhm_ew, fwhm_ns,
    sun_alt, sun_az, times, output_path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.dates as mdates

    fig, (ax_sky, ax_dist, ax_resp) = plt.subplots(
        1, 3, figsize=(30, 10),
    )

    # ── Panel 1: Zenithal sky map ────────────────────────────────────────
    n_beams = len(beam_alt)
    za = 90.0 - beam_alt
    az_rad = np.deg2rad(beam_az)
    xs = za * np.sin(az_rad)
    ys = za * np.cos(az_rad)

    theta = np.linspace(0, 2 * np.pi, 256)

    # Zoom to dumped beams region with some padding
    dumped_xs = xs[DUMPED_BEAMS]
    dumped_ys = ys[DUMPED_BEAMS]

    # Sun track projection
    sun_above = sun_alt > 0
    sun_za = 90.0 - sun_alt
    sun_az_rad = np.deg2rad(sun_az)
    sun_xs = sun_za * np.sin(sun_az_rad)
    sun_ys = sun_za * np.cos(sun_az_rad)

    # Determine plot limits from dumped beams + sun track
    all_x = np.concatenate([dumped_xs, sun_xs[sun_above]])
    all_y = np.concatenate([dumped_ys, sun_ys[sun_above]])
    margin = max(fwhm_ew, fwhm_ns) * 2.5
    x_cen = (all_x.min() + all_x.max()) / 2
    y_cen = (all_y.min() + all_y.max()) / 2
    span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 + margin

    # Altitude circles
    for alt in [15, 30, 45, 60, 75, 80, 85]:
        r = 90 - alt
        circle_x = r * np.cos(theta)
        circle_y = r * np.sin(theta)
        ax_sky.plot(circle_x, circle_y,
                    color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
        # Label only if visible in the plot region
        if abs(-r - y_cen) < span:
            ax_sky.text(0.4, -r - 0.3, f"{alt}\u00b0", ha="left", va="top",
                        fontsize=7, color="gray", alpha=0.5)

    # Azimuth lines
    for az in np.arange(0, 360, 45):
        az_r = np.deg2rad(az)
        ax_sky.plot([0, 70 * np.sin(az_r)], [0, 70 * np.cos(az_r)],
                    color="gray", linewidth=0.5, linestyle="--", alpha=0.3)

    # All beam ellipses (light gray)
    for i in range(n_beams):
        x, y = xs[i], ys[i]
        # Skip beams far outside view
        if abs(x - x_cen) > span * 1.3 or abs(y - y_cen) > span * 1.3:
            continue
        if za[i] < fwhm_ns:
            w = h = max(fwhm_ew, fwhm_ns)
            angle = 0
        else:
            w, h = fwhm_ew, fwhm_ns
            angle = -beam_az[i]
        ell = Ellipse((x, y), width=w, height=h, angle=angle,
                      facecolor="lightgray", edgecolor="gray",
                      linewidth=0.3, alpha=0.25)
        ax_sky.add_patch(ell)

    # Highlighted dumped beam ellipses
    for bi in DUMPED_BEAMS:
        x, y = xs[bi], ys[bi]
        color = BEAM_COLORS[bi]
        if za[bi] < fwhm_ns:
            w = h = max(fwhm_ew, fwhm_ns)
            angle = 0
        else:
            w, h = fwhm_ew, fwhm_ns
            angle = -beam_az[bi]
        ell = Ellipse((x, y), width=w, height=h, angle=angle,
                      facecolor=color, edgecolor=color,
                      linewidth=2.0, alpha=0.5)
        ax_sky.add_patch(ell)
        ax_sky.annotate(
            str(bi), (x, y), fontsize=8, ha="center", va="center",
            fontweight="bold", color="k",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7),
            zorder=9,
        )

    # Sun track
    above_idx = np.where(sun_above)[0]
    if len(above_idx) > 1:
        d_az_diff = np.abs(np.diff(sun_az[above_idx]))
        breaks = np.where(d_az_diff > 90)[0] + 1
        segments = np.split(above_idx, breaks)
    else:
        segments = [above_idx]

    for seg in segments:
        if len(seg) < 2:
            continue
        ax_sky.plot(sun_xs[seg], sun_ys[seg], color="gold",
                    linewidth=3.0, alpha=0.9, zorder=5)

    # Time tick marks every 30 min
    dt_list = [t.to_datetime(timezone=PST) for t in times]
    for i, t in enumerate(dt_list):
        if not sun_above[i]:
            continue
        if t.minute == 0 or t.minute == 30:
            # Only label on the hour
            ax_sky.plot(sun_xs[i], sun_ys[i], "o", color="gold",
                        markersize=4, markeredgecolor="k",
                        markeredgewidth=0.5, zorder=6)
            if t.minute == 0:
                ax_sky.annotate(
                    t.strftime("%H:%M"), (sun_xs[i], sun_ys[i]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="k", fontweight="bold", zorder=7,
                )

    # Mark sun at transit (closest to az=180, max alt)
    transit_mask = sun_above & (np.abs(sun_az - 180) < 5)
    if np.any(transit_mask):
        i_transit = np.where(transit_mask)[0][np.argmax(sun_alt[transit_mask])]
        ax_sky.plot(sun_xs[i_transit], sun_ys[i_transit], "*",
                    color="gold", markersize=18, markeredgecolor="k",
                    markeredgewidth=1.0, zorder=8)

    ax_sky.set_xlim(x_cen - span, x_cen + span)
    ax_sky.set_ylim(y_cen - span, y_cen + span)
    ax_sky.set_aspect("equal")
    ax_sky.set_xlabel("\u2190 West     East \u2192", fontsize=11)
    ax_sky.set_ylabel("\u2190 South     North \u2192", fontsize=11)
    ax_sky.set_title("Beam Grid", fontsize=13,
                     fontweight="bold")

    info = (f"{n_beams} beams | "
            f"FWHM: {fwhm_ew:.2f}\u00b0 (E-W) \u00d7 {fwhm_ns:.2f}\u00b0 (N-S)")
    props = dict(boxstyle="round", facecolor="none", alpha=0.85)
    ax_sky.text(0.02, 0.02, info, transform=ax_sky.transAxes, fontsize=9,
                verticalalignment="bottom", bbox=props, family="monospace")

    # ── Panel 2: Angular distance from sun vs time ───────────────────────
    dt_mpl = [t.to_datetime(timezone=PST) for t in times]
    fwhm_radius = max(fwhm_ew, fwhm_ns) / 2.0

    for bi in DUMPED_BEAMS:
        r_deg, _ = angular_distance_to_beam(
            sun_alt, sun_az, beam_alt[bi], beam_az[bi], fwhm_ew, fwhm_ns,
        )
        color = BEAM_COLORS[bi]
        ax_dist.plot(dt_mpl, r_deg, color=color, linewidth=1.5,
                     label=f"Beam {bi}", alpha=0.9)

    ax_dist.axhline(fwhm_radius, color="k", linestyle="--", linewidth=1.0,
                    alpha=0.6, label=f"FWHM/2 = {fwhm_radius:.2f}\u00b0")
    ax_dist.set_ylim(0, 50)
    ax_dist.set_xlabel(f"Time (PST)", fontsize=11)
    ax_dist.set_ylabel("Angular distance from Sun (\u00b0)", fontsize=11)
    ax_dist.set_title("Angular Distance: Sun \u2194 Beam Center",
                      fontsize=13, fontweight="bold")
    ax_dist.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=PST))
    ax_dist.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax_dist.legend(fontsize=8, loc="upper right", ncol=2)
    ax_dist.grid(True, linestyle="--", alpha=0.3)
    fig.autofmt_xdate(rotation=45)

    # ── Panel 3: Expected Gaussian beam response vs time ─────────────────
    for bi in DUMPED_BEAMS:
        _, r_norm = angular_distance_to_beam(
            sun_alt, sun_az, beam_alt[bi], beam_az[bi], fwhm_ew, fwhm_ns,
        )
        response = gaussian_beam_response(r_norm)
        color = BEAM_COLORS[bi]
        ax_resp.plot(dt_mpl, response, color=color, linewidth=1.5,
                     label=f"Beam {bi}", alpha=0.9)

    ax_resp.set_ylim(-0.02, 1.05)
    ax_resp.set_xlabel(f"Time (PST)", fontsize=11)
    ax_resp.set_ylabel("Expected normalized response", fontsize=11)
    ax_resp.set_title("Expected Gaussian Beam Response to Sun",
                      fontsize=13, fontweight="bold")
    ax_resp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=PST))
    ax_resp.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax_resp.legend(fontsize=8, loc="upper right", ncol=2)
    ax_resp.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(
        f"Sun Transit Diagnostic \u2014 {DATE} (OVRO)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Sun transit beamforming diagnostic plot (2026-03-12).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "weights", help="HDF5 weights file (512-beam)",
    )
    parser.add_argument(
        "-o", "--output", default="examples/sun_transit_diagnostic_20260312.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--dt", type=float, default=2.0,
        help="Time step in minutes",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    # Load beams
    print(f"Loading beams from: {weights_path}")
    beam_alt, beam_az, beam_names, fwhm_ew, fwhm_ns = \
        load_beams_from_hdf5(str(weights_path))
    print(f"  {len(beam_alt)} beams, FWHM: {fwhm_ew:.2f}\u00b0 \u00d7 "
          f"{fwhm_ns:.2f}\u00b0")

    # Verify dumped beams are valid
    for bi in DUMPED_BEAMS:
        if bi >= len(beam_alt):
            print(f"Error: beam index {bi} out of range "
                  f"(max {len(beam_alt)-1})", file=sys.stderr)
            sys.exit(1)

    print(f"\nDumped beams ({len(DUMPED_BEAMS)}):")
    for bi in DUMPED_BEAMS:
        print(f"  Beam {bi:3d}: alt={beam_alt[bi]:.2f}\u00b0, "
              f"az={beam_az[bi]:.2f}\u00b0")

    # Compute sun track: 09:00-15:00 PST on 2026-03-12
    h_start, m_start = map(int, TIME_START_PST.split(":"))
    h_end, m_end = map(int, TIME_END_PST.split(":"))

    date = datetime.strptime(DATE, "%Y-%m-%d")
    start_local = date.replace(hour=h_start, minute=m_start, tzinfo=PST)
    end_local = date.replace(hour=h_end, minute=m_end, tzinfo=PST)

    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    print(f"\nSun track: {TIME_START_PST}-{TIME_END_PST} PST "
          f"({start_utc.strftime('%H:%M')}-{end_utc.strftime('%H:%M')} UTC)")

    times, sun_alt, sun_az = compute_source_track(
        "Sun", start_utc, end_utc, dt_minutes=args.dt,
    )

    # Find transit
    above = sun_alt > 0
    if np.any(above):
        i_max = np.argmax(sun_alt)
        t_transit = times[i_max].to_datetime(timezone=PST)
        print(f"  Sun transit: alt={sun_alt[i_max]:.2f}\u00b0, "
              f"az={sun_az[i_max]:.2f}\u00b0 at {t_transit.strftime('%H:%M')} PST")

    # Generate plot
    output_path = Path(args.output)
    make_diagnostic_plot(
        beam_alt, beam_az, fwhm_ew, fwhm_ns,
        sun_alt, sun_az, times, output_path,
    )


if __name__ == "__main__":
    main()
